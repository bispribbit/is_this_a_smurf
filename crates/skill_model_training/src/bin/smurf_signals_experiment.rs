#![expect(
    clippy::indexing_slicing,
    reason = "slot arrays indexed by slots below TOTAL_SLOTS, playlist arrays by playlist index"
)]

//! Can the hand-written smurf references, or plays-like models per stat group, improve the
//! smurf flag?
//!
//! 1. **Training rules**: drop the training labels that match one reference rule
//!    ([`skill_model_training::smurf_references`]) on top of label cleaning, for the match
//!    model (rank metrics) and the plays-like model (flag). A rule cannot grade a model it
//!    trained, so each variant is read on the other rule.
//! 2. **Stat groups**: one plays-like model per group of stats (aerial, ground and flips,
//!    speed and boost, all opponent-independent; and the named mechanics: speed flips, flip
//!    resets, dribbles, double taps, power shots, 50/50s, bumps, demos, ...). A smurf should stand out in several groups, a strong
//!    player's one good game in one. Flags need two or three groups, or the full model plus
//!    two groups.
//! 3. **Weighted groups**: the group margins averaged, each weighted by how well its group
//!    estimates rank (1 / RMSE² on the early-stopping split).
//!
//! 4. **Mechanics above the lobby tier**: count the named mechanics a player does more than
//!    the lobby tier's 90th or 95th percentile ([`skill_model::mechanics`]); flag two to
//!    five, alone or on top of the weighted groups.
//!
//! Every variant flags the same number of players per playlist (the shipped flag's count,
//! then 1.5× that), so the reference columns compare recall at equal flag rates. Margins are
//! in divisions over the other players, each scored half by the model and half by the
//! shipped bundle's whole-match model, as the shipped flag does.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin smurf_signals_experiment

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use feature_extractor::{MATCH_STAT_COUNT, TOTAL_SLOTS};
use skill_model::coaching::tier_index;
use skill_model::mechanics::MechanicTables;
use skill_model::{
    PlaysLikeModel, STAT_GROUPS, SkillModelBundle, TabularLayout, TabularSkillModel,
    smurf_margin_over_others_tiers,
};
use skill_model_training::dataset::{Lobby, Partition, evaluate, read_lobbies, train_model};
use skill_model_training::evaluation::{
    MINIMUM_PLAYERS_PER_LOBBY, PlayerPrediction, compute_lobby_metrics_with_minimum,
};
use skill_model_training::gradient_boosting::GradientBoostingConfig;
use skill_model_training::label_cleaning::{
    CLEAN_MARGIN_MMR, SlotScores, clean_labels_in_rounds, early_stopping_rmse,
    train_plays_like_model,
};
use skill_model_training::mechanic_table::build_mechanic_tables;
use skill_model_training::smurf_references::{
    HIGH_RANK_TIER, MECHANIC_STATS, as_tiers, is_dominant_smurf, is_mechanical_smurf, median,
    reference_mechanics,
};
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(about = "Smurf flag from reference-rule training drops and per-group models")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

    /// Shipped bundle, for the whole-match model the others are blended with.
    #[arg(long, default_value = "data/skill_model.bin")]
    bundle: PathBuf,
}

/// High quantiles a mechanic rate must beat, within the lobby's tier.
const MECHANIC_QUANTILES: [f32; 2] = [0.9, 0.95];

/// One playlist size and its report name.
struct Playlist {
    players_per_team: usize,
    name: &'static str,
}

const PLAYLISTS: [Playlist; 3] = [
    Playlist {
        players_per_team: 1,
        name: "duels",
    },
    Playlist {
        players_per_team: 2,
        name: "doubles",
    },
    Playlist {
        players_per_team: 3,
        name: "standard",
    },
];

/// Extra training labels dropped on top of label cleaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainingRule {
    None,
    Mechanical,
    Dominant,
}

impl TrainingRule {
    const fn name(self) -> &'static str {
        match self {
            Self::None => "cleaning only",
            Self::Mechanical => "drop mechanical",
            Self::Dominant => "drop dominant",
        }
    }
}

const TRAINING_RULES: [TrainingRule; 3] = [
    TrainingRule::None,
    TrainingRule::Mechanical,
    TrainingRule::Dominant,
];

/// Median Champion I values of [`MECHANIC_STATS`], per playlist (index = players per team − 1).
struct MechanicReferences {
    by_playlist: [[f32; MECHANIC_STATS.len()]; 3],
}

impl MechanicReferences {
    fn new(lobbies: &[Lobby]) -> Self {
        Self {
            by_playlist: core::array::from_fn(|index| {
                let playlist: Vec<Lobby> = lobbies
                    .iter()
                    .filter(|lobby| lobby.players_per_team() == Some(index + 1))
                    .cloned()
                    .collect();
                reference_mechanics(&playlist)
            }),
        }
    }

    fn for_lobby(&self, lobby: &Lobby) -> &[f32; MECHANIC_STATS.len()] {
        &self.by_playlist[playlist_index(lobby)]
    }
}

fn playlist_index(lobby: &Lobby) -> usize {
    lobby.players_per_team().unwrap_or(3).clamp(1, 3) - 1
}

fn matches_rule(
    rule: TrainingRule,
    lobby: &Lobby,
    slot: usize,
    references: &MechanicReferences,
) -> bool {
    match rule {
        TrainingRule::None => false,
        TrainingRule::Mechanical => is_mechanical_smurf(lobby, slot, references.for_lobby(lobby)),
        TrainingRule::Dominant => is_dominant_smurf(lobby, slot),
    }
}

/// Training lobbies with a rule's labels dropped.
struct RuleDrops {
    lobbies: Vec<Lobby>,
    dropped_by_playlist: [usize; 3],
}

fn drop_rule_labels(
    lobbies: &[Lobby],
    rule: TrainingRule,
    references: &MechanicReferences,
) -> RuleDrops {
    let mut drops = RuleDrops {
        lobbies: lobbies.to_vec(),
        dropped_by_playlist: [0; 3],
    };
    for lobby in &mut drops.lobbies {
        if lobby.partition == Partition::Evaluation {
            continue;
        }
        let matching: Vec<usize> = lobby
            .labelled_slots()
            .filter(|&slot| matches_rule(rule, lobby, slot, references))
            .collect();
        drops.dropped_by_playlist[playlist_index(lobby)] += matching.len();
        for slot in matching {
            lobby.targets[slot] = 0.0;
        }
    }
    drops
}

/// Match-model rank metrics per playlist on the evaluation split.
fn print_playlist_metrics(lobbies: &[Lobby], model: &TabularSkillModel) {
    for playlist in &PLAYLISTS {
        let playlist_lobbies: Vec<Lobby> = lobbies
            .iter()
            .filter(|lobby| lobby.players_per_team() == Some(playlist.players_per_team))
            .cloned()
            .collect();
        let predictions: Vec<PlayerPrediction> = evaluate(&playlist_lobbies, model)
            .iter()
            .map(|scored| scored.prediction)
            .collect();
        let metrics = compute_lobby_metrics_with_minimum(
            &predictions,
            MINIMUM_PLAYERS_PER_LOBBY.min(playlist.players_per_team * 2),
        );
        println!(
            "    {:<9} within_r={:.3} ±{:.3}  concordance={:.3}  lobby_rmse={:.0}  player_rmse={:.0}",
            playlist.name,
            metrics.within_r,
            metrics.within_r_standard_error,
            metrics.concordance,
            metrics.lobby_rmse,
            metrics.player_rmse,
        );
    }
}

/// A model's plays-like divisions on every evaluation replay.
struct ModelTiers {
    by_replay: HashMap<Uuid, SlotScores>,
}

impl ModelTiers {
    fn new(evaluation: &[&Lobby], model: &PlaysLikeModel) -> Self {
        Self {
            by_replay: evaluation
                .iter()
                .map(|lobby| {
                    (
                        lobby.replay_id,
                        as_tiers(model.predict_stats(&lobby.stat_references())),
                    )
                })
                .collect(),
        }
    }
}

/// Divisions over the median of the other players, each scored half by `tiers` and half by
/// the whole-match model.
fn margin_over_others(tiers: &SlotScores, match_tiers: &SlotScores, slot: usize) -> Option<f32> {
    let own = tiers[slot]?;
    let mut others: Vec<f32> = (0..TOTAL_SLOTS)
        .filter(|&other| other != slot)
        .filter_map(|other| {
            let tier = tiers[other]?;
            Some(match_tiers[other].map_or(tier, |match_tier| f32::midpoint(tier, match_tier)))
        })
        .collect();
    (!others.is_empty()).then(|| own - median(&mut others))
}

/// One evaluation player.
struct EvaluatedPlayer {
    label: f32,
    mechanical: bool,
    dominant: bool,
    /// Margin per model, in the order the models were passed.
    margins: Vec<f32>,
    /// Per mechanic table: mechanics above the lobby tier's high quantile.
    mechanics_above: Vec<usize>,
}

fn evaluated_players(
    lobbies: &[&Lobby],
    models: &[&ModelTiers],
    match_tiers: &HashMap<Uuid, SlotScores>,
    references: &MechanicReferences,
    mechanic_tables: &[MechanicTables],
) -> Vec<EvaluatedPlayer> {
    let mut players = Vec::new();
    for lobby in lobbies {
        let Some(lobby_match_tiers) = match_tiers.get(&lobby.replay_id) else {
            continue;
        };
        let lobby_tiers: Vec<&SlotScores> = models
            .iter()
            .filter_map(|model| model.by_replay.get(&lobby.replay_id))
            .collect();
        if lobby_tiers.len() != models.len() {
            continue;
        }
        for slot in 0..TOTAL_SLOTS {
            let margins: Option<Vec<f32>> = lobby_tiers
                .iter()
                .map(|tiers| margin_over_others(tiers, lobby_match_tiers, slot))
                .collect();
            let Some(margins) = margins else {
                continue;
            };
            let Some(stats) = lobby.stats[slot] else {
                continue;
            };
            let mut others: Vec<f32> = (0..TOTAL_SLOTS)
                .filter(|&other| other != slot)
                .filter_map(|other| lobby_match_tiers[other])
                .collect();
            let lobby_tier = median(&mut others).round().max(0.0) as usize;
            let players_per_team = lobby.players_per_team().unwrap_or(3);
            players.push(EvaluatedPlayer {
                label: lobby.targets[slot],
                mechanical: is_mechanical_smurf(lobby, slot, references.for_lobby(lobby)),
                dominant: is_dominant_smurf(lobby, slot),
                margins,
                mechanics_above: mechanic_tables
                    .iter()
                    .map(|tables| {
                        tables
                            .for_players_per_team(players_per_team)
                            .map_or(0, |table| table.above_tier(&stats, lobby_tier).len())
                    })
                    .collect(),
            });
        }
    }
    players
}

/// One flag rule's row: how many it flags and which references it catches.
fn print_fixed_rule(players: &[EvaluatedPlayer], name: &str, flagged: &[bool]) {
    let picked: Vec<&EvaluatedPlayer> = players
        .iter()
        .zip(flagged)
        .filter(|(_, flagged)| **flagged)
        .map(|(player, _)| player)
        .collect();
    let high_rank = picked
        .iter()
        .filter(|player| player.label > 0.0 && tier_index(player.label) >= HIGH_RANK_TIER)
        .count();
    println!(
        "      {name:<40} {:>6.1}% {:>7}/{:<4} {:>7}/{:<4} {:>9.0}%",
        100.0 * picked.len() as f64 / players.len().max(1) as f64,
        picked.iter().filter(|player| player.mechanical).count(),
        players.iter().filter(|player| player.mechanical).count(),
        picked.iter().filter(|player| player.dominant).count(),
        players.iter().filter(|player| player.dominant).count(),
        100.0 * high_rank as f64 / picked.len().max(1) as f64,
    );
}

/// The `count` highest scores, as flags.
fn top(scores: &[f32], count: usize) -> Vec<bool> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
    let mut flagged = vec![false; scores.len()];
    for index in order.into_iter().take(count) {
        flagged[index] = true;
    }
    flagged
}

/// A way of ranking players for a flag: higher scores are flagged first.
struct Selection {
    name: String,
    scores: Vec<f32>,
}

/// Each selection flags its `flagged_count` highest-scoring players.
fn print_equal_rate(players: &[EvaluatedPlayer], selections: &[Selection], flagged_count: usize) {
    let mechanical = players.iter().filter(|player| player.mechanical).count();
    let dominant = players.iter().filter(|player| player.dominant).count();
    println!(
        "      flagging {flagged_count} of {} players ({:.1}%)",
        players.len(),
        100.0 * flagged_count as f64 / players.len().max(1) as f64
    );
    println!(
        "      {:<26} {:>12} {:>12} {:>10}",
        "", "mechanical", "dominant", "high-rank"
    );
    for selection in selections {
        let mut order: Vec<usize> = (0..players.len()).collect();
        order.sort_by(|&left, &right| selection.scores[right].total_cmp(&selection.scores[left]));
        let flagged: Vec<&EvaluatedPlayer> = order
            .into_iter()
            .take(flagged_count)
            .map(|index| &players[index])
            .collect();
        let high_rank = flagged
            .iter()
            .filter(|player| player.label > 0.0 && tier_index(player.label) >= HIGH_RANK_TIER)
            .count();
        println!(
            "      {:<26} {:>7}/{:<4} {:>7}/{:<4} {:>9.0}%",
            selection.name,
            flagged.iter().filter(|player| player.mechanical).count(),
            mechanical,
            flagged.iter().filter(|player| player.dominant).count(),
            dominant,
            100.0 * high_rank as f64 / flagged.len().max(1) as f64,
        );
    }
}

/// The `rank`-th largest of `values` (0 = largest).
fn nth_largest(values: &[f32], rank: usize) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    sorted.get(rank).copied().unwrap_or(f32::NEG_INFINITY)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = GradientBoostingConfig::default();
    let lobbies = read_lobbies(&args.match_stats)?;
    let bundle = SkillModelBundle::from_bytes(
        &std::fs::read(&args.bundle)
            .with_context(|| format!("reading {}", args.bundle.display()))?,
    )
    .map_err(|error| anyhow::anyhow!("decoding bundle: {error}"))?;
    let references = MechanicReferences::new(&lobbies);
    let match_layout = TabularLayout {
        own: true,
        deviation: true,
        lobby_mean: true,
        team_context: true,
        stats: (0..MATCH_STAT_COUNT).collect(),
    };

    println!("== label cleaning, as train does (one round, then a second for plays-like)");
    let cleaned_once = clean_labels_in_rounds(
        &lobbies,
        &PlaysLikeModel::own_stats_layout(),
        &config,
        1,
        CLEAN_MARGIN_MMR,
    );
    let cleaned_twice = clean_labels_in_rounds(
        &cleaned_once.lobbies,
        &PlaysLikeModel::own_stats_layout(),
        &config,
        1,
        CLEAN_MARGIN_MMR,
    );

    // Built from labels cleaned once: smurfs are exactly the high tail at low tiers.
    let mechanic_tables: Vec<MechanicTables> = MECHANIC_QUANTILES
        .iter()
        .map(|&quantile| build_mechanic_tables(&cleaned_once.lobbies, quantile))
        .collect();

    let evaluation: Vec<&Lobby> = lobbies
        .iter()
        .filter(|lobby| lobby.partition == Partition::Evaluation)
        .collect();
    let match_tiers: HashMap<Uuid, SlotScores> = evaluation
        .iter()
        .map(|lobby| {
            (
                lobby.replay_id,
                as_tiers(bundle.match_model.predict_stats(&lobby.stat_references())),
            )
        })
        .collect();

    let mut rule_tiers = Vec::new();
    for rule in TRAINING_RULES {
        println!("== {}", rule.name());
        let match_drops = drop_rule_labels(&cleaned_once.lobbies, rule, &references);
        println!(
            "  rule dropped (duels, doubles, standard): {:?} more training labels",
            match_drops.dropped_by_playlist
        );
        let match_model = train_model(&match_drops.lobbies, &match_layout, &config);
        println!("  match model:");
        print_playlist_metrics(&lobbies, &match_model);
        let plays_like_drops = drop_rule_labels(&cleaned_twice.lobbies, rule, &references);
        let plays_like_model = train_plays_like_model(
            &plays_like_drops.lobbies,
            &PlaysLikeModel::opponent_independent_layout(),
            &config,
        );
        rule_tiers.push(ModelTiers::new(&evaluation, &plays_like_model));
    }

    println!("== stat groups");
    let mut group_tiers = Vec::new();
    let mut group_weights = Vec::new();
    for group in &STAT_GROUPS {
        let model = train_plays_like_model(&cleaned_twice.lobbies, &group.layout(), &config);
        let rmse = early_stopping_rmse(&cleaned_twice.lobbies, &model);
        println!(
            "  {:<16} {} stats  early-stopping rmse={rmse:.0}",
            group.name,
            group.stats.len()
        );
        group_weights.push((1.0 / (rmse * rmse)) as f32);
        group_tiers.push(ModelTiers::new(&evaluation, &model));
    }

    let models: Vec<&ModelTiers> = rule_tiers.iter().chain(&group_tiers).collect();
    let group_range = TRAINING_RULES.len()..models.len();
    println!("== evaluation split, every variant flagging the same number of players");
    for playlist in &PLAYLISTS {
        let playlist_lobbies: Vec<&Lobby> = evaluation
            .iter()
            .copied()
            .filter(|lobby| lobby.players_per_team() == Some(playlist.players_per_team))
            .collect();
        let players = evaluated_players(
            &playlist_lobbies,
            &models,
            &match_tiers,
            &references,
            &mechanic_tables,
        );
        let shipped_margin = smurf_margin_over_others_tiers(playlist.players_per_team);
        let shipped_count = players
            .iter()
            .filter(|player| player.margins[0] > shipped_margin)
            .count();

        let by_margin = |name: &str, model: usize| Selection {
            name: name.to_owned(),
            scores: players.iter().map(|player| player.margins[model]).collect(),
        };
        let by_groups = |name: &str, score: &dyn Fn(&[f32], f32) -> f32| Selection {
            name: name.to_owned(),
            scores: players
                .iter()
                .map(|player| score(&player.margins[group_range.clone()], player.margins[0]))
                .collect(),
        };
        // Weighs as many groups as it is given: the first three are the movement groups.
        let weighted = |groups: &[f32]| {
            let weights = &group_weights[..groups.len()];
            groups
                .iter()
                .zip(weights)
                .map(|(margin, weight)| margin * weight)
                .sum::<f32>()
                / weights.iter().sum::<f32>()
        };
        let rule_selections: Vec<Selection> = TRAINING_RULES
            .iter()
            .enumerate()
            .map(|(index, rule)| by_margin(rule.name(), index))
            .collect();
        let mut group_selections = vec![by_margin("full model (shipped)", 0)];
        for (index, group) in STAT_GROUPS.iter().enumerate() {
            group_selections.push(by_margin(group.name, TRAINING_RULES.len() + index));
        }
        group_selections.extend([
            by_groups("2 of 4 groups", &|groups, _| nth_largest(groups, 1)),
            by_groups("3 of 4 groups", &|groups, _| nth_largest(groups, 2)),
            by_groups("full + 2 groups", &|groups, full| {
                full.min(nth_largest(groups, 1) + 1.0)
            }),
            by_groups("weighted movement groups", &|groups, _| {
                weighted(&groups[..3])
            }),
            by_groups("weighted groups", &|groups, _| weighted(groups)),
            by_groups("full + weighted groups", &|groups, full| {
                f32::midpoint(full, weighted(groups))
            }),
        ]);

        println!(
            "  {} ({} players, shipped margin {shipped_margin})",
            playlist.name,
            players.len()
        );
        for flagged_count in [shipped_count, shipped_count * 3 / 2] {
            println!("    training rules (read each on the other rule's column):");
            print_equal_rate(&players, &rule_selections, flagged_count);
            println!("    stat groups and weighted:");
            print_equal_rate(&players, &group_selections, flagged_count);
        }

        println!("    mechanics above the lobby tier (lobby = median of the others' match tiers):");
        println!(
            "      {:<40} {:>7} {:>12} {:>12} {:>10}",
            "", "flagged", "mechanical", "dominant", "high-rank"
        );
        let weighted_scores = group_selections
            .iter()
            .find(|selection| selection.name == "weighted groups")
            .map(|selection| selection.scores.clone())
            .unwrap_or_default();
        let weighted_flags = top(&weighted_scores, shipped_count);
        print_fixed_rule(&players, "weighted groups (shipped count)", &weighted_flags);
        for (table_index, quantile) in MECHANIC_QUANTILES.iter().enumerate() {
            for minimum in 2..=5 {
                let rule: Vec<bool> = players
                    .iter()
                    .map(|player| player.mechanics_above[table_index] >= minimum)
                    .collect();
                print_fixed_rule(
                    &players,
                    &format!("{minimum}+ above p{:.0}", quantile * 100.0),
                    &rule,
                );
                let union: Vec<bool> = rule
                    .iter()
                    .zip(&weighted_flags)
                    .map(|(rule, weighted)| *rule || *weighted)
                    .collect();
                print_fixed_rule(
                    &players,
                    &format!("weighted or {minimum}+ above p{:.0}", quantile * 100.0),
                    &union,
                );
            }
        }
    }
    Ok(())
}
