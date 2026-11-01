#![expect(
    clippy::indexing_slicing,
    reason = "slot arrays indexed by slots below TOTAL_SLOTS, stat arrays by stat indices"
)]

//! How many label cleaning rounds should the plays-like model (the smurf flag's own-stats
//! score) train on, and should the other players be scored half by the whole-match model?
//!
//! For each variant this cleans the labels the way `train` does (out-of-fold scores on
//! every own stat, repeated per round), trains the plays-like model on the
//! [`OPPONENT_INDEPENDENT_STATS`], and flags players whose plays-like division is more than
//! N divisions over the median of the other players, on each playlist's evaluation split.
//! The blended variants score the others with the midpoint of their plays-like division and
//! the shipped bundle's whole-match division. Flags are measured against:
//!
//! - **mechanical smurfs**: labelled Gold II or lower, Champion I mechanics on most aerial
//!   stats, and their team won by at least two. The plays-like model sees these stats, so
//!   this set favours it;
//! - **dominant smurfs**: labelled Gold II or lower, their team won by at least three, and
//!   they had the most possession and touches in the lobby. Built only from stats the
//!   plays-like model never sees;
//! - **party positives**, the label-based proxy (label ≥ 150 over the lobby's median);
//! - where the flags land: a flag on a Diamond-or-higher label is most likely wrong.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin smurf_experiment

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use feature_extractor::{MATCH_STAT_COUNT, TOTAL_SLOTS};
use replay_structs::Rank;
use replay_structs::TeamSize;
use skill_model::coaching::tier_index;
use skill_model::{OPPONENT_INDEPENDENT_STATS, PlaysLikeModel, SkillModelBundle};
use skill_model_training::dataset::{Lobby, Partition, read_lobbies};
use skill_model_training::evaluation::PROXY_POSITIVE_MARGIN_MMR;
use skill_model_training::gradient_boosting::GradientBoostingConfig;
use skill_model_training::label_cleaning::{
    SlotScores, clean_labels_in_rounds, train_plays_like_model,
};
use skill_model_training::smurf_references::{
    HIGH_RANK_TIER, MECHANIC_STATS, as_tiers, is_dominant_smurf, is_mechanical_smurf, median,
    reference_mechanics, team_goal_difference,
};
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(about = "Which stats the plays-like model should see")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

    /// Shipped bundle, for the whole-match model the blended variants score opponents with.
    #[arg(long, default_value = "data/skill_model.bin")]
    bundle: PathBuf,

    /// Training labels this far below the player's out-of-fold plays-like prediction are
    /// dropped as likely smurfs.
    #[arg(long, default_value_t = 300.0)]
    clean_margin: f32,

    /// Flag margins, in divisions over the lobby's median plays-like division.
    #[arg(long, value_delimiter = ',', default_value = "3,4,5,6")]
    tier_margins: Vec<f32>,

    /// Examples to print per playlist and variant.
    #[arg(long, default_value_t = 6)]
    examples: usize,
}

/// A per-player score, by replay.
struct Scorer {
    name: &'static str,
    scores: HashMap<Uuid, SlotScores>,
    /// What the other players are scored with when computing a player's margin; their own
    /// `scores` when absent.
    baseline: Option<HashMap<Uuid, SlotScores>>,
    /// Flag margins to report, in divisions.
    margins: Vec<f32>,
}

/// One plays-like variant under test.
struct Variant {
    name: &'static str,
    /// Label cleaning rounds before the plays-like model is trained.
    cleaning_rounds: usize,
    /// Score the other players with the midpoint of their plays-like and whole-match
    /// divisions instead of plays-like alone.
    blend_others_with_match_model: bool,
}

/// One evaluation player as the flag report sees them.
struct EvaluatedPlayer {
    replay_id: Uuid,
    slot: usize,
    label: f32,
    goal_difference: f32,
    mechanical: bool,
    dominant: bool,
    /// `Some(true)` when the label is ≥ 150 over the lobby's median label.
    party_positive: Option<bool>,
    /// Score over the lobby's median score, per scorer.
    margins: Vec<f32>,
    /// Raw score per scorer.
    scores: Vec<f32>,
}

fn evaluated_players(
    lobbies: &[&Lobby],
    scorers: &[Scorer],
    reference: &[f32; MECHANIC_STATS.len()],
) -> Vec<EvaluatedPlayer> {
    let mut players = Vec::new();
    for lobby in lobbies {
        let lobby_scores: Vec<&SlotScores> = scorers
            .iter()
            .filter_map(|scorer| scorer.scores.get(&lobby.replay_id))
            .collect();
        let baseline_scores: Vec<&SlotScores> = scorers
            .iter()
            .filter_map(|scorer| {
                scorer
                    .baseline
                    .as_ref()
                    .unwrap_or(&scorer.scores)
                    .get(&lobby.replay_id)
            })
            .collect();
        if lobby_scores.len() != scorers.len()
            || baseline_scores.len() != scorers.len()
            || lobby.stats.iter().flatten().count() < 2
        {
            continue;
        }
        let mut labels: Vec<f32> = lobby
            .labelled_slots()
            .map(|slot| lobby.targets[slot])
            .collect();
        let label_median = (labels.len() >= 2).then(|| median(&mut labels));
        for slot in 0..TOTAL_SLOTS {
            if lobby.stats[slot].is_none() {
                continue;
            }
            let label = lobby.targets[slot];
            let scores: Vec<f32> = lobby_scores
                .iter()
                .map(|scores| scores[slot].unwrap_or(0.0))
                .collect();
            players.push(EvaluatedPlayer {
                replay_id: lobby.replay_id,
                slot,
                label,
                goal_difference: team_goal_difference(lobby, slot),
                mechanical: is_mechanical_smurf(lobby, slot, reference),
                dominant: is_dominant_smurf(lobby, slot),
                party_positive: label_median
                    .filter(|_| label > 0.0)
                    .map(|median| label - median >= PROXY_POSITIVE_MARGIN_MMR),
                // Over the median of the *other* players: in a duel the lobby median is half
                // way to the smurf, which halves their margin.
                margins: baseline_scores
                    .iter()
                    .zip(&scores)
                    .map(|(baseline_scores, score)| {
                        let mut others: Vec<f32> = baseline_scores
                            .iter()
                            .enumerate()
                            .filter(|(other, _)| *other != slot)
                            .filter_map(|(_, score)| *score)
                            .collect();
                        score - median(&mut others)
                    })
                    .collect(),
                scores,
            });
        }
    }
    players
}

fn print_flag_report(players: &[EvaluatedPlayer], scorers: &[Scorer]) {
    let mechanical = players.iter().filter(|player| player.mechanical).count();
    let dominant = players.iter().filter(|player| player.dominant).count();
    let positives = players
        .iter()
        .filter(|player| player.party_positive == Some(true))
        .count();
    println!(
        "    {} players, {mechanical} mechanical smurfs, {dominant} dominant smurfs, {positives} party positives",
        players.len()
    );
    println!(
        "    {:<12} {:>7} {:>9} {:>12} {:>12} {:>16} {:>10}",
        "plays like", "margin", "flagged", "mechanical", "dominant", "party precision", "high-rank"
    );
    for (index, scorer) in scorers.iter().enumerate() {
        for &margin in &scorer.margins {
            let flagged: Vec<&EvaluatedPlayer> = players
                .iter()
                .filter(|player| player.margins[index] > margin)
                .collect();
            let mechanical_flagged = flagged.iter().filter(|player| player.mechanical).count();
            let dominant_flagged = flagged.iter().filter(|player| player.dominant).count();
            let checkable = flagged
                .iter()
                .filter(|player| player.party_positive.is_some())
                .count();
            let party_correct = flagged
                .iter()
                .filter(|player| player.party_positive == Some(true))
                .count();
            let high_rank = flagged
                .iter()
                .filter(|player| player.label > 0.0 && tier_index(player.label) >= HIGH_RANK_TIER)
                .count();
            println!(
                "    {:<12} {:>+7.0} {:>8.2}% {:>7}/{:<4} {:>7}/{:<4} {:>9}/{:<6} {:>9.0}%",
                scorer.name,
                margin,
                100.0 * flagged.len() as f64 / players.len().max(1) as f64,
                mechanical_flagged,
                mechanical,
                dominant_flagged,
                dominant,
                party_correct,
                checkable,
                100.0 * high_rank as f64 / flagged.len().max(1) as f64,
            );
        }
    }
}

fn print_examples(players: &[EvaluatedPlayer], scorer_index: usize, count: usize) {
    let mut ranked: Vec<&EvaluatedPlayer> = players.iter().collect();
    ranked
        .sort_by(|left, right| right.margins[scorer_index].total_cmp(&left.margins[scorer_index]));
    for player in ranked.into_iter().take(count) {
        let label = if player.label > 0.0 {
            Rank::from_tier_index(tier_index(player.label)).short_name()
        } else {
            "?"
        };
        println!(
            "      {} slot {}  label {label:<3}  plays like {:<3} (+{:.1} divisions over the others)  goal difference {:+.0}{}",
            player.replay_id,
            player.slot,
            Rank::from_tier_index(player.scores[scorer_index] as usize).short_name(),
            player.margins[scorer_index],
            player.goal_difference,
            match (player.mechanical, player.dominant) {
                (true, true) => "  mechanical, dominant",
                (true, false) => "  mechanical",
                (false, true) => "  dominant",
                (false, false) => "",
            },
        );
    }
}

/// How well a variant's plays-like score estimates the label, on evaluation players.
fn print_accuracy(lobbies: &[Lobby], model: &PlaysLikeModel) {
    for playlist in TeamSize::all() {
        let mut errors = Vec::new();
        for lobby in lobbies.iter().filter(|lobby| {
            lobby.partition == Partition::Evaluation
                && lobby.players_per_team() == Some(playlist.players_per_team())
        }) {
            let scores = model.predict_stats(&lobby.stat_references());
            for slot in lobby.labelled_slots() {
                if let Some(score) = scores[slot] {
                    errors.push(f64::from(score - lobby.targets[slot]));
                }
            }
        }
        let count = errors.len().max(1) as f64;
        println!(
            "  {:<9} plays-like vs label: rmse={:.0}  mean error={:+.0}  players={}",
            playlist.name(),
            (errors.iter().map(|error| error * error).sum::<f64>() / count).sqrt(),
            errors.iter().sum::<f64>() / count,
            errors.len(),
        );
    }
}

/// The plays-like model on labels cleaned `rounds` times.
fn cleaned_plays_like_model(
    lobbies: &[Lobby],
    rounds: usize,
    margin: f32,
    config: &GradientBoostingConfig,
) -> PlaysLikeModel {
    let cleaned = clean_labels_in_rounds(
        lobbies,
        &PlaysLikeModel::own_stats_layout(),
        config,
        rounds,
        margin,
    );
    train_plays_like_model(
        &cleaned.lobbies,
        &PlaysLikeModel::opponent_independent_layout(),
        config,
    )
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
    let variants = [
        Variant {
            name: "1 round",
            cleaning_rounds: 1,
            blend_others_with_match_model: false,
        },
        Variant {
            name: "1 rnd blend",
            cleaning_rounds: 1,
            blend_others_with_match_model: true,
        },
        Variant {
            name: "2 rounds",
            cleaning_rounds: 2,
            blend_others_with_match_model: false,
        },
        Variant {
            name: "2 rnd blend",
            cleaning_rounds: 2,
            blend_others_with_match_model: true,
        },
    ];
    println!(
        "opponent-independent plays-like model: {} of {} stats",
        OPPONENT_INDEPENDENT_STATS.len(),
        MATCH_STAT_COUNT
    );

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
    let mut models_by_rounds: HashMap<usize, PlaysLikeModel> = HashMap::new();
    let mut scorers = Vec::new();
    for variant in &variants {
        if let std::collections::hash_map::Entry::Vacant(vacant) =
            models_by_rounds.entry(variant.cleaning_rounds)
        {
            println!(
                "== {} cleaning round(s) (≥ {} MMR)",
                variant.cleaning_rounds, args.clean_margin
            );
            let model = cleaned_plays_like_model(
                &lobbies,
                variant.cleaning_rounds,
                args.clean_margin,
                &config,
            );
            print_accuracy(&lobbies, &model);
            vacant.insert(model);
        }
        let Some(model) = models_by_rounds.get(&variant.cleaning_rounds) else {
            continue;
        };
        let scores: HashMap<Uuid, SlotScores> = evaluation
            .iter()
            .map(|lobby| {
                (
                    lobby.replay_id,
                    as_tiers(model.predict_stats(&lobby.stat_references())),
                )
            })
            .collect();
        let baseline = variant.blend_others_with_match_model.then(|| {
            scores
                .iter()
                .filter_map(|(replay_id, plays_like)| {
                    let match_model = match_tiers.get(replay_id)?;
                    let blended: SlotScores = core::array::from_fn(|slot| {
                        plays_like[slot]
                            .zip(match_model[slot])
                            .map(|(plays_like, match_model)| f32::midpoint(plays_like, match_model))
                    });
                    Some((*replay_id, blended))
                })
                .collect()
        });
        scorers.push(Scorer {
            name: variant.name,
            margins: args.tier_margins.clone(),
            scores,
            baseline,
        });
    }

    println!("== evaluation split");
    for playlist in TeamSize::all() {
        let playlist_lobbies: Vec<&Lobby> = evaluation
            .iter()
            .copied()
            .filter(|lobby| lobby.players_per_team() == Some(playlist.players_per_team()))
            .collect();
        if playlist_lobbies.is_empty() {
            continue;
        }
        let all_playlist: Vec<Lobby> = lobbies
            .iter()
            .filter(|lobby| lobby.players_per_team() == Some(playlist.players_per_team()))
            .cloned()
            .collect();
        let reference = reference_mechanics(&all_playlist);
        println!("  {} ({} replays)", playlist.name(), playlist_lobbies.len());
        let players = evaluated_players(&playlist_lobbies, &scorers, &reference);
        print_flag_report(&players, &scorers);
        for (index, scorer) in scorers.iter().enumerate() {
            println!("    biggest {} margins:", scorer.name);
            print_examples(&players, index, args.examples);
        }
    }
    Ok(())
}
