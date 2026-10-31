#![expect(
    clippy::indexing_slicing,
    reason = "slot arrays indexed by slots below TOTAL_SLOTS, stat arrays by stat indices"
)]

//! Can a "plays like" score from a player's own stats find the smurfs the shipped model
//! misses?
//!
//! The shipped model predicts a lobby level plus each player's gap to it, and the gap is
//! learned from rank labels. Duels and doubles labels almost never differ inside a lobby,
//! and a smurf is labelled with the smurf account's rank, so it learns that no gap exists.
//! This experiment:
//!
//! 1. Trains an **own-stats** model (each player's own stats only, no lobby features) with
//!    out-of-fold predictions for every training player.
//! 2. **Cleans labels**: training players whose own-stats prediction is at least
//!    `--clean-margin` over their label lose that label, then both models are retrained.
//! 3. Scores smurf flags (score over the lobby's median score, in MMR or in tiers) from the shipped model and
//!    from the own-stats models on each playlist's evaluation split, against:
//!    - **obvious smurfs**, defined without any model: labelled Gold II or lower, Champion I
//!      mechanics on most aerial stats, and their team won by at least two goals;
//!    - **party positives**, the label-based proxy (label ≥ 150 over the lobby's median);
//!    - where the flags land: a flag on a Diamond-or-higher label is most likely wrong.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin smurf_experiment

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use feature_extractor::{MATCH_STAT_NAMES, SLOTS_PER_TEAM, TOTAL_SLOTS};
use skill_model::SkillModelBundle;
use skill_model::coaching::tier_index;
use skill_model_training::dataset::{Lobby, Partition, evaluate, read_lobbies, train_model};
use skill_model_training::evaluation::{
    MINIMUM_PLAYERS_PER_LOBBY, PROXY_POSITIVE_MARGIN_MMR, PlayerPrediction,
    compute_lobby_metrics_with_minimum,
};
use skill_model_training::gradient_boosting::GradientBoostingConfig;
use skill_model_training::label_cleaning::{
    SlotScores, clean_labels, out_of_fold_plays_like, train_plays_like_model,
};
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(about = "Own-stats smurf scores and label cleaning")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

    #[arg(long, default_value = "data/skill_model.bin")]
    bundle: PathBuf,

    /// Training labels this far below the player's out-of-fold own-stats prediction are
    /// dropped as likely smurfs.
    #[arg(long, default_value_t = 300.0)]
    clean_margin: f32,

    /// Flag margins over the lobby's median score to report, in MMR.
    #[arg(long, value_delimiter = ',', default_value = "100,150,200,300")]
    flag_margins: Vec<f32>,

    /// Flag margins for the tier-based scores, in tiers (3 tiers = one named rank).
    #[arg(long, value_delimiter = ',', default_value = "3,4,5,6")]
    tier_margins: Vec<f32>,

    /// Examples to print per playlist.
    #[arg(long, default_value_t = 8)]
    examples: usize,
}

/// Rank names by tier index.
const TIER_NAMES: [&str; 22] = [
    "B1", "B2", "B3", "S1", "S2", "S3", "G1", "G2", "G3", "P1", "P2", "P3", "D1", "D2", "D3", "C1",
    "C2", "C3", "GC1", "GC2", "GC3", "SSL",
];

/// Highest tier an obvious smurf can be labelled with (Gold II).
const OBVIOUS_MAXIMUM_TIER: usize = 7;

/// Tier whose median mechanics an obvious smurf must reach (Champion I).
const OBVIOUS_REFERENCE_TIER: usize = 15;

/// Aerial mechanics compared with the reference tier.
const MECHANIC_STATS: [&str; 6] = [
    "fast_aerial_fraction",
    "air_roll_fraction",
    "aerial_touch_fraction",
    "high_air_fraction",
    "aerial_dodges_per_minute",
    "wavedash_like_dodges_per_minute",
];

/// How many of [`MECHANIC_STATS`] must reach the reference tier's median.
const MINIMUM_MECHANICS_AT_REFERENCE: usize = 4;

/// Goals the obvious smurf's team must win by.
const MINIMUM_GOAL_DIFFERENCE: f32 = 2.0;

/// Labels at or above this tier (Diamond I) make a flag most likely wrong.
const HIGH_RANK_TIER: usize = 12;

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

/// A per-player score, by replay.
struct Scorer {
    name: &'static str,
    scores: HashMap<Uuid, SlotScores>,
    /// Flag margins to report, in the score's unit (MMR, or tiers).
    margins: Vec<f32>,
}

fn stat_index(name: &str) -> usize {
    MATCH_STAT_NAMES
        .iter()
        .position(|stat| *stat == name)
        .unwrap_or(usize::MAX)
}

/// Scores as tier indices, so a margin means the same number of divisions at every rank.
fn as_tiers(scores: SlotScores) -> SlotScores {
    scores.map(|score| score.map(|mmr| tier_index(mmr) as f32))
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    let upper = values.get(middle).copied().unwrap_or(0.0);
    if values.len().is_multiple_of(2) {
        let lower = values.get(middle.wrapping_sub(1)).copied().unwrap_or(upper);
        f32::midpoint(lower, upper)
    } else {
        upper
    }
}

/// Median of each mechanic stat among training players labelled with the reference tier.
fn reference_mechanics(lobbies: &[Lobby]) -> [f32; MECHANIC_STATS.len()] {
    core::array::from_fn(|index| {
        let stat = stat_index(MECHANIC_STATS[index]);
        let mut values: Vec<f32> = lobbies
            .iter()
            .filter(|lobby| lobby.partition != Partition::Evaluation)
            .flat_map(|lobby| {
                lobby
                    .labelled_slots()
                    .filter(|&slot| tier_index(lobby.targets[slot]) == OBVIOUS_REFERENCE_TIER)
                    .filter_map(|slot| lobby.stats[slot].map(|stats| stats[stat]))
            })
            .collect();
        median(&mut values)
    })
}

fn team_goal_difference(lobby: &Lobby, slot: usize) -> f32 {
    let goals = stat_index("goals");
    let team = usize::from(slot >= SLOTS_PER_TEAM);
    let mut team_goals = [0.0f32; 2];
    for (other, stats) in lobby.stats.iter().enumerate() {
        if let Some(stats) = stats {
            team_goals[usize::from(other >= SLOTS_PER_TEAM)] += stats[goals];
        }
    }
    team_goals[team] - team_goals[1 - team]
}

fn is_obvious_smurf(lobby: &Lobby, slot: usize, reference: &[f32; MECHANIC_STATS.len()]) -> bool {
    let label = lobby.targets[slot];
    let Some(stats) = lobby.stats[slot].as_ref() else {
        return false;
    };
    if label <= 0.0 || tier_index(label) > OBVIOUS_MAXIMUM_TIER {
        return false;
    }
    let at_reference = MECHANIC_STATS
        .iter()
        .zip(reference)
        .filter(|(name, median)| **median > 0.0 && stats[stat_index(name)] >= **median)
        .count();
    at_reference >= MINIMUM_MECHANICS_AT_REFERENCE
        && team_goal_difference(lobby, slot) >= MINIMUM_GOAL_DIFFERENCE
}

/// One evaluation player as the flag report sees them.
struct EvaluatedPlayer {
    replay_id: Uuid,
    slot: usize,
    label: f32,
    goal_difference: f32,
    obvious: bool,
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
        if lobby_scores.len() != scorers.len() {
            continue;
        }
        let medians: Vec<f32> = lobby_scores
            .iter()
            .map(|scores| median(&mut scores.iter().flatten().copied().collect::<Vec<_>>()))
            .collect();
        let mut labels: Vec<f32> = lobby
            .labelled_slots()
            .map(|slot| lobby.targets[slot])
            .collect();
        let label_median = (labels.len() >= 2).then(|| median(&mut labels));
        for slot in 0..TOTAL_SLOTS {
            if lobby.stats[slot].is_none() || lobby.stats.iter().flatten().count() < 2 {
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
                obvious: is_obvious_smurf(lobby, slot, reference),
                party_positive: label_median
                    .filter(|_| label > 0.0)
                    .map(|median| label - median >= PROXY_POSITIVE_MARGIN_MMR),
                margins: scores
                    .iter()
                    .zip(&medians)
                    .map(|(score, median)| score - median)
                    .collect(),
                scores,
            });
        }
    }
    players
}

fn print_flag_report(players: &[EvaluatedPlayer], scorers: &[Scorer]) {
    let obvious = players.iter().filter(|player| player.obvious).count();
    let positives = players
        .iter()
        .filter(|player| player.party_positive == Some(true))
        .count();
    println!(
        "    {} players, {obvious} obvious smurfs, {positives} party positives",
        players.len()
    );
    println!(
        "    {:<14} {:>7} {:>9} {:>14} {:>16} {:>12}",
        "flag", "margin", "flagged", "obvious recall", "party precision", "high-rank"
    );
    for (index, scorer) in scorers.iter().enumerate() {
        for &margin in &scorer.margins {
            let flagged: Vec<&EvaluatedPlayer> = players
                .iter()
                .filter(|player| player.margins[index] > margin)
                .collect();
            let obvious_flagged = flagged.iter().filter(|player| player.obvious).count();
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
                "    {:<14} {:>+7.0} {:>8.2}% {:>8}/{:<5} {:>9}/{:<6} {:>11.0}%",
                scorer.name,
                margin,
                100.0 * flagged.len() as f64 / players.len().max(1) as f64,
                obvious_flagged,
                obvious,
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
            TIER_NAMES[tier_index(player.label)]
        } else {
            "?"
        };
        println!(
            "    {} slot {}  label {label:<3}  plays like {:<3} ({:+.0} over lobby)  goal difference {:+.0}{}",
            player.replay_id,
            player.slot,
            TIER_NAMES[tier_index(player.scores[scorer_index])],
            player.margins[scorer_index],
            player.goal_difference,
            if player.obvious { "  obvious" } else { "" },
        );
    }
}

fn main_model_metrics(
    label: &str,
    lobbies: &[Lobby],
    model: &skill_model::TabularSkillModel,
    players_per_team: usize,
) {
    let predictions: Vec<PlayerPrediction> = evaluate(lobbies, model)
        .iter()
        .map(|scored| scored.prediction)
        .collect();
    let metrics = compute_lobby_metrics_with_minimum(
        &predictions,
        MINIMUM_PLAYERS_PER_LOBBY.min(players_per_team * 2),
    );
    println!(
        "    {label:<34} within_r={:.3} ±{:.3}  lobby_rmse={:.0}  player_rmse={:.0}  concordance={:.3}",
        metrics.within_r,
        metrics.within_r_standard_error,
        metrics.lobby_rmse,
        metrics.player_rmse,
        metrics.concordance,
    );
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = GradientBoostingConfig::default();
    let lobbies = read_lobbies(&args.match_stats)?;
    let shipped = SkillModelBundle::from_bytes(
        &std::fs::read(&args.bundle)
            .with_context(|| format!("reading {}", args.bundle.display()))?,
    )
    .map_err(|error| anyhow::anyhow!("decoding bundle: {error}"))?
    .match_model;

    println!("== 1. own-stats scores");
    let out_of_fold = out_of_fold_plays_like(&lobbies, &config);
    println!("  own-stats model, all training data");
    let own = train_plays_like_model(&lobbies, &config);

    println!(
        "== 2. label cleaning (own-stats prediction ≥ label + {})",
        args.clean_margin
    );
    let cleaned_labels = clean_labels(&lobbies, &out_of_fold, args.clean_margin);
    cleaned_labels.counts.print_report();
    let cleaned = cleaned_labels.lobbies;
    println!("  own-stats model, cleaned labels");
    let own_cleaned = train_plays_like_model(&cleaned, &config);
    println!("  full model, cleaned labels");
    let main_cleaned = train_model(&cleaned, &shipped.absolute_layout, &config);

    let evaluation: Vec<&Lobby> = lobbies
        .iter()
        .filter(|lobby| lobby.partition == Partition::Evaluation)
        .collect();
    let scores_of = |predict: &dyn Fn(&Lobby) -> SlotScores| -> HashMap<Uuid, SlotScores> {
        evaluation
            .iter()
            .map(|lobby| (lobby.replay_id, predict(lobby)))
            .collect()
    };
    let scorers = [
        Scorer {
            name: "shipped",
            margins: args.flag_margins.clone(),
            scores: scores_of(&|lobby| shipped.predict_stats(&lobby.stat_references())),
        },
        Scorer {
            name: "full cleaned",
            margins: args.flag_margins.clone(),
            scores: scores_of(&|lobby| main_cleaned.predict_stats(&lobby.stat_references())),
        },
        Scorer {
            name: "own",
            margins: args.flag_margins.clone(),
            scores: scores_of(&|lobby| own.predict_stats(&lobby.stat_references())),
        },
        Scorer {
            name: "own cleaned",
            margins: args.flag_margins.clone(),
            scores: scores_of(&|lobby| own_cleaned.predict_stats(&lobby.stat_references())),
        },
        Scorer {
            name: "own tiers",
            margins: args.tier_margins.clone(),
            scores: scores_of(&|lobby| as_tiers(own.predict_stats(&lobby.stat_references()))),
        },
        Scorer {
            name: "own cl. tiers",
            margins: args.tier_margins.clone(),
            scores: scores_of(&|lobby| {
                as_tiers(own_cleaned.predict_stats(&lobby.stat_references()))
            }),
        },
    ];
    let own_cleaned_index = 3;

    // Evaluation lobbies without the players the raw own-stats model calls smurfs, to see
    // whether cleaning helps on the players whose labels can be trusted.
    let evaluation_trusted: Vec<Lobby> = lobbies
        .iter()
        .filter(|lobby| lobby.partition == Partition::Evaluation)
        .map(|lobby| {
            let mut trusted = lobby.clone();
            let scores = own.predict_stats(&lobby.stat_references());
            for (target, score) in trusted.targets.iter_mut().zip(scores) {
                if *target > 0.0 && score.is_some_and(|score| score - *target >= args.clean_margin)
                {
                    *target = 0.0;
                }
            }
            trusted
        })
        .collect();

    println!("== 3. evaluation split");
    for playlist in &PLAYLISTS {
        let playlist_lobbies: Vec<&Lobby> = evaluation
            .iter()
            .copied()
            .filter(|lobby| lobby.players_per_team() == Some(playlist.players_per_team))
            .collect();
        if playlist_lobbies.is_empty() {
            continue;
        }
        let all_playlist: Vec<Lobby> = lobbies
            .iter()
            .filter(|lobby| lobby.players_per_team() == Some(playlist.players_per_team))
            .cloned()
            .collect();
        let reference = reference_mechanics(&all_playlist);
        println!(
            "  {} ({} replays); Champion I mechanics medians: {}",
            playlist.name,
            playlist_lobbies.len(),
            MECHANIC_STATS
                .iter()
                .zip(&reference)
                .map(|(name, value)| format!("{name}={value:.2}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let playlist_evaluation: Vec<Lobby> =
            playlist_lobbies.iter().map(|l| (*l).clone()).collect();
        let playlist_trusted: Vec<Lobby> = evaluation_trusted
            .iter()
            .filter(|lobby| lobby.players_per_team() == Some(playlist.players_per_team))
            .cloned()
            .collect();
        main_model_metrics(
            "shipped, all labels",
            &playlist_evaluation,
            &shipped,
            playlist.players_per_team,
        );
        main_model_metrics(
            "cleaned, all labels",
            &playlist_evaluation,
            &main_cleaned,
            playlist.players_per_team,
        );
        main_model_metrics(
            "shipped, labels without smurfs",
            &playlist_trusted,
            &shipped,
            playlist.players_per_team,
        );
        main_model_metrics(
            "cleaned, labels without smurfs",
            &playlist_trusted,
            &main_cleaned,
            playlist.players_per_team,
        );
        let players = evaluated_players(&playlist_lobbies, &scorers, &reference);
        print_flag_report(&players, &scorers);
        println!("    biggest own-cleaned margins:");
        print_examples(&players, own_cleaned_index, args.examples);
    }
    Ok(())
}
