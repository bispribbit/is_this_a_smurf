#![expect(
    clippy::indexing_slicing,
    reason = "slot arrays indexed by slots below TOTAL_SLOTS, stat arrays by stat indices"
)]

//! Which stats should the plays-like model (the smurf flag's own-stats score) see?
//!
//! For each variant (every stat, or only [`OPPONENT_INDEPENDENT_STATS`]) this trains the
//! plays-like model the way `train` does: out-of-fold scores, label cleaning, then a model
//! on the cleaned labels. It then flags players whose plays-like division is more than N
//! divisions over the median of the other players, on each playlist's evaluation split,
//! against:
//!
//! - **mechanical smurfs**: labelled Gold II or lower, Champion I mechanics on most aerial
//!   stats, and their team won by at least two. Both variants see these stats, so this set
//!   favours them, the opponent-independent one most;
//! - **dominant smurfs**: labelled Gold II or lower, their team won by at least three, and
//!   they had the most possession and touches in the lobby. Built only from stats the
//!   opponent-independent variant never sees;
//! - **party positives**, the label-based proxy (label ≥ 150 over the lobby's median);
//! - where the flags land: a flag on a Diamond-or-higher label is most likely wrong.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin smurf_experiment

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use feature_extractor::{MATCH_STAT_NAMES, SLOTS_PER_TEAM, TOTAL_SLOTS};
use skill_model::coaching::tier_index;
use skill_model::{OPPONENT_INDEPENDENT_STATS, PlaysLikeModel, TabularLayout};
use skill_model_training::dataset::{Lobby, Partition, read_lobbies};
use skill_model_training::evaluation::PROXY_POSITIVE_MARGIN_MMR;
use skill_model_training::gradient_boosting::GradientBoostingConfig;
use skill_model_training::label_cleaning::{
    SlotScores, clean_labels, out_of_fold_plays_like, train_plays_like_model,
};
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(about = "Which stats the plays-like model should see")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

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

/// Rank names by tier index.
const TIER_NAMES: [&str; 22] = [
    "B1", "B2", "B3", "S1", "S2", "S3", "G1", "G2", "G3", "P1", "P2", "P3", "D1", "D2", "D3", "C1",
    "C2", "C3", "GC1", "GC2", "GC3", "SSL",
];

/// Highest tier a mechanical or dominant smurf can be labelled with (Gold II).
const OBVIOUS_MAXIMUM_TIER: usize = 7;

/// Tier whose median mechanics a mechanical smurf must reach (Champion I).
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

/// Goals a mechanical smurf's team must win by.
const MINIMUM_GOAL_DIFFERENCE: f32 = 2.0;

/// Goals a dominant smurf's team must win by.
const DOMINANT_GOAL_DIFFERENCE: f32 = 3.0;

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
    /// Flag margins to report, in divisions.
    margins: Vec<f32>,
}

/// One plays-like variant under test.
struct Variant {
    name: &'static str,
    layout: TabularLayout,
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

/// Labelled Gold II or lower, Champion I mechanics on most aerial stats, won by two.
fn is_mechanical_smurf(
    lobby: &Lobby,
    slot: usize,
    reference: &[f32; MECHANIC_STATS.len()],
) -> bool {
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

/// Labelled Gold II or lower, won by three, and the most possession and touches in the
/// lobby.
fn is_dominant_smurf(lobby: &Lobby, slot: usize) -> bool {
    let label = lobby.targets[slot];
    if label <= 0.0 || tier_index(label) > OBVIOUS_MAXIMUM_TIER {
        return false;
    }
    let leads = |name: &str| {
        let stat = stat_index(name);
        let own = lobby.stats[slot].map_or(f32::NEG_INFINITY, |stats| stats[stat]);
        lobby
            .stats
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != slot)
            .filter_map(|(_, stats)| stats.map(|stats| stats[stat]))
            .all(|value| own > value)
    };
    team_goal_difference(lobby, slot) >= DOMINANT_GOAL_DIFFERENCE
        && leads("possession_fraction")
        && leads("touches_per_minute")
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
        if lobby_scores.len() != scorers.len() || lobby.stats.iter().flatten().count() < 2 {
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
                margins: lobby_scores
                    .iter()
                    .zip(&scores)
                    .map(|(lobby_scores, score)| {
                        let mut others: Vec<f32> = lobby_scores
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
            TIER_NAMES[tier_index(player.label)]
        } else {
            "?"
        };
        println!(
            "      {} slot {}  label {label:<3}  plays like {:<3} (+{:.1} divisions over the others)  goal difference {:+.0}{}",
            player.replay_id,
            player.slot,
            TIER_NAMES[(player.scores[scorer_index] as usize).min(21)],
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
    for playlist in &PLAYLISTS {
        let mut errors = Vec::new();
        for lobby in lobbies.iter().filter(|lobby| {
            lobby.partition == Partition::Evaluation
                && lobby.players_per_team() == Some(playlist.players_per_team)
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
            playlist.name,
            (errors.iter().map(|error| error * error).sum::<f64>() / count).sqrt(),
            errors.iter().sum::<f64>() / count,
            errors.len(),
        );
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = GradientBoostingConfig::default();
    let lobbies = read_lobbies(&args.match_stats)?;
    let variants = [
        Variant {
            name: "all stats",
            layout: PlaysLikeModel::own_stats_layout(),
        },
        Variant {
            name: "independent",
            layout: PlaysLikeModel::opponent_independent_layout(),
        },
    ];
    println!(
        "opponent-independent variant: {} of {} stats",
        OPPONENT_INDEPENDENT_STATS.len(),
        MATCH_STAT_NAMES.len()
    );

    let evaluation: Vec<&Lobby> = lobbies
        .iter()
        .filter(|lobby| lobby.partition == Partition::Evaluation)
        .collect();
    let mut scorers = Vec::new();
    for variant in &variants {
        println!(
            "== {}: out-of-fold scores, label cleaning (≥ {} MMR), cleaned model",
            variant.name, args.clean_margin
        );
        let out_of_fold = out_of_fold_plays_like(&lobbies, &variant.layout, &config);
        let cleaned = clean_labels(&lobbies, &out_of_fold, args.clean_margin);
        cleaned.counts.print_report();
        let model = train_plays_like_model(&cleaned.lobbies, &variant.layout, &config);
        print_accuracy(&lobbies, &model);
        scorers.push(Scorer {
            name: variant.name,
            margins: args.tier_margins.clone(),
            scores: evaluation
                .iter()
                .map(|lobby| {
                    (
                        lobby.replay_id,
                        as_tiers(model.predict_stats(&lobby.stat_references())),
                    )
                })
                .collect(),
        });
    }

    println!("== evaluation split");
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
        println!("  {} ({} replays)", playlist.name, playlist_lobbies.len());
        let players = evaluated_players(&playlist_lobbies, &scorers, &reference);
        print_flag_report(&players, &scorers);
        for (index, scorer) in scorers.iter().enumerate() {
            println!("    biggest {} margins:", scorer.name);
            print_examples(&players, index, args.examples);
        }
    }
    Ok(())
}
