#![expect(
    clippy::indexing_slicing,
    reason = "slot arrays indexed by labelled slots, stat arrays by stat indices, tier names by tier_index"
)]

//! Low-rank players who do a mechanic as often as a Champion I, one mechanic at a time.
//!
//! For each playlist and mechanic: the Champion I median rate (training labels), how many
//! players labelled Gold II or lower reach it (and, for context, how many Platinum and
//! Diamond players do), and how many of the evaluation-split ones the shipped smurf flag
//! catches. Then the players who reach it on several mechanics at once, with examples.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin mechanic_smurfs

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use skill_model::coaching::tier_index;
use skill_model::{SkillModelBundle, smurf_suspicions};
use skill_model_training::dataset::{Lobby, Partition, read_lobbies};
use skill_model_training::smurf_references::{
    TIER_NAMES, median, stat_index, team_goal_difference,
};
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(about = "Low-rank players doing each mechanic at a Champion I rate")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

    #[arg(long, default_value = "data/skill_model.bin")]
    bundle: PathBuf,

    /// Mechanics at the Champion I rate that make a multi-mechanic candidate.
    #[arg(long, default_value_t = 3)]
    minimum_mechanics: usize,

    /// Examples to print per playlist.
    #[arg(long, default_value_t = 8)]
    examples: usize,
}

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

/// Fast aerials first, as the reference the others are compared with.
const MECHANICS: [&str; 11] = [
    "fast_aerial_fraction",
    "speed_flips_per_minute",
    "half_flips_per_minute",
    "wavedashes_per_minute",
    "flip_resets_per_minute",
    "ceiling_touches_per_minute",
    "air_dribble_touches_per_minute",
    "double_taps_per_minute",
    "power_shots_per_minute",
    "fifty_fifties_per_minute",
    "demos_inflicted",
];

/// Highest tier counted as low rank (Gold II).
const LOW_RANK_MAXIMUM_TIER: usize = 7;
/// Tier whose median is the bar (Champion I).
const REFERENCE_TIER: usize = 15;
/// Platinum I to Diamond III, for context.
const MIDDLE_TIERS: core::ops::RangeInclusive<usize> = 9..=14;

/// One labelled player.
struct Player {
    replay_id: Uuid,
    slot: usize,
    tier: usize,
    evaluation: bool,
    flagged: bool,
    goal_difference: f32,
    /// Per mechanic: at or above the Champion I median.
    at_reference: [bool; MECHANICS.len()],
}

fn reference_rates(lobbies: &[&Lobby]) -> [f32; MECHANICS.len()] {
    core::array::from_fn(|index| {
        let stat = stat_index(MECHANICS[index]);
        let mut values: Vec<f32> = lobbies
            .iter()
            .filter(|lobby| lobby.partition != Partition::Evaluation)
            .flat_map(|lobby| {
                lobby
                    .labelled_slots()
                    .filter(|&slot| tier_index(lobby.targets[slot]) == REFERENCE_TIER)
                    .filter_map(|slot| lobby.stats[slot].map(|stats| stats[stat]))
            })
            .collect();
        median(&mut values)
    })
}

fn players(
    lobbies: &[&Lobby],
    reference: &[f32; MECHANICS.len()],
    bundle: &SkillModelBundle,
    players_per_team: usize,
) -> Vec<Player> {
    let mut players = Vec::new();
    for lobby in lobbies {
        let stats = lobby.stat_references();
        let flags = smurf_suspicions(
            &bundle.match_model.predict_stats(&stats),
            &bundle.plays_like_model.predict_stats(&stats),
            players_per_team,
        );
        for slot in lobby.labelled_slots() {
            let Some(values) = lobby.stats[slot] else {
                continue;
            };
            players.push(Player {
                replay_id: lobby.replay_id,
                slot,
                tier: tier_index(lobby.targets[slot]),
                evaluation: lobby.partition == Partition::Evaluation,
                flagged: flags[slot].is_some(),
                goal_difference: team_goal_difference(lobby, slot),
                // A Champion I median of zero (flip resets) means "did it at all".
                at_reference: core::array::from_fn(|index| {
                    let value = values[stat_index(MECHANICS[index])];
                    value > 0.0 && value >= reference[index]
                }),
            });
        }
    }
    players
}

fn share(count: usize, total: usize) -> f64 {
    100.0 * count as f64 / total.max(1) as f64
}

fn print_playlist(playlist: &Playlist, players: &[Player], reference: &[f32], args: &Args) {
    let low: Vec<&Player> = players
        .iter()
        .filter(|player| player.tier <= LOW_RANK_MAXIMUM_TIER)
        .collect();
    let middle: Vec<&Player> = players
        .iter()
        .filter(|player| MIDDLE_TIERS.contains(&player.tier))
        .collect();
    let low_evaluation: Vec<&Player> = low.iter().copied().filter(|p| p.evaluation).collect();
    println!(
        "== {} ({} players labelled Gold II or lower, {} of them in the evaluation split)",
        playlist.name,
        low.len(),
        low_evaluation.len()
    );
    println!(
        "  {:<32} {:>10} {:>18} {:>12} {:>22}",
        "mechanic", "Champ I", "low rank at it", "Plat-Dia", "evaluation: flagged"
    );
    for (index, mechanic) in MECHANICS.iter().enumerate() {
        let found = low.iter().filter(|p| p.at_reference[index]).count();
        let middle_found = middle.iter().filter(|p| p.at_reference[index]).count();
        let evaluation_found: Vec<&&Player> = low_evaluation
            .iter()
            .filter(|p| p.at_reference[index])
            .collect();
        let caught = evaluation_found.iter().filter(|p| p.flagged).count();
        println!(
            "  {mechanic:<32} {:>10.3} {:>8} ({:>5.1}%) {:>10.1}% {:>12}/{:<4} ({:.0}%)",
            reference[index],
            found,
            share(found, low.len()),
            share(middle_found, middle.len()),
            caught,
            evaluation_found.len(),
            share(caught, evaluation_found.len()),
        );
    }

    let count_at = |player: &Player| player.at_reference.iter().filter(|hit| **hit).count();
    for minimum in [args.minimum_mechanics, args.minimum_mechanics + 1] {
        let candidates: Vec<&Player> = low
            .iter()
            .copied()
            .filter(|p| count_at(p) >= minimum)
            .collect();
        let evaluation: Vec<&Player> = candidates
            .iter()
            .copied()
            .filter(|p| p.evaluation)
            .collect();
        let won = evaluation
            .iter()
            .filter(|p| p.goal_difference > 0.0)
            .count();
        let caught = evaluation.iter().filter(|p| p.flagged).count();
        println!(
            "  {minimum}+ mechanics at Champion I rate: {} low-rank players ({:.1}%; Plat-Dia {:.1}%), evaluation {}: {won} won, {caught} flagged",
            candidates.len(),
            share(candidates.len(), low.len()),
            share(
                middle.iter().filter(|p| count_at(p) >= minimum).count(),
                middle.len()
            ),
            evaluation.len(),
        );
    }

    let mut examples: Vec<&Player> = low_evaluation.clone();
    examples.sort_by_key(|player| core::cmp::Reverse(count_at(player)));
    println!("  evaluation examples, most mechanics first:");
    for player in examples.into_iter().take(args.examples) {
        let names: Vec<&str> = MECHANICS
            .iter()
            .zip(player.at_reference)
            .filter(|(_, hit)| *hit)
            .map(|(name, _)| *name)
            .collect();
        println!(
            "    {} slot {}  label {:<3}  goal difference {:+.0}  {}  {}",
            player.replay_id,
            player.slot,
            TIER_NAMES[player.tier],
            player.goal_difference,
            if player.flagged {
                "FLAGGED"
            } else {
                "not flagged"
            },
            names.join(", "),
        );
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bundle = SkillModelBundle::from_bytes(
        &std::fs::read(&args.bundle)
            .with_context(|| format!("reading {}", args.bundle.display()))?,
    )
    .map_err(|error| anyhow::anyhow!("decoding bundle: {error}"))?;
    let lobbies = read_lobbies(&args.match_stats)?;
    for playlist in &PLAYLISTS {
        let playlist_lobbies: Vec<&Lobby> = lobbies
            .iter()
            .filter(|lobby| lobby.players_per_team() == Some(playlist.players_per_team))
            .collect();
        let reference = reference_rates(&playlist_lobbies);
        let players = players(
            &playlist_lobbies,
            &reference,
            &bundle,
            playlist.players_per_team,
        );
        print_playlist(playlist, &players, &reference, &args);
    }
    Ok(())
}
