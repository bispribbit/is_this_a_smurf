#![expect(
    clippy::indexing_slicing,
    reason = "slot arrays indexed by slots below TOTAL_SLOTS"
)]

//! Smurf examples for a playlist: evaluation replays where the shipped bundle flags a player
//! (prediction over the lobby median prediction by `--margin`, by default the app's
//! [`SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR`]), biggest margin first.
//!
//! A flag is *confirmed* when the player's own label is also at least
//! [`PROXY_POSITIVE_MARGIN_MMR`] over the lobby's median label (mostly a stronger friend in
//! a party), the only smurf-like case where the truth is known.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin find_smurf_examples -- --players-per-team 1

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use feature_extractor::TOTAL_SLOTS;
use skill_model::{SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR, SkillModelBundle};
use skill_model_training::dataset::{Lobby, Partition, read_lobbies};
use skill_model_training::evaluation::PROXY_POSITIVE_MARGIN_MMR;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(about = "List flagged smurfs in a playlist's evaluation replays")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

    #[arg(long, default_value = "data/skill_model.bin")]
    bundle: PathBuf,

    /// Playlist size: 1 duels, 2 doubles, 3 standard.
    #[arg(long)]
    players_per_team: usize,

    /// Flag margin over the lobby median prediction (defaults to the app's).
    #[arg(long, default_value_t = SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR)]
    margin: f32,

    /// How many examples to print.
    #[arg(long, default_value_t = 15)]
    count: usize,
}

/// One flagged player.
struct FlaggedPlayer {
    replay_id: Uuid,
    slot: usize,
    prediction_mmr: f32,
    /// Prediction minus the lobby's median prediction.
    margin_mmr: f32,
    /// Label MMR; `0` when unknown.
    label_mmr: f32,
    /// Label minus the lobby's median label, when at least two labels are known.
    label_margin_mmr: Option<f32>,
    /// Every present player's label, by slot.
    lobby_labels: Vec<f32>,
}

impl FlaggedPlayer {
    fn confirmed(&self) -> bool {
        self.label_margin_mmr
            .is_some_and(|margin| margin >= PROXY_POSITIVE_MARGIN_MMR)
    }
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

fn flagged_players(lobby: &Lobby, bundle: &SkillModelBundle, margin: f32) -> Vec<FlaggedPlayer> {
    let predictions = bundle.match_model.predict_stats(&lobby.stat_references());
    let mut present: Vec<f32> = predictions.iter().flatten().copied().collect();
    if present.len() < 2 {
        return Vec::new();
    }
    let prediction_median = median(&mut present);
    let mut labels: Vec<f32> = lobby
        .labelled_slots()
        .map(|slot| lobby.targets[slot])
        .collect();
    let label_median = (labels.len() >= 2).then(|| median(&mut labels));
    (0..TOTAL_SLOTS)
        .filter_map(|slot| {
            let prediction_mmr = predictions[slot]?;
            let margin_mmr = prediction_mmr - prediction_median;
            (margin_mmr > margin).then(|| {
                let label_mmr = lobby.targets[slot];
                FlaggedPlayer {
                    replay_id: lobby.replay_id,
                    slot,
                    prediction_mmr,
                    margin_mmr,
                    label_mmr,
                    label_margin_mmr: label_median
                        .filter(|_| label_mmr > 0.0)
                        .map(|median| label_mmr - median),
                    lobby_labels: (0..TOTAL_SLOTS)
                        .filter(|&other| lobby.stats[other].is_some())
                        .map(|other| lobby.targets[other])
                        .collect(),
                }
            })
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bundle = SkillModelBundle::from_bytes(
        &std::fs::read(&args.bundle)
            .with_context(|| format!("reading {}", args.bundle.display()))?,
    )
    .map_err(|error| anyhow::anyhow!("decoding bundle: {error}"))?;
    let lobbies: Vec<Lobby> = read_lobbies(&args.match_stats)?
        .into_iter()
        .filter(|lobby| {
            lobby.partition == Partition::Evaluation
                && lobby.players_per_team() == Some(args.players_per_team)
        })
        .collect();

    let mut flagged: Vec<FlaggedPlayer> = lobbies
        .iter()
        .flat_map(|lobby| flagged_players(lobby, &bundle, args.margin))
        .collect();
    let players: usize = lobbies
        .iter()
        .map(|lobby| lobby.stats.iter().flatten().count())
        .sum();
    // Players whose label alone makes them smurf-like, flagged or not.
    let positives: usize = lobbies
        .iter()
        .map(|lobby| {
            let mut labels: Vec<f32> = lobby
                .labelled_slots()
                .map(|slot| lobby.targets[slot])
                .collect();
            if labels.len() < 2 {
                return 0;
            }
            let label_median = median(&mut labels.clone());
            labels.retain(|label| label - label_median >= PROXY_POSITIVE_MARGIN_MMR);
            labels.len()
        })
        .sum();
    let confirmed = flagged.iter().filter(|player| player.confirmed()).count();
    let checkable = flagged
        .iter()
        .filter(|player| player.label_margin_mmr.is_some())
        .count();
    println!(
        "{} evaluation replays, {players} players, {positives} ranked ≥{PROXY_POSITIVE_MARGIN_MMR} over their lobby: {} flagged at +{}, {confirmed} of {checkable} with known labels confirmed",
        lobbies.len(),
        flagged.len(),
        args.margin,
    );

    // Confirmed first, then the biggest margins.
    flagged.sort_by(|left, right| {
        right
            .confirmed()
            .cmp(&left.confirmed())
            .then(right.margin_mmr.total_cmp(&left.margin_mmr))
    });
    for player in flagged.iter().take(args.count) {
        let label_margin = player
            .label_margin_mmr
            .map_or_else(|| "?".to_string(), |margin| format!("{margin:+.0}"));
        println!(
            "  {} {} slot {}  predicted {:.0} ({:+.0} over lobby)  label {:.0} ({label_margin} over lobby)  lobby labels {:?}",
            if player.confirmed() {
                "CONFIRMED"
            } else {
                "flagged  "
            },
            player.replay_id,
            player.slot,
            player.prediction_mmr,
            player.margin_mmr,
            player.label_mmr,
            player.lobby_labels,
        );
    }
    Ok(())
}
