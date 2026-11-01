#![expect(
    clippy::indexing_slicing,
    reason = "slot arrays indexed by slots below TOTAL_SLOTS"
)]

//! Writes the shipped bundle's per-player predictions for every replay to a CSV: label,
//! whole-match MMR, plays-like MMR and smurf flag per (replay, slot). For analyses joined
//! with data outside the stats CSV, such as account ids.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin export_predictions

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use feature_extractor::TOTAL_SLOTS;
use skill_model::mechanics::mechanics_above_lobby;
use skill_model::{SkillModelBundle, SlotEvidence, SmurfReason, smurf_suspicions};
use skill_model_training::dataset::{Partition, read_lobbies};

#[derive(Parser, Debug)]
#[command(about = "Export per-player predictions of the shipped bundle")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

    #[arg(long, default_value = "data/skill_model.bin")]
    bundle: PathBuf,

    #[arg(long, default_value = "data/predictions.csv")]
    out: PathBuf,
}

const fn reason_name(reason: Option<SmurfReason>) -> &'static str {
    match reason {
        None => "",
        Some(SmurfReason::DerankingAboveLobby) => "deranking",
        Some(SmurfReason::RankedAboveLobby) => "ranked",
        Some(SmurfReason::PlaysAboveLobby) => "plays",
        Some(SmurfReason::MechanicsAboveLobby) => "mechanics",
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
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&args.out)
            .with_context(|| format!("creating {}", args.out.display()))?,
    );
    writeln!(
        out,
        "replay_id,slot,evaluation,players_per_team,target_mmr,match_mmr,plays_like_mmr,smurf"
    )?;
    let mut rows = 0usize;
    for lobby in &lobbies {
        let Some(players_per_team) = lobby.players_per_team() else {
            continue;
        };
        let stats = lobby.stat_references();
        let match_mmr = bundle.match_model.predict_stats(&stats);
        let plays_like = bundle.plays_like_groups.predict_stats(&stats);
        let plays_like_mmr = plays_like.combined_mmr();
        let mechanics = mechanics_above_lobby(
            bundle.mechanics.for_players_per_team(players_per_team),
            &stats,
            &match_mmr,
        );
        let flags = smurf_suspicions(
            &match_mmr,
            &plays_like,
            &SlotEvidence::per_slot(&mechanics, &stats),
            players_per_team,
        );
        for slot in 0..TOTAL_SLOTS {
            let Some(match_value) = match_mmr[slot] else {
                continue;
            };
            let Some(plays_like_value) = plays_like_mmr[slot] else {
                continue;
            };
            writeln!(
                out,
                "{},{slot},{},{players_per_team},{},{match_value:.1},{plays_like_value:.1},{}",
                lobby.replay_id,
                u8::from(lobby.partition == Partition::Evaluation),
                lobby.targets[slot],
                reason_name(flags[slot]),
            )?;
            rows += 1;
        }
    }
    out.flush()?;
    println!("export_predictions: {rows} rows → {}", args.out.display());
    Ok(())
}
