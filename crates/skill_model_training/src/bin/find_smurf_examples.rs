#![expect(
    clippy::indexing_slicing,
    reason = "slot arrays indexed by slots below TOTAL_SLOTS, tier names by tier_index"
)]

//! Smurf examples for a playlist: every player the shipped bundle flags
//! ([`skill_model::smurf_suspicions`], the app's rule) in the evaluation replays, those
//! who play furthest above their lobby first.
//!
//! A flag is *confirmed* when the player's own label is also at least
//! [`PROXY_POSITIVE_MARGIN_MMR`] over the lobby's median label (mostly a stronger friend in
//! a party). A smurf's label is the smurf account's rank, so most real smurfs show as
//! labelled with the lobby: compare the label with what they play like.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin find_smurf_examples -- --players-per-team 1

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use skill_model::coaching::tier_index;
use skill_model::mechanics::mechanics_above_lobby;
use skill_model::{SkillModelBundle, SmurfReason, smurf_suspicions};
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

    /// How many examples to print.
    #[arg(long, default_value_t = 15)]
    count: usize,
}

/// Rank names by tier index.
const TIER_NAMES: [&str; 22] = [
    "B1", "B2", "B3", "S1", "S2", "S3", "G1", "G2", "G3", "P1", "P2", "P3", "D1", "D2", "D3", "C1",
    "C2", "C3", "GC1", "GC2", "GC3", "SSL",
];

/// One flagged player.
struct FlaggedPlayer {
    replay_id: Uuid,
    slot: usize,
    reason: SmurfReason,
    match_mmr: f32,
    plays_like_mmr: f32,
    /// Plays-like divisions over the lobby's lowest plays-like division.
    divisions_over_lowest: usize,
    /// Label MMR; `0` when unknown.
    label_mmr: f32,
    /// Label minus the lobby's median label, when at least two labels are known.
    label_margin_mmr: Option<f32>,
    /// Every present player's label tier, in slot order.
    lobby_labels: Vec<&'static str>,
}

impl FlaggedPlayer {
    fn confirmed(&self) -> bool {
        self.label_margin_mmr
            .is_some_and(|margin| margin >= PROXY_POSITIVE_MARGIN_MMR)
    }
}

fn tier_name(mmr: f32) -> &'static str {
    if mmr > 0.0 {
        TIER_NAMES[tier_index(mmr)]
    } else {
        "?"
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

fn flagged_players(
    lobby: &Lobby,
    bundle: &SkillModelBundle,
    players_per_team: usize,
) -> Vec<FlaggedPlayer> {
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
        &skill_model::SlotEvidence::per_slot(&mechanics, &stats),
        players_per_team,
    );
    let lowest_tier = plays_like_mmr
        .iter()
        .flatten()
        .map(|&mmr| tier_index(mmr))
        .min()
        .unwrap_or(0);
    let mut labels: Vec<f32> = lobby
        .labelled_slots()
        .map(|slot| lobby.targets[slot])
        .collect();
    let label_median = (labels.len() >= 2).then(|| median(&mut labels));
    flags
        .iter()
        .enumerate()
        .filter_map(|(slot, flag)| {
            let reason = (*flag)?;
            let label_mmr = lobby.targets[slot];
            let plays_like_mmr = plays_like_mmr[slot]?;
            Some(FlaggedPlayer {
                replay_id: lobby.replay_id,
                slot,
                reason,
                match_mmr: match_mmr[slot]?,
                plays_like_mmr,
                divisions_over_lowest: tier_index(plays_like_mmr).saturating_sub(lowest_tier),
                label_mmr,
                label_margin_mmr: label_median
                    .filter(|_| label_mmr > 0.0)
                    .map(|median| label_mmr - median),
                lobby_labels: lobby
                    .stats
                    .iter()
                    .zip(&lobby.targets)
                    .filter(|(stats, _)| stats.is_some())
                    .map(|(_, &target)| tier_name(target))
                    .collect(),
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
        .flat_map(|lobby| flagged_players(lobby, &bundle, args.players_per_team))
        .collect();
    let players: usize = lobbies
        .iter()
        .map(|lobby| lobby.stats.iter().flatten().count())
        .sum();
    let with_reason = |reason: SmurfReason| {
        flagged
            .iter()
            .filter(|player| player.reason == reason)
            .count()
    };
    let confirmed = flagged.iter().filter(|player| player.confirmed()).count();
    println!(
        "{} evaluation replays, {players} players: {} flagged ({} deranking, {} ranked above the lobby, {} play above it, {} mechanics above it), {confirmed} confirmed by labels",
        lobbies.len(),
        flagged.len(),
        with_reason(SmurfReason::DerankingAboveLobby),
        with_reason(SmurfReason::RankedAboveLobby),
        with_reason(SmurfReason::PlaysAboveLobby),
        with_reason(SmurfReason::MechanicsAboveLobby),
    );

    flagged.sort_by(|left, right| {
        right
            .divisions_over_lowest
            .cmp(&left.divisions_over_lowest)
            .then(right.plays_like_mmr.total_cmp(&left.plays_like_mmr))
    });
    for player in flagged.iter().take(args.count) {
        println!(
            "  {} slot {}  {:<10} card {:<3} plays like {:<3}  label {:<3}{}  lobby labels {:?}",
            player.replay_id,
            player.slot,
            match player.reason {
                SmurfReason::RankedAboveLobby => "ranked",
                SmurfReason::DerankingAboveLobby => "deranking",
                SmurfReason::PlaysAboveLobby => "plays",
                SmurfReason::MechanicsAboveLobby => "mechanics",
            },
            tier_name(player.match_mmr),
            tier_name(player.plays_like_mmr),
            tier_name(player.label_mmr),
            if player.confirmed() {
                " (confirmed)"
            } else {
                ""
            },
            player.lobby_labels,
        );
    }
    Ok(())
}
