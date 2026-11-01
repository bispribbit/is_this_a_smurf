//! Parses every downloaded replay into per-player stats for training.
//!
//! Writes two CSVs in one pass: whole-match stats (one row per player per replay) and
//! window stats (one row per player per window of `--window-seconds` of live play, with a
//! trailing `window_index` column). A third CSV maps each (replay, slot) to the player's
//! account id from the Ballchasing metadata, for account-level label cleaning. Slots follow the app's convention: blue sorted by name,
//! then orange. Labels come from the database; `0` means the rank is unknown.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin extract_stats

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use config::get_base_path;
use database::{
    assign_dataset_splits, initialize_pool, list_replay_players_by_replay, list_replays_by_split,
};
use feature_extractor::{
    MatchStat, PlayerMatchStats, PlayerRoster, SLOTS_PER_TEAM, TOTAL_SLOTS,
    compute_player_match_stats, compute_player_window_stats, time_windows,
};
use rayon::prelude::*;
use replay_parser::parse_replay_from_bytes;
use replay_structs::{DatasetSplit, ReplayPlayer};
use tracing::info;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

/// Share of new replays assigned to the training split.
const TRAINING_RATIO: f64 = 0.9;

#[derive(Parser, Debug)]
#[command(about = "Extract per-player match and window stats for training")]
struct Args {
    /// Whole-match stats CSV.
    #[arg(long, default_value = "data/match_stats.csv")]
    match_out: PathBuf,

    /// Window stats CSV.
    #[arg(long, default_value = "data/window_stats.csv")]
    window_out: PathBuf,

    /// Account id per (replay, slot) CSV.
    #[arg(long, default_value = "data/accounts.csv")]
    accounts_out: PathBuf,

    /// Window length in seconds of live play. Must match what the app uses.
    #[arg(long, default_value_t = 60.0)]
    window_seconds: f32,

    /// Only take this many replays per split (smoke runs).
    #[arg(long)]
    limit: Option<usize>,

    /// Parsing threads.
    #[arg(long, default_value_t = 16)]
    threads: usize,
}

/// A replay ready to be parsed.
struct PendingReplay {
    replay_id: Uuid,
    split: DatasetSplit,
    file_path: String,
    roster: PlayerRoster,
    targets: [f32; TOTAL_SLOTS],
    /// "platform:id" per slot, from the Ballchasing metadata.
    accounts: [Option<String>; TOTAL_SLOTS],
}

/// One output row.
struct StatsRow {
    replay_id: Uuid,
    split: DatasetSplit,
    slot: usize,
    target_mmr: f32,
    stats: PlayerMatchStats,
    window_index: Option<usize>,
}

/// Both CSVs' rows for one replay.
#[derive(Default)]
struct ReplayRows {
    whole_match: Vec<StatsRow>,
    windows: Vec<StatsRow>,
}

/// Slot names and labels in the app's order (blue then orange, each sorted by name).
fn roster_and_targets(players: &[ReplayPlayer]) -> PendingSlots {
    let mut blue: Vec<&ReplayPlayer> = players.iter().filter(|p| p.team == 0).collect();
    let mut orange: Vec<&ReplayPlayer> = players.iter().filter(|p| p.team == 1).collect();
    blue.sort_by(|a, b| a.player_name.cmp(&b.player_name));
    orange.sort_by(|a, b| a.player_name.cmp(&b.player_name));
    let player_at = |slot: usize| -> Option<&ReplayPlayer> {
        if slot < SLOTS_PER_TEAM {
            blue.get(slot).copied()
        } else {
            orange.get(slot - SLOTS_PER_TEAM).copied()
        }
    };
    PendingSlots {
        roster: PlayerRoster {
            names: core::array::from_fn(|slot| {
                player_at(slot).map_or_else(String::new, |p| p.player_name.clone())
            }),
        },
        targets: core::array::from_fn(|slot| {
            player_at(slot)
                .filter(|p| p.rank_known)
                .map_or(0.0, |p| p.rank_division.mmr_middle() as f32)
        }),
    }
}

/// Roster and labels for one replay.
struct PendingSlots {
    roster: PlayerRoster,
    targets: [f32; TOTAL_SLOTS],
}

fn to_rows(
    pending: &PendingReplay,
    stats: [Option<PlayerMatchStats>; TOTAL_SLOTS],
    window_index: Option<usize>,
) -> impl Iterator<Item = StatsRow> + '_ {
    stats
        .into_iter()
        .enumerate()
        .filter_map(move |(slot, stats)| {
            Some(StatsRow {
                replay_id: pending.replay_id,
                split: pending.split,
                slot,
                target_mmr: pending.targets.get(slot).copied().unwrap_or(0.0),
                stats: stats?,
                window_index,
            })
        })
}

fn process(pending: &PendingReplay, base_path: &Path, window_seconds: f32) -> ReplayRows {
    let bytes = std::fs::read(base_path.join(&pending.file_path))
        .or_else(|_| std::fs::read(base_path.join("replays").join(&pending.file_path)));
    let Ok(bytes) = bytes else {
        return ReplayRows::default();
    };
    let Ok(parsed) = parse_replay_from_bytes(&bytes) else {
        return ReplayRows::default();
    };
    let whole_match = to_rows(
        pending,
        compute_player_match_stats(&parsed, &pending.roster),
        None,
    )
    .collect();
    let windows = time_windows(&parsed, window_seconds)
        .into_iter()
        .enumerate()
        .flat_map(|(index, range)| {
            to_rows(
                pending,
                compute_player_window_stats(&parsed, &pending.roster, range),
                Some(index),
            )
            .collect::<Vec<_>>()
        })
        .collect();
    ReplayRows {
        whole_match,
        windows,
    }
}

/// "platform:id" of each roster slot, matched by team and name in the Ballchasing metadata.
fn account_ids(
    metadata: &serde_json::Value,
    roster: &PlayerRoster,
) -> [Option<String>; TOTAL_SLOTS] {
    core::array::from_fn(|slot| {
        let name = roster.names.get(slot).filter(|name| !name.is_empty())?;
        let team = if slot < SLOTS_PER_TEAM {
            "blue"
        } else {
            "orange"
        };
        let player = metadata
            .get(team)?
            .get("players")?
            .as_array()?
            .iter()
            .find(|player| player.get("name").and_then(serde_json::Value::as_str) == Some(name))?;
        let id = player.get("id")?;
        let platform = id.get("platform")?.as_str()?;
        let account = id.get("id")?.as_str()?;
        (!account.is_empty()).then(|| format!("{platform}:{account}"))
    })
}

fn write_accounts_csv(path: &Path, pending: &[PendingReplay]) -> Result<()> {
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?,
    );
    writeln!(out, "replay_id,slot,account")?;
    for replay in pending {
        for (slot, account) in replay.accounts.iter().enumerate() {
            if let Some(account) = account {
                // Ids are alphanumeric; quote anyway so a stray comma cannot shift columns.
                writeln!(
                    out,
                    "{},{slot},\"{}\"",
                    replay.replay_id,
                    account.replace('"', "")
                )?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

fn write_csv(path: &Path, rows: &[StatsRow], windowed: bool) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?,
    );
    write!(out, "replay_id,split,slot,target_mmr,live_seconds")?;
    for stat in MatchStat::all() {
        write!(out, ",{stat}")?;
    }
    if windowed {
        write!(out, ",window_index")?;
    }
    writeln!(out)?;
    for row in rows {
        let split = match row.split {
            DatasetSplit::Training => "training",
            DatasetSplit::Evaluation => "evaluation",
        };
        write!(
            out,
            "{},{split},{},{:.1},{:.2}",
            row.replay_id, row.slot, row.target_mmr, row.stats.live_seconds
        )?;
        for value in &row.stats.values {
            write!(out, ",{value:.6}")?;
        }
        if let Some(window_index) = row.window_index {
            write!(out, ",{window_index}")?;
        }
        writeln!(out)?;
    }
    out.flush()?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();
    let args = Args::parse();
    let database_url =
        std::env::var("DATABASE_URL").context("DATABASE_URL environment variable is required")?;
    initialize_pool(&database_url).await?;

    // Newly downloaded replays have no split yet; give them one (90 % training) so they are
    // used, and so the evaluation split stays a fixed, never-trained-on set.
    let assigned = assign_dataset_splits(TRAINING_RATIO).await?;
    if assigned.training + assigned.evaluation > 0 {
        info!(
            training = assigned.training,
            evaluation = assigned.evaluation,
            "assigned dataset splits to new replays"
        );
    }

    let mut pending = Vec::new();
    for split in [DatasetSplit::Training, DatasetSplit::Evaluation] {
        let mut replays = list_replays_by_split(split).await?;
        // The query has no ORDER BY; sort so `--limit` and the output are reproducible.
        replays.sort_by_key(|replay| replay.id);
        if let Some(limit) = args.limit {
            replays.truncate(limit);
        }
        info!(?split, replays = replays.len(), "listing players");
        for replay in replays {
            let players = list_replay_players_by_replay(replay.id).await?;
            if players.is_empty() {
                continue;
            }
            let PendingSlots { roster, targets } = roster_and_targets(&players);
            let accounts = account_ids(&replay.metadata, &roster);
            pending.push(PendingReplay {
                replay_id: replay.id,
                split,
                file_path: replay.file_path,
                roster,
                targets,
                accounts,
            });
        }
    }
    info!(replays = pending.len(), "parsing");

    let base_path = get_base_path();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    let processed = std::sync::atomic::AtomicUsize::new(0);
    let per_replay: Vec<ReplayRows> = pool.install(|| {
        pending
            .par_iter()
            .map(|replay| {
                let done = processed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if done.is_multiple_of(2000) {
                    info!(done, total = pending.len(), "progress");
                }
                process(replay, &base_path, args.window_seconds)
            })
            .collect()
    });

    let mut whole_match = Vec::new();
    let mut windows = Vec::new();
    for rows in per_replay {
        whole_match.extend(rows.whole_match);
        windows.extend(rows.windows);
    }
    write_csv(&args.match_out, &whole_match, false)?;
    write_accounts_csv(&args.accounts_out, &pending)?;
    write_csv(&args.window_out, &windows, true)?;
    println!(
        "extract_stats: {} replays → {} match rows ({}), {} window rows ({})",
        pending.len(),
        whole_match.len(),
        args.match_out.display(),
        windows.len(),
        args.window_out.display()
    );
    Ok(())
}
