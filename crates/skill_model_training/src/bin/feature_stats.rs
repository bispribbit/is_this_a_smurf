#![expect(
    clippy::indexing_slicing,
    reason = "tier arrays indexed by tier indices below TIER_COUNT and stat rows by MatchStat::index"
)]

//! Every match stat, per playlist: how it relates to rank and how much it varies within one.
//!
//! Reads the whole-match stats CSV written by `extract_stats`, keeps players with a known
//! rank and over two minutes of play, and writes one row per (stat, playlist) to
//! `docs/features.csv`. `docs/features.md` explains the columns and what they say about
//! each stat.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin feature_stats

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use feature_extractor::{MATCH_STAT_COUNT, MatchStat};
use replay_structs::{Rank, TeamSize};
use skill_model::coaching::tier_index;

/// Bronze I to SSL.
const TIER_COUNT: usize = 22;
/// Players with less live play are left out: their rates are a handful of events.
const MINIMUM_LIVE_SECONDS: f32 = 120.0;
/// Tiers with fewer players do not count toward the average within-tier spread.
const MINIMUM_PLAYERS_FOR_SPREAD: usize = 30;
/// SSL.
const HIGHEST_TIER: usize = TIER_COUNT - 1;

#[derive(Parser, Debug)]
#[command(about = "Measure every match stat against rank, per playlist")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

    #[arg(long, default_value = "docs/features.csv")]
    out: PathBuf,
}

/// One labelled player-match.
struct Player {
    mmr: f32,
    tier: usize,
    stats: [f32; MATCH_STAT_COUNT],
}

/// What one stat says about rank in one playlist.
struct StatSummary {
    spearman_rho: f64,
    eta_squared: f64,
    separation: f64,
    within_tier_iqr: f64,
    p10: f64,
    p50: f64,
    p90: f64,
    nonzero_share_lowest: f64,
    nonzero_share_highest: f64,
    tier_medians: [f64; TIER_COUNT],
}

fn read_players(path: &PathBuf) -> Result<Vec<Player>> {
    let mut reader =
        csv::Reader::from_path(path).with_context(|| format!("opening {}", path.display()))?;
    let headers = reader.headers()?.clone();
    let column = |name: &str| {
        headers
            .iter()
            .position(|header| header == name)
            .with_context(|| format!("{}: no {name} column", path.display()))
    };
    let mmr_column = column("target_mmr")?;
    let live_seconds_column = column("live_seconds")?;
    let stat_columns: Vec<usize> = MatchStat::all()
        .map(|stat| column(stat.name()))
        .collect::<Result<_>>()?;
    let mut players = Vec::new();
    for record in reader.records() {
        let record = record?;
        let number = |index: usize| -> f32 {
            record
                .get(index)
                .and_then(|value| value.parse().ok())
                .unwrap_or(0.0)
        };
        let mmr = number(mmr_column);
        if mmr <= 0.0 || number(live_seconds_column) < MINIMUM_LIVE_SECONDS {
            continue;
        }
        let mut stats = [0.0; MATCH_STAT_COUNT];
        for (value, &index) in stats.iter_mut().zip(&stat_columns) {
            *value = number(index);
        }
        players.push(Player {
            mmr,
            tier: tier_index(mmr).min(HIGHEST_TIER),
            stats,
        });
    }
    Ok(players)
}

/// Value at quantile `quantile` of sorted `values` (nearest rank).
fn quantile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

/// Ranks with ties sharing their average rank.
fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut result = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start;
        while end + 1 < order.len()
            && values[order[end + 1]]
                .total_cmp(&values[order[start]])
                .is_eq()
        {
            end += 1;
        }
        let shared = (start + end) as f64 / 2.0;
        for &index in &order[start..=end] {
            result[index] = shared;
        }
        start = end + 1;
    }
    result
}

fn pearson(first: &[f64], second: &[f64]) -> f64 {
    let count = first.len() as f64;
    let first_mean = first.iter().sum::<f64>() / count;
    let second_mean = second.iter().sum::<f64>() / count;
    let mut covariance = 0.0;
    let mut first_spread = 0.0;
    let mut second_spread = 0.0;
    for (first_value, second_value) in first.iter().zip(second) {
        let first_deviation = first_value - first_mean;
        let second_deviation = second_value - second_mean;
        covariance = first_deviation.mul_add(second_deviation, covariance);
        first_spread = first_deviation.mul_add(first_deviation, first_spread);
        second_spread = second_deviation.mul_add(second_deviation, second_spread);
    }
    if first_spread == 0.0 || second_spread == 0.0 {
        return f64::NAN;
    }
    covariance / (first_spread.sqrt() * second_spread.sqrt())
}

fn nonzero_share(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().filter(|value| **value != 0.0).count() as f64 / values.len() as f64
}

fn summarize(players: &[&Player], mmr_ranks: &[f64], stat: MatchStat) -> StatSummary {
    let values: Vec<f64> = players
        .iter()
        .map(|player| f64::from(player.stats[stat.index()]))
        .collect();
    let mut by_tier: [Vec<f64>; TIER_COUNT] = core::array::from_fn(|_| Vec::new());
    for (player, value) in players.iter().zip(&values) {
        by_tier[player.tier].push(*value);
    }
    for tier_values in &mut by_tier {
        tier_values.sort_by(f64::total_cmp);
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let total: f64 = values.iter().map(|value| (value - mean).powi(2)).sum();
    let between: f64 = by_tier
        .iter()
        .filter(|tier_values| !tier_values.is_empty())
        .map(|tier_values| {
            let tier_mean = tier_values.iter().sum::<f64>() / tier_values.len() as f64;
            tier_values.len() as f64 * (tier_mean - mean).powi(2)
        })
        .sum();
    let spreads: Vec<f64> = by_tier
        .iter()
        .filter(|tier_values| tier_values.len() >= MINIMUM_PLAYERS_FOR_SPREAD)
        .map(|tier_values| quantile(tier_values, 0.75) - quantile(tier_values, 0.25))
        .collect();
    let within_tier_iqr = if spreads.is_empty() {
        f64::NAN
    } else {
        spreads.iter().sum::<f64>() / spreads.len() as f64
    };
    let gap = quantile(&by_tier[HIGHEST_TIER], 0.5) - quantile(&by_tier[0], 0.5);
    let separation = if within_tier_iqr > 0.0 {
        gap / within_tier_iqr
    } else if gap == 0.0 {
        f64::NAN
    } else {
        f64::INFINITY.copysign(gap)
    };
    let mut sorted = values.clone();
    sorted.sort_by(f64::total_cmp);
    StatSummary {
        spearman_rho: pearson(&ranks(&values), mmr_ranks),
        eta_squared: if total > 0.0 {
            between / total
        } else {
            f64::NAN
        },
        separation,
        within_tier_iqr,
        p10: quantile(&sorted, 0.1),
        p50: quantile(&sorted, 0.5),
        p90: quantile(&sorted, 0.9),
        nonzero_share_lowest: nonzero_share(&by_tier[0]),
        nonzero_share_highest: nonzero_share(&by_tier[HIGHEST_TIER]),
        tier_medians: core::array::from_fn(|tier| quantile(&by_tier[tier], 0.5)),
    }
}

/// Four significant digits, trailing zeros dropped.
fn format_number(value: f64) -> String {
    if !value.is_finite() {
        return if value.is_nan() {
            "nan".to_string()
        } else if value > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let magnitude = value.abs().log10().floor() as i32;
    let decimals = usize::try_from((3 - magnitude).max(0)).unwrap_or(0);
    let text = format!("{value:.decimals$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let players = read_players(&args.match_stats)?;
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&args.out)
            .with_context(|| format!("creating {}", args.out.display()))?,
    );
    write!(
        out,
        "stat,playlist,players,spearman_rho,eta_squared,separation,within_tier_iqr,p10,p50,p90,nonzero_share_b1,nonzero_share_ssl"
    )?;
    for tier in 0..TIER_COUNT {
        write!(out, ",median_{}", Rank::from_tier_index(tier).short_name())?;
    }
    writeln!(out)?;

    let playlists: Vec<TeamSize> = TeamSize::all().collect();
    let by_playlist: Vec<Vec<&Player>> = playlists
        .iter()
        .map(|playlist| {
            players
                .iter()
                .filter(|player| {
                    player.stats[MatchStat::PlayersPerTeam.index()].round() as usize
                        == playlist.players_per_team()
                })
                .collect()
        })
        .collect();
    let mmr_ranks: Vec<Vec<f64>> = by_playlist
        .iter()
        .map(|group| {
            ranks(
                &group
                    .iter()
                    .map(|player| f64::from(player.mmr))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    for stat in MatchStat::all().filter(|stat| *stat != MatchStat::PlayersPerTeam) {
        for ((playlist, group), group_ranks) in playlists.iter().zip(&by_playlist).zip(&mmr_ranks) {
            if group.is_empty() {
                continue;
            }
            let summary = summarize(group, group_ranks, stat);
            let numbers = [
                summary.spearman_rho,
                summary.eta_squared,
                summary.separation,
                summary.within_tier_iqr,
                summary.p10,
                summary.p50,
                summary.p90,
                summary.nonzero_share_lowest,
                summary.nonzero_share_highest,
            ];
            write!(out, "{stat},{playlist},{}", group.len())?;
            for number in numbers.iter().chain(&summary.tier_medians) {
                write!(out, ",{}", format_number(*number))?;
            }
            writeln!(out)?;
        }
    }
    out.flush()?;
    println!("wrote {}", args.out.display());
    Ok(())
}
