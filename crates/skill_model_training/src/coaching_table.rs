#![expect(
    clippy::indexing_slicing,
    reason = "per-tier vectors of length TIER_COUNT indexed by tier < TIER_COUNT"
)]

//! Builds the per-tier reference table behind the "what the next rank does" roast.

use skill_model::coaching::{
    Better, COACHING_TIPS, CoachingTable, CoachingTables, TierReference, tier_index,
};

use crate::dataset::{Lobby, Partition};

/// Number of tiers, Bronze I to Supersonic Legend.
const TIER_COUNT: usize = 22;

/// Tiers compared at each end of the ladder when checking a tip's direction.
const DIRECTION_CHECK_TIERS: usize = 6;

/// Players a tier needs for its medians to be used. Thinner tiers (rare ranks in duels and
/// doubles) borrow the nearest tier that has enough.
const MINIMUM_PLAYERS_PER_TIER: usize = 50;

/// Robust spread: inter-quartile range / 1.349 (equals the standard deviation for a normal
/// distribution, but ignores the long tails these stats have).
const IQR_TO_SPREAD: f32 = 1.349;

fn quantile(sorted: &[f32], fraction: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f32 * fraction).round() as usize;
    sorted.get(index).copied().unwrap_or(0.0)
}

/// Largest playlist size (standard).
const MAXIMUM_PLAYERS_PER_TEAM: usize = 3;

/// Builds one table per playlist size (duels, doubles, standard).
#[must_use]
pub fn build_coaching_tables(lobbies: &[Lobby]) -> CoachingTables {
    CoachingTables {
        by_players_per_team: (1..=MAXIMUM_PLAYERS_PER_TEAM)
            .map(|players_per_team| {
                let playlist: Vec<&Lobby> = lobbies
                    .iter()
                    .filter(|lobby| lobby.players_per_team() == Some(players_per_team))
                    .collect();
                if playlist.is_empty() {
                    CoachingTable::default()
                } else {
                    build_coaching_table(&playlist)
                }
            })
            .collect(),
    }
}

/// Builds the table from labelled players of the **training** partition (never the
/// evaluation split, so the roast is scored on data it was not built from).
#[must_use]
pub fn build_coaching_table(lobbies: &[&Lobby]) -> CoachingTable {
    let stat_columns: Vec<usize> = COACHING_TIPS.iter().map(|tip| tip.stat.index()).collect();

    // values[tier][tip] = every labelled player's value of that tip's stat.
    let mut values: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); COACHING_TIPS.len()]; TIER_COUNT];
    for lobby in lobbies
        .iter()
        .filter(|l| l.partition == Partition::Training)
    {
        for slot in lobby.labelled_slots() {
            let Some(Some(stats)) = lobby.stats.get(slot) else {
                continue;
            };
            let Some(&target) = lobby.targets.get(slot) else {
                continue;
            };
            let Some(tier_values) = values.get_mut(tier_index(target)) else {
                continue;
            };
            for (tip_values, column) in tier_values.iter_mut().zip(&stat_columns) {
                if let Some(value) = stats.get(*column) {
                    tip_values.push(*value);
                }
            }
        }
    }

    let players_per_tier: Vec<usize> = values
        .iter()
        .map(|tier_values| tier_values.first().map_or(0, Vec::len))
        .collect();
    if players_per_tier
        .iter()
        .all(|&count| count < MINIMUM_PLAYERS_PER_TIER)
    {
        return CoachingTable::default();
    }

    let mut tiers: Vec<TierReference> = values
        .iter_mut()
        .map(|tier_values| {
            let mut medians = Vec::with_capacity(COACHING_TIPS.len());
            let mut spreads = Vec::with_capacity(COACHING_TIPS.len());
            for tip_values in tier_values.iter_mut() {
                tip_values.sort_by(f32::total_cmp);
                medians.push(quantile(tip_values, 0.5));
                spreads.push(
                    (quantile(tip_values, 0.75) - quantile(tip_values, 0.25)) / IQR_TO_SPREAD,
                );
            }
            TierReference { medians, spreads }
        })
        .collect();
    let populated: Vec<usize> = (0..TIER_COUNT)
        .filter(|&tier| players_per_tier[tier] >= MINIMUM_PLAYERS_PER_TIER)
        .collect();
    for tier in 0..TIER_COUNT {
        if players_per_tier[tier] >= MINIMUM_PLAYERS_PER_TIER {
            continue;
        }
        // Borrowing makes the tier identical to its neighbour, so no tip shows progression
        // into it: players compared against it get "nothing to fix" rather than nonsense.
        if let Some(&nearest) = populated.iter().min_by_key(|&&other| other.abs_diff(tier)) {
            tiers[tier] = tiers[nearest].clone();
        }
    }

    let enabled = COACHING_TIPS
        .iter()
        .enumerate()
        .map(|(tip_index, tip)| {
            let mean_over = |range: core::ops::Range<usize>| {
                let count = range.len().max(1) as f32;
                range
                    .filter_map(|tier| tiers.get(tier)?.medians.get(tip_index).copied())
                    .sum::<f32>()
                    / count
            };
            let low = mean_over(0..DIRECTION_CHECK_TIERS);
            let high = mean_over(TIER_COUNT - DIRECTION_CHECK_TIERS..TIER_COUNT);
            match tip.better {
                Better::Higher => high > low,
                Better::Lower => high < low,
            }
        })
        .collect();

    CoachingTable { tiers, enabled }
}
