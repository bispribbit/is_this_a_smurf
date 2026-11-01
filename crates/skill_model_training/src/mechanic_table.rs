#![expect(
    clippy::indexing_slicing,
    reason = "per-tier vectors of length TIER_COUNT indexed by tier < TIER_COUNT"
)]

//! Builds the per-tier mechanic rates behind the "does far more than their lobby" check
//! ([`skill_model::mechanics`]).

use skill_model::coaching::tier_index;
use skill_model::mechanics::{MECHANICS, MechanicTable, MechanicTables, MechanicTier};

use crate::dataset::{Lobby, Partition};

/// Number of tiers, Bronze I to Supersonic Legend.
const TIER_COUNT: usize = 22;

/// Players a tier needs for its quantiles to be used. Thinner tiers borrow the nearest tier
/// that has enough.
const MINIMUM_PLAYERS_PER_TIER: usize = 50;

/// Largest playlist size (standard).
const MAXIMUM_PLAYERS_PER_TEAM: usize = 3;

fn quantile(sorted: &[f32], fraction: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f32 * fraction).round() as usize;
    sorted.get(index).copied().unwrap_or(0.0)
}

/// One table per playlist size, from the training partition's labels. Pass labels already
/// cleaned of likely smurfs: smurfs are exactly the high tail at low tiers.
#[must_use]
pub fn build_mechanic_tables(lobbies: &[Lobby], high_quantile: f32) -> MechanicTables {
    MechanicTables {
        by_players_per_team: (1..=MAXIMUM_PLAYERS_PER_TEAM)
            .map(|players_per_team| {
                let playlist: Vec<&Lobby> = lobbies
                    .iter()
                    .filter(|lobby| lobby.players_per_team() == Some(players_per_team))
                    .collect();
                build_mechanic_table(&playlist, high_quantile)
            })
            .collect(),
    }
}

fn build_mechanic_table(lobbies: &[&Lobby], high_quantile: f32) -> MechanicTable {
    let columns: Vec<usize> = MECHANICS
        .iter()
        .map(|mechanic| mechanic.stat.index())
        .collect();
    // values[tier][mechanic] = every labelled player's rate.
    let mut values: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); MECHANICS.len()]; TIER_COUNT];
    for lobby in lobbies
        .iter()
        .filter(|lobby| lobby.partition == Partition::Training)
    {
        for slot in lobby.labelled_slots() {
            let Some(stats) = lobby.stats[slot] else {
                continue;
            };
            let tier_values = &mut values[tier_index(lobby.targets[slot])];
            for (mechanic_values, column) in tier_values.iter_mut().zip(&columns) {
                if let Some(value) = stats.get(*column) {
                    mechanic_values.push(*value);
                }
            }
        }
    }
    let players_per_tier: Vec<usize> = values
        .iter()
        .map(|tier_values| tier_values.first().map_or(0, Vec::len))
        .collect();
    let populated: Vec<usize> = (0..TIER_COUNT)
        .filter(|&tier| players_per_tier[tier] >= MINIMUM_PLAYERS_PER_TIER)
        .collect();
    if populated.is_empty() {
        return MechanicTable::default();
    }
    let mut tiers: Vec<MechanicTier> = values
        .iter_mut()
        .map(|tier_values| {
            let mut tier = MechanicTier::default();
            for mechanic_values in tier_values.iter_mut() {
                mechanic_values.sort_by(f32::total_cmp);
                tier.medians.push(quantile(mechanic_values, 0.5));
                tier.highs.push(quantile(mechanic_values, high_quantile));
            }
            tier
        })
        .collect();
    for tier in 0..TIER_COUNT {
        if players_per_tier[tier] >= MINIMUM_PLAYERS_PER_TIER {
            continue;
        }
        if let Some(&nearest) = populated.iter().min_by_key(|&&other| other.abs_diff(tier)) {
            tiers[tier] = tiers[nearest].clone();
        }
    }
    MechanicTable { tiers }
}
