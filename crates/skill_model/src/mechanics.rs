//! Named mechanics a player does far more than their lobby's rank: per rank, how often each
//! mechanic is done, and which ones a player is out of distribution on.
//!
//! A smurf is not only a Bronze doing Champion things: a Gold smurf in a Diamond lobby is
//! out of distribution for Diamond. So the bar is per tier: the [`MechanicTable`] holds,
//! for every tier, each mechanic's median and a high quantile among players labelled with
//! that tier (cleaned of likely smurfs). A player is above their lobby on a mechanic when
//! their rate beats the lobby tier's high quantile.

use feature_extractor::{MATCH_STAT_COUNT, MATCH_STAT_NAMES, TOTAL_SLOTS};
use serde::{Deserialize, Serialize};

use crate::coaching::tier_index;

/// Quantile of the lobby tier a mechanic rate must beat to count as far above it.
pub const MECHANIC_HIGH_QUANTILE: f32 = 0.95;

/// One named mechanic: the stat that measures it and how the card names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mechanic {
    pub stat: &'static str,
    pub label: &'static str,
}

/// Mechanics checked against the lobby's tier, in [`MechanicTier`] order.
pub const MECHANICS: &[Mechanic] = &[
    Mechanic {
        stat: "fast_aerial_fraction",
        label: "fast aerials",
    },
    Mechanic {
        stat: "speed_flips_per_minute",
        label: "speed flips",
    },
    Mechanic {
        stat: "half_flips_per_minute",
        label: "half flips",
    },
    Mechanic {
        stat: "wavedashes_per_minute",
        label: "wavedashes",
    },
    Mechanic {
        stat: "flip_resets_per_minute",
        label: "flip resets",
    },
    Mechanic {
        stat: "ceiling_touches_per_minute",
        label: "ceiling plays",
    },
    Mechanic {
        stat: "dribble_fraction",
        label: "dribbles",
    },
    Mechanic {
        stat: "flicks_per_minute",
        label: "flicks",
    },
    Mechanic {
        stat: "air_dribble_touches_per_minute",
        label: "air dribbles",
    },
    Mechanic {
        stat: "double_taps_per_minute",
        label: "double taps",
    },
    Mechanic {
        stat: "power_shots_per_minute",
        label: "power shots",
    },
];

/// Singular display name per tier index (0 = Bronze I … 21 = SSL).
pub const TIER_SINGULARS: [&str; 22] = [
    "Bronze I",
    "Bronze II",
    "Bronze III",
    "Silver I",
    "Silver II",
    "Silver III",
    "Gold I",
    "Gold II",
    "Gold III",
    "Platinum I",
    "Platinum II",
    "Platinum III",
    "Diamond I",
    "Diamond II",
    "Diamond III",
    "Champ I",
    "Champ II",
    "Champ III",
    "GC I",
    "GC II",
    "GC III",
    "SSL",
];

/// One tier's reference rates, aligned with [`MECHANICS`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MechanicTier {
    /// Median rate among players labelled with this tier.
    pub medians: Vec<f32>,
    /// The high quantile the table was built with: beating it is out of distribution.
    pub highs: Vec<f32>,
}

/// Reference rates for every tier, Bronze I (index 0) to Supersonic Legend (index 21).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MechanicTable {
    pub tiers: Vec<MechanicTier>,
}

/// One [`MechanicTable`] per playlist size (index = players per team − 1).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MechanicTables {
    pub by_players_per_team: Vec<MechanicTable>,
}

/// A mechanic a player does more than their lobby's tier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MechanicEvidence {
    pub mechanic: &'static Mechanic,
    /// Highest tier whose median rate the player reaches.
    pub typical_tier: usize,
    pub rate: f32,
}

/// Tier of the players other than `slot`: the median of their whole-match tiers, rounded
/// up between two.
fn others_tier(match_mmr: &[Option<f32>; TOTAL_SLOTS], slot: usize) -> Option<usize> {
    let mut tiers: Vec<usize> = match_mmr
        .iter()
        .enumerate()
        .filter(|(other, _)| *other != slot)
        .filter_map(|(_, mmr)| mmr.map(tier_index))
        .collect();
    tiers.sort_unstable();
    let middle = tiers.len() / 2;
    let upper = *tiers.get(middle)?;
    Some(if tiers.len().is_multiple_of(2) {
        tiers
            .get(middle.wrapping_sub(1))
            .map_or(upper, |lower| (lower + upper).div_ceil(2))
    } else {
        upper
    })
}

/// Per slot: the mechanics the player does more than the other players' tier's high
/// quantile, highest typical tier first. Empty without a table or other scored players.
#[must_use]
pub fn mechanics_above_lobby(
    table: Option<&MechanicTable>,
    players: &[Option<&[f32; MATCH_STAT_COUNT]>; TOTAL_SLOTS],
    match_mmr: &[Option<f32>; TOTAL_SLOTS],
) -> [Vec<MechanicEvidence>; TOTAL_SLOTS] {
    core::array::from_fn(|slot| {
        let Some(Some(stats)) = players.get(slot) else {
            return Vec::new();
        };
        table
            .zip(others_tier(match_mmr, slot))
            .map(|found| found.0.above_tier(stats.as_slice(), found.1))
            .unwrap_or_default()
    })
}

impl MechanicTables {
    /// The table for a playlist size, or the standard table when that size has none.
    #[must_use]
    pub fn for_players_per_team(&self, players_per_team: usize) -> Option<&MechanicTable> {
        let has_tiers = |table: &&MechanicTable| !table.tiers.is_empty();
        players_per_team
            .checked_sub(1)
            .and_then(|index| self.by_players_per_team.get(index))
            .filter(has_tiers)
            .or_else(|| self.by_players_per_team.get(2).filter(has_tiers))
    }
}

impl MechanicTable {
    /// Mechanics on which `stats` beats `lobby_tier`'s high quantile, highest typical tier
    /// first.
    #[must_use]
    pub fn above_tier(&self, stats: &[f32], lobby_tier: usize) -> Vec<MechanicEvidence> {
        let Some(reference) = self.tiers.get(lobby_tier) else {
            return Vec::new();
        };
        let mut evidence: Vec<MechanicEvidence> = MECHANICS
            .iter()
            .enumerate()
            .filter_map(|(index, mechanic)| {
                let stat = MATCH_STAT_NAMES
                    .iter()
                    .position(|name| *name == mechanic.stat)?;
                let rate = *stats.get(stat)?;
                let high = *reference.highs.get(index)?;
                (rate > 0.0 && rate > high).then(|| MechanicEvidence {
                    mechanic,
                    typical_tier: self.typical_tier(index, rate),
                    rate,
                })
            })
            .collect();
        evidence.sort_by_key(|item| core::cmp::Reverse(item.typical_tier));
        evidence
    }

    /// Highest tier whose median rate of mechanic `index` is at most `rate`.
    fn typical_tier(&self, index: usize, rate: f32) -> usize {
        self.tiers
            .iter()
            .enumerate()
            .filter(|(_, tier)| {
                tier.medians
                    .get(index)
                    .is_some_and(|median| *median <= rate)
            })
            .map(|(tier, _)| tier)
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats_with(stat: &str, value: f32) -> Vec<f32> {
        MATCH_STAT_NAMES
            .iter()
            .map(|name| if *name == stat { value } else { 0.0 })
            .collect()
    }

    #[test]
    fn beating_the_lobby_tiers_high_quantile_is_evidence() {
        let table = MechanicTable {
            tiers: (0..22)
                .map(|tier| MechanicTier {
                    medians: vec![tier as f32 * 0.1; MECHANICS.len()],
                    highs: vec![(tier as f32).mul_add(0.1, 0.3); MECHANICS.len()],
                })
                .collect(),
        };
        let stats = stats_with("speed_flips_per_minute", 1.5);
        let evidence = table.above_tier(&stats, 2);
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].mechanic.label, "speed flips");
        assert_eq!(evidence[0].typical_tier, 15);
        assert_eq!(table.above_tier(&stats, 12).len(), 0);
    }
}
