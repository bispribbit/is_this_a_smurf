#![expect(
    clippy::indexing_slicing,
    reason = "slot arrays indexed by slots below TOTAL_SLOTS, stat arrays by stat indices"
)]

//! Hand-written smurf references for the experiments. Labels cannot confirm a smurf (their
//! label is the smurf account's), so the smurf flag is measured against two rules instead:
//!
//! - **mechanical smurfs**: labelled Gold II or lower, Champion I mechanics on most aerial
//!   stats, and their team won by at least two;
//! - **dominant smurfs**: labelled Gold II or lower, their team won by at least three, and
//!   they had the most possession and touches in the lobby.
//!
//! A rule that picks training labels or flags players can no longer grade them, so an
//! experiment using one keeps the other as its yardstick.

use feature_extractor::{MatchStat, SLOTS_PER_TEAM};
use skill_model::coaching::tier_index;

use crate::dataset::{Lobby, Partition};
use crate::label_cleaning::SlotScores;

/// Highest tier a mechanical or dominant smurf can be labelled with (Gold II).
pub const OBVIOUS_MAXIMUM_TIER: usize = 7;

/// Tier whose median mechanics a mechanical smurf must reach (Champion I).
pub const OBVIOUS_REFERENCE_TIER: usize = 15;

/// Aerial mechanics compared with the reference tier.
pub const MECHANIC_STATS: [MatchStat; 6] = [
    MatchStat::FastAerialFraction,
    MatchStat::AirRollFraction,
    MatchStat::AerialTouchFraction,
    MatchStat::HighAirFraction,
    MatchStat::AerialDodgesPerMinute,
    MatchStat::WavedashLikeDodgesPerMinute,
];

/// How many of [`MECHANIC_STATS`] must reach the reference tier's median.
pub const MINIMUM_MECHANICS_AT_REFERENCE: usize = 4;

/// Goals a mechanical smurf's team must win by.
pub const MINIMUM_GOAL_DIFFERENCE: f32 = 2.0;

/// Goals a dominant smurf's team must win by.
pub const DOMINANT_GOAL_DIFFERENCE: f32 = 3.0;

/// Labels at or above this tier (Diamond I) make a flag most likely wrong.
pub const HIGH_RANK_TIER: usize = 12;

/// Scores as tier indices, so a margin means the same number of divisions at every rank.
pub fn as_tiers(scores: SlotScores) -> SlotScores {
    scores.map(|score| score.map(|mmr| tier_index(mmr) as f32))
}

pub fn median(values: &mut [f32]) -> f32 {
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
pub fn reference_mechanics(lobbies: &[Lobby]) -> [f32; MECHANIC_STATS.len()] {
    core::array::from_fn(|index| {
        let stat = MECHANIC_STATS[index].index();
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

pub fn team_goal_difference(lobby: &Lobby, slot: usize) -> f32 {
    let goals = MatchStat::Goals.index();
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
pub fn is_mechanical_smurf(
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
        .filter(|(stat, median)| **median > 0.0 && stats[stat.index()] >= **median)
        .count();
    at_reference >= MINIMUM_MECHANICS_AT_REFERENCE
        && team_goal_difference(lobby, slot) >= MINIMUM_GOAL_DIFFERENCE
}

/// Labelled Gold II or lower, won by three, and the most possession and touches in the
/// lobby.
pub fn is_dominant_smurf(lobby: &Lobby, slot: usize) -> bool {
    let label = lobby.targets[slot];
    if label <= 0.0 || tier_index(label) > OBVIOUS_MAXIMUM_TIER {
        return false;
    }
    let leads = |stat: MatchStat| {
        let stat = stat.index();
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
        && leads(MatchStat::PossessionFraction)
        && leads(MatchStat::TouchesPerMinute)
}
