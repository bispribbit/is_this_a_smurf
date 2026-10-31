//! The rank a player plays like, as a weighted mix of one [`PlaysLikeModel`] per stat group.
//!
//! One model over every own stat lets a strong group (aerials, say) hide a weak one. A
//! smurf stands out in several groups at once, so each group gets its own model, and the
//! smurf flag averages the groups' margins over the lobby, each weighted by how well the
//! group estimates rank (1 / RMSE²). At the same flag rate this caught more smurfs in every
//! playlist and put far fewer flags on Diamond-and-up players (`docs/model.md`).

use feature_extractor::{MATCH_STAT_COUNT, MATCH_STAT_NAMES, PlayerMatchStats, TOTAL_SLOTS};

use crate::model::{PlaysLikeModel, TabularLayout};

/// One stat group: its name and the stats its model sees (plus `players_per_team`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatGroup {
    pub name: &'static str,
    pub stats: &'static [&'static str],
}

impl StatGroup {
    /// Own values of the group's stats and `players_per_team`.
    #[must_use]
    pub fn layout(&self) -> TabularLayout {
        TabularLayout {
            stats: self
                .stats
                .iter()
                .chain(&["players_per_team"])
                .filter_map(|name| MATCH_STAT_NAMES.iter().position(|stat| stat == name))
                .collect(),
            ..PlaysLikeModel::own_stats_layout()
        }
    }
}

/// The groups the shipped smurf flag is built from. The first three are movement stats that
/// do not depend on the opponents; the named mechanics include touches, 50/50s and bumps.
pub const STAT_GROUPS: [StatGroup; 4] = [
    StatGroup {
        name: "aerial",
        stats: &[
            "low_air_fraction",
            "high_air_fraction",
            "wall_fraction",
            "mean_height",
            "aerial_touch_fraction",
            "airborne_upright_fraction",
            "aerial_dodges_per_minute",
            "air_roll_fraction",
            "mean_air_angular_speed",
            "fast_aerial_fraction",
            "double_jumps_per_minute",
        ],
    },
    StatGroup {
        name: "ground and flips",
        stats: &[
            "ground_fraction",
            "jumps_per_minute",
            "jumps_pressed_per_minute",
            "dodges_per_minute",
            "wavedash_like_dodges_per_minute",
            "pitch_positive_dodge_fraction",
            "pitch_negative_dodge_fraction",
            "roll_dodge_fraction",
            "diagonal_dodge_fraction",
            "fast_dodge_fraction",
            "powerslide_fraction",
            "reverse_fraction",
            "mean_absolute_steer",
            "mean_ball_speed_after_touch",
            "mean_touch_speed_gain",
        ],
    },
    StatGroup {
        name: "speed and boost",
        stats: &[
            "mean_speed",
            "supersonic_fraction",
            "slow_fraction",
            "mean_boost",
            "boost_empty_fraction",
            "boost_full_fraction",
            "boost_collected_per_minute",
            "boost_used_per_minute",
            "big_pads_per_minute",
            "small_pads_per_minute",
            "boost_while_supersonic_fraction",
            "slow_with_boost_fraction",
            "supersonic_per_boost",
            "boosting_fraction",
        ],
    },
    StatGroup {
        name: "named mechanics",
        stats: &[
            "speed_flips_per_minute",
            "half_flips_per_minute",
            "wavedashes_per_minute",
            "flip_resets_per_minute",
            "ceiling_touches_per_minute",
            "dribble_fraction",
            "flicks_per_minute",
            "air_dribble_touches_per_minute",
            "double_taps_per_minute",
            "wall_touch_fraction",
            "flip_touch_fraction",
            "power_shots_per_minute",
            "fifty_fifties_per_minute",
            "fifty_fifty_win_fraction",
            "bumps_inflicted_per_minute",
            "bumps_received_per_minute",
            "teammate_bumps_per_minute",
            "demos_inflicted",
            "demos_received",
        ],
    },
];

/// One group's model and its weight in the mix.
#[derive(Debug, Clone)]
pub struct PlaysLikeGroup {
    pub name: String,
    /// 1 / RMSE² of the group's model on the early-stopping split.
    pub weight: f32,
    pub model: PlaysLikeModel,
}

/// Every group's model.
#[derive(Debug, Clone, Default)]
pub struct PlaysLikeGroups {
    pub groups: Vec<PlaysLikeGroup>,
}

/// One group's predictions for a lobby.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupPrediction {
    pub weight: f32,
    pub mmr: [Option<f32>; TOTAL_SLOTS],
}

/// Every group's predictions for a lobby.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlaysLikePredictions {
    pub groups: Vec<GroupPrediction>,
}

impl PlaysLikeGroups {
    /// Predicts every present slot from raw stat arrays.
    #[must_use]
    pub fn predict_stats(
        &self,
        players: &[Option<&[f32; MATCH_STAT_COUNT]>; TOTAL_SLOTS],
    ) -> PlaysLikePredictions {
        PlaysLikePredictions {
            groups: self
                .groups
                .iter()
                .map(|group| GroupPrediction {
                    weight: group.weight,
                    mmr: group.model.predict_stats(players),
                })
                .collect(),
        }
    }

    /// Predicts every present slot from per-player match stats.
    #[must_use]
    pub fn predict(
        &self,
        players: &[Option<PlayerMatchStats>; TOTAL_SLOTS],
    ) -> PlaysLikePredictions {
        let values: [Option<&[f32; MATCH_STAT_COUNT]>; TOTAL_SLOTS] =
            core::array::from_fn(|slot| {
                players
                    .get(slot)
                    .and_then(Option::as_ref)
                    .map(|stats| &stats.values)
            });
        self.predict_stats(&values)
    }
}

impl PlaysLikePredictions {
    /// Weighted mean of the groups' MMRs per slot: the "plays like" rank on the card.
    #[must_use]
    pub fn combined_mmr(&self) -> [Option<f32>; TOTAL_SLOTS] {
        core::array::from_fn(|slot| {
            let mut weighted = 0.0;
            let mut weights = 0.0;
            for group in &self.groups {
                if let Some(Some(mmr)) = group.mmr.get(slot) {
                    weighted += group.weight * mmr;
                    weights += group.weight;
                }
            }
            (weights > 0.0).then(|| weighted / weights)
        })
    }
}
