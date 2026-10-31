#![expect(
    clippy::indexing_slicing,
    reason = "fixed-size stat and slot arrays indexed by bounded layout indices"
)]

//! Tabular per-player skill model: whole-match stats → gradient-boosted trees.
//!
//! The shipped model (see `docs/model.md`). It is deliberately free of `burn` and of any training code so it runs unchanged in the WASM
//! app: a match is parsed, [`feature_extractor::compute_player_match_stats`] summarises
//! each player, and [`TabularSkillModel::predict`] turns the summaries into one MMR
//! estimate per player.
//!
//! Two ensembles are combined:
//!
//! * the **absolute** model predicts each player's MMR from their own stats, the lobby's
//!   mean stats and the difference — its lobby mean is the lobby level;
//! * the **within** model predicts each player's deviation from the lobby's mean label
//!   from deviation and lobby-mean stats — centred on the lobby, it orders the players.
//!
//! Prediction = lobby level + `deviation_scale` × centred deviation, floored at 0 MMR.

use feature_extractor::{MATCH_STAT_COUNT, PlayerMatchStats, SLOTS_PER_TEAM, TOTAL_SLOTS};
use serde::{Deserialize, Serialize};

/// Index of `goals` in [`feature_extractor::MATCH_STAT_NAMES`], used for team goal difference.
const GOALS_STAT: usize = 33;

/// One node of a frozen regression tree.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FrozenNode {
    /// Terminal value.
    Leaf {
        /// Contribution before the learning rate.
        value: f32,
    },
    /// Rows with `features[feature] <= threshold` go left.
    Split {
        /// Feature column.
        feature: usize,
        /// Raw (unbinned) threshold.
        threshold: f32,
        /// Left child index.
        left: usize,
        /// Right child index.
        right: usize,
    },
}

/// A trained gradient-boosted ensemble with raw-value thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenEnsemble {
    /// Starting prediction (training target mean).
    pub base_prediction: f32,
    /// Shrinkage applied to every tree.
    pub learning_rate: f32,
    /// Trees as flat node lists, root at index 0.
    pub trees: Vec<Vec<FrozenNode>>,
}

impl FrozenEnsemble {
    /// Predicts one row.
    #[must_use]
    pub fn predict(&self, features: &[f32]) -> f32 {
        let total: f32 = self
            .trees
            .iter()
            .map(|tree| {
                let mut index = 0;
                loop {
                    match tree.get(index) {
                        Some(FrozenNode::Leaf { value }) => return *value,
                        Some(FrozenNode::Split {
                            feature,
                            threshold,
                            left,
                            right,
                        }) => {
                            let value = features.get(*feature).copied().unwrap_or(0.0);
                            index = if value <= *threshold { *left } else { *right };
                        }
                        None => return 0.0,
                    }
                }
            })
            .sum();
        self.learning_rate.mul_add(total, self.base_prediction)
    }
}

/// Which columns a model sees, in a fixed order shared by training and inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TabularLayout {
    /// Player's own stat value.
    pub own: bool,
    /// Own − lobby mean.
    pub deviation: bool,
    /// Lobby mean.
    pub lobby_mean: bool,
    /// Own − own-team mean, own-team mean − opponent mean, and team goal difference.
    pub team_context: bool,
    /// Stat indices kept, into [`feature_extractor::MATCH_STAT_NAMES`].
    pub stats: Vec<usize>,
}

/// Per-lobby aggregates the layout needs.
#[derive(Debug, Clone)]
pub struct LobbySummary {
    /// Mean stats over present players.
    pub mean: [f32; MATCH_STAT_COUNT],
    /// Mean stats per team (blue, orange) over present players.
    pub team_mean: [[f32; MATCH_STAT_COUNT]; 2],
    /// Goals per team (sum of player goals).
    pub team_goals: [f32; 2],
}

impl LobbySummary {
    /// Summarises the present players of one lobby (blue slots first, then orange).
    #[must_use]
    pub fn new(players: &[Option<&[f32; MATCH_STAT_COUNT]>; TOTAL_SLOTS]) -> Self {
        let mut mean = [0.0; MATCH_STAT_COUNT];
        let mut team_mean = [[0.0; MATCH_STAT_COUNT]; 2];
        let mut team_goals = [0.0; 2];
        let present = players.iter().flatten().count().max(1) as f32;
        for team in 0..2 {
            let members: Vec<&[f32; MATCH_STAT_COUNT]> = players
                .iter()
                .enumerate()
                .filter(|(slot, _)| usize::from(*slot >= SLOTS_PER_TEAM) == team)
                .filter_map(|(_, stats)| *stats)
                .collect();
            for member in &members {
                for (index, value) in member.iter().enumerate() {
                    team_mean[team][index] += value / members.len() as f32;
                    mean[index] += value / present;
                }
                team_goals[team] += member[GOALS_STAT];
            }
        }
        Self {
            mean,
            team_mean,
            team_goals,
        }
    }
}

impl TabularLayout {
    /// Appends one player's feature row to `out`.
    pub fn push_player_features(
        &self,
        lobby: &LobbySummary,
        slot: usize,
        stats: &[f32; MATCH_STAT_COUNT],
        out: &mut Vec<f32>,
    ) {
        for &stat in &self.stats {
            let own = stats[stat];
            let mean = lobby.mean[stat];
            if self.own {
                out.push(own);
            }
            if self.deviation {
                out.push(own - mean);
            }
            if self.lobby_mean {
                out.push(mean);
            }
        }
        if self.team_context {
            let team = usize::from(slot >= SLOTS_PER_TEAM);
            let own_team = &lobby.team_mean[team];
            let opponents = &lobby.team_mean[1 - team];
            for &stat in &self.stats {
                out.push(stats[stat] - own_team[stat]);
                out.push(own_team[stat] - opponents[stat]);
            }
            out.push(lobby.team_goals[team] - lobby.team_goals[1 - team]);
        }
    }

    /// Column names, matching [`Self::push_player_features`].
    #[must_use]
    pub fn column_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for &stat in &self.stats {
            let name = feature_extractor::MATCH_STAT_NAMES
                .get(stat)
                .copied()
                .unwrap_or("?");
            if self.own {
                names.push(format!("own:{name}"));
            }
            if self.deviation {
                names.push(format!("dev:{name}"));
            }
            if self.lobby_mean {
                names.push(format!("lobby:{name}"));
            }
        }
        if self.team_context {
            for &stat in &self.stats {
                let name = feature_extractor::MATCH_STAT_NAMES
                    .get(stat)
                    .copied()
                    .unwrap_or("?");
                names.push(format!("vs_team:{name}"));
                names.push(format!("team_vs_opp:{name}"));
            }
            names.push("team_goal_difference".to_string());
        }
        names
    }
}

/// The shippable model: two ensembles, their layouts, and the deviation scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabularSkillModel {
    /// Stat names the model was trained on, for a compatibility check at load time.
    pub stat_names: Vec<String>,
    /// Layout of the absolute model's input.
    pub absolute_layout: TabularLayout,
    /// Absolute MMR model.
    pub absolute: FrozenEnsemble,
    /// Layout of the within model's input.
    pub within_layout: TabularLayout,
    /// Deviation-from-lobby model.
    pub within: FrozenEnsemble,
    /// Multiplier on centred deviations, fitted on held-out lobbies.
    pub deviation_scale: f32,
}

impl TabularSkillModel {
    /// True when the model was trained on exactly the stats this build computes.
    #[must_use]
    pub fn matches_current_stats(&self) -> bool {
        self.stat_names.len() == MATCH_STAT_COUNT
            && self
                .stat_names
                .iter()
                .zip(feature_extractor::MATCH_STAT_NAMES)
                .all(|(trained, current)| trained == current)
    }

    /// Predicts every present slot from raw stat arrays. Absent slots are `None`.
    #[must_use]
    pub fn predict_stats(
        &self,
        players: &[Option<&[f32; MATCH_STAT_COUNT]>; TOTAL_SLOTS],
    ) -> [Option<f32>; TOTAL_SLOTS] {
        let lobby = LobbySummary::new(players);
        let mut absolute = [None; TOTAL_SLOTS];
        let mut deviation = [None; TOTAL_SLOTS];
        let mut row = Vec::new();
        for (slot, stats) in players.iter().enumerate() {
            let Some(stats) = stats else {
                continue;
            };
            row.clear();
            self.absolute_layout
                .push_player_features(&lobby, slot, stats, &mut row);
            absolute[slot] = Some(self.absolute.predict(&row));
            row.clear();
            self.within_layout
                .push_player_features(&lobby, slot, stats, &mut row);
            deviation[slot] = Some(self.within.predict(&row));
        }
        let present = absolute.iter().flatten().count().max(1) as f32;
        let level = absolute.iter().flatten().sum::<f32>() / present;
        let mean_deviation = deviation.iter().flatten().sum::<f32>() / present;
        core::array::from_fn(|slot| {
            // Ranks start at 0 MMR (Bronze I); a low lobby plus a negative gap can dip below.
            deviation[slot].map(|value| {
                self.deviation_scale
                    .mul_add(value - mean_deviation, level)
                    .max(0.0)
            })
        })
    }

    /// Predicts every present slot from per-player match stats.
    #[must_use]
    pub fn predict(
        &self,
        players: &[Option<PlayerMatchStats>; TOTAL_SLOTS],
    ) -> [Option<f32>; TOTAL_SLOTS] {
        let values: [Option<&[f32; MATCH_STAT_COUNT]>; TOTAL_SLOTS] =
            core::array::from_fn(|slot| players[slot].as_ref().map(|stats| &stats.values));
        self.predict_stats(&values)
    }
}

/// Stats that describe how a player drives, manages boost and handles the car, whoever
/// they play against.
///
/// Left out: anything the opponents shape. Possession, touches, field position, distances,
/// scoreboard, kickoffs, demos and the situational positioning stats all look different
/// when a smurf has the ball in the other half all game: in a duel they are a split with
/// one opponent. `players_per_team` stays so the trees know the playlist.
pub const OPPONENT_INDEPENDENT_STATS: &[&str] = &[
    "mean_speed",
    "supersonic_fraction",
    "slow_fraction",
    "ground_fraction",
    "low_air_fraction",
    "high_air_fraction",
    "wall_fraction",
    "mean_height",
    "jumps_per_minute",
    "mean_boost",
    "boost_empty_fraction",
    "boost_full_fraction",
    "boost_collected_per_minute",
    "boost_used_per_minute",
    "big_pads_per_minute",
    "small_pads_per_minute",
    "boost_while_supersonic_fraction",
    "mean_ball_speed_after_touch",
    "aerial_touch_fraction",
    "mean_touch_speed_gain",
    "airborne_upright_fraction",
    "slow_with_boost_fraction",
    "supersonic_per_boost",
    "jumps_pressed_per_minute",
    "double_jumps_per_minute",
    "dodges_per_minute",
    "wavedash_like_dodges_per_minute",
    "aerial_dodges_per_minute",
    "pitch_positive_dodge_fraction",
    "pitch_negative_dodge_fraction",
    "roll_dodge_fraction",
    "diagonal_dodge_fraction",
    "fast_dodge_fraction",
    "boosting_fraction",
    "powerslide_fraction",
    "air_roll_fraction",
    "reverse_fraction",
    "mean_absolute_steer",
    "mean_air_angular_speed",
    "fast_aerial_fraction",
    "players_per_team",
];

/// Predicts each player's MMR from their own stats only: the rank they *play like*,
/// whatever lobby they are in.
///
/// [`TabularSkillModel`] predicts a lobby level plus a gap learned from rank labels, and
/// those labels hardly ever differ inside a duel or doubles lobby, so it cannot show one
/// player far above the rest. This model has no lobby input, so a smurf's stats speak for
/// themselves. It is noisier per player and only feeds the smurf flag
/// ([`crate::smurf_suspicions`]).
#[derive(Debug, Clone)]
pub struct PlaysLikeModel {
    /// Columns the ensemble sees: own stats only.
    pub layout: TabularLayout,
    pub ensemble: FrozenEnsemble,
}

impl PlaysLikeModel {
    /// The layout the model is trained with: every stat, own values only.
    #[must_use]
    pub fn own_stats_layout() -> TabularLayout {
        TabularLayout {
            own: true,
            deviation: false,
            lobby_mean: false,
            team_context: false,
            stats: (0..MATCH_STAT_COUNT).collect(),
        }
    }

    /// Own values of [`OPPONENT_INDEPENDENT_STATS`] only.
    #[must_use]
    pub fn opponent_independent_layout() -> TabularLayout {
        TabularLayout {
            stats: OPPONENT_INDEPENDENT_STATS
                .iter()
                .filter_map(|name| {
                    feature_extractor::MATCH_STAT_NAMES
                        .iter()
                        .position(|stat| stat == name)
                })
                .collect(),
            ..Self::own_stats_layout()
        }
    }

    /// Predicts every present slot from raw stat arrays. Absent slots are `None`.
    #[must_use]
    pub fn predict_stats(
        &self,
        players: &[Option<&[f32; MATCH_STAT_COUNT]>; TOTAL_SLOTS],
    ) -> [Option<f32>; TOTAL_SLOTS] {
        let lobby = LobbySummary::new(players);
        let mut row = Vec::new();
        core::array::from_fn(|slot| {
            let stats = players[slot]?;
            row.clear();
            self.layout
                .push_player_features(&lobby, slot, stats, &mut row);
            Some(self.ensemble.predict(&row).max(0.0))
        })
    }

    /// Predicts every present slot from per-player match stats.
    #[must_use]
    pub fn predict(
        &self,
        players: &[Option<PlayerMatchStats>; TOTAL_SLOTS],
    ) -> [Option<f32>; TOTAL_SLOTS] {
        let values: [Option<&[f32; MATCH_STAT_COUNT]>; TOTAL_SLOTS] =
            core::array::from_fn(|slot| players[slot].as_ref().map(|stats| &stats.values));
        self.predict_stats(&values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stump(threshold: f32, left: f32, right: f32) -> Vec<FrozenNode> {
        vec![
            FrozenNode::Split {
                feature: 0,
                threshold,
                left: 1,
                right: 2,
            },
            FrozenNode::Leaf { value: left },
            FrozenNode::Leaf { value: right },
        ]
    }

    #[test]
    fn frozen_ensemble_follows_thresholds() {
        let ensemble = FrozenEnsemble {
            base_prediction: 100.0,
            learning_rate: 0.5,
            trees: vec![stump(1.0, -10.0, 10.0), stump(2.0, -4.0, 4.0)],
        };
        assert!((ensemble.predict(&[0.5]) - 0.5f32.mul_add(-14.0, 100.0)).abs() < 1e-5);
        assert!((ensemble.predict(&[1.5]) - 0.5f32.mul_add(6.0, 100.0)).abs() < 1e-5);
        assert!((ensemble.predict(&[3.0]) - 0.5f32.mul_add(14.0, 100.0)).abs() < 1e-5);
    }

    /// The within model's output is centred on the lobby, so the lobby mean prediction is
    /// exactly the absolute model's lobby mean.
    #[test]
    fn lobby_mean_comes_from_the_absolute_model() {
        let layout = TabularLayout {
            own: true,
            deviation: false,
            lobby_mean: false,
            team_context: false,
            stats: vec![0],
        };
        let model = TabularSkillModel {
            stat_names: feature_extractor::MATCH_STAT_NAMES
                .iter()
                .map(ToString::to_string)
                .collect(),
            absolute_layout: layout.clone(),
            absolute: FrozenEnsemble {
                base_prediction: 1000.0,
                learning_rate: 1.0,
                trees: vec![],
            },
            within_layout: layout,
            within: FrozenEnsemble {
                base_prediction: 0.0,
                learning_rate: 1.0,
                trees: vec![stump(0.5, -50.0, 50.0)],
            },
            deviation_scale: 1.0,
        };
        let mut low = [0.0; MATCH_STAT_COUNT];
        let mut high = [0.0; MATCH_STAT_COUNT];
        low[0] = 0.0;
        high[0] = 1.0;
        let mut players = [None; TOTAL_SLOTS];
        players[0] = Some(&low);
        players[1] = Some(&high);
        players[SLOTS_PER_TEAM] = Some(&low);
        players[SLOTS_PER_TEAM + 1] = Some(&high);
        let predictions = model.predict_stats(&players);
        let present: Vec<f32> = predictions.iter().flatten().copied().collect();
        assert_eq!(present.len(), 4);
        let mean = present.iter().sum::<f32>() / 4.0;
        assert!((mean - 1000.0).abs() < 1e-3);
        assert!(predictions[1] > predictions[0]);
        assert!(model.matches_current_stats());
    }
}
