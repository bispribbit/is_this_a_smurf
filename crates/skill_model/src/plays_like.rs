//! The rank a player plays like, as a weighted mix of one [`PlaysLikeModel`] per stat group.
//!
//! One model over every own stat lets a strong group (aerials, say) hide a weak one. A
//! smurf stands out in several groups at once, so each group gets its own model, and the
//! smurf flag averages the groups' margins over the lobby, each weighted by how well the
//! group estimates rank (1 / RMSE²). At the same flag rate this caught more smurfs in every
//! playlist and put far fewer flags on Diamond-and-up players (`docs/model.md`).

use feature_extractor::{MATCH_STAT_COUNT, MatchStat, PlayerMatchStats, TOTAL_SLOTS};

use crate::model::{PlaysLikeModel, TabularLayout};

/// One stat group: its name and the stats its model sees (plus `players_per_team`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatGroup {
    pub name: &'static str,
    pub stats: &'static [MatchStat],
}

impl StatGroup {
    /// Own values of the group's stats and `players_per_team`.
    #[must_use]
    pub fn layout(&self) -> TabularLayout {
        TabularLayout {
            stats: self
                .stats
                .iter()
                .chain(&[MatchStat::PlayersPerTeam])
                .map(|stat| stat.index())
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
            MatchStat::LowAirFraction,
            MatchStat::HighAirFraction,
            MatchStat::WallFraction,
            MatchStat::MeanHeight,
            MatchStat::AerialTouchFraction,
            MatchStat::AirborneUprightFraction,
            MatchStat::CleanLandingFraction,
            MatchStat::LandingSpeedKeptFraction,
            MatchStat::AerialDodgesPerMinute,
            MatchStat::AirRollFraction,
            MatchStat::MeanAirAngularSpeed,
            MatchStat::FastAerialFraction,
            MatchStat::DoubleJumpsPerMinute,
        ],
    },
    StatGroup {
        name: "ground and flips",
        stats: &[
            MatchStat::GroundFraction,
            MatchStat::JumpsPerMinute,
            MatchStat::JumpsPressedPerMinute,
            MatchStat::DodgesPerMinute,
            MatchStat::WavedashLikeDodgesPerMinute,
            MatchStat::PitchPositiveDodgeFraction,
            MatchStat::PitchNegativeDodgeFraction,
            MatchStat::RollDodgeFraction,
            MatchStat::DiagonalDodgeFraction,
            MatchStat::FastDodgeFraction,
            MatchStat::PowerslideFraction,
            MatchStat::ReverseFraction,
            MatchStat::MeanAbsoluteSteer,
            MatchStat::MeanBallSpeedAfterTouch,
            MatchStat::MeanTouchSpeedGain,
        ],
    },
    StatGroup {
        name: "speed and boost",
        stats: &[
            MatchStat::MeanSpeed,
            MatchStat::SupersonicFraction,
            MatchStat::SlowFraction,
            MatchStat::MeanBoost,
            MatchStat::BoostEmptyFraction,
            MatchStat::BoostFullFraction,
            MatchStat::BoostCollectedPerMinute,
            MatchStat::BoostUsedPerMinute,
            MatchStat::BigPadsPerMinute,
            MatchStat::SmallPadsPerMinute,
            MatchStat::BoostWhileSupersonicFraction,
            MatchStat::SlowWithBoostFraction,
            MatchStat::SupersonicPerBoost,
            MatchStat::BoostingFraction,
        ],
    },
    StatGroup {
        name: "named mechanics",
        stats: &[
            MatchStat::SpeedFlipsPerMinute,
            MatchStat::HalfFlipsPerMinute,
            MatchStat::WavedashesPerMinute,
            MatchStat::TornadoSpinsPerMinute,
            MatchStat::FlipResetsPerMinute,
            MatchStat::CeilingTouchesPerMinute,
            MatchStat::DribbleFraction,
            MatchStat::FlicksPerMinute,
            MatchStat::AirDribbleTouchesPerMinute,
            MatchStat::DoubleTapsPerMinute,
            MatchStat::WallTouchFraction,
            MatchStat::FlipTouchFraction,
            MatchStat::PowerShotsPerMinute,
            MatchStat::FiftyFiftiesPerMinute,
            MatchStat::FiftyFiftyWinFraction,
            MatchStat::BumpsInflictedPerMinute,
            MatchStat::BumpsReceivedPerMinute,
            MatchStat::TeammateBumpsPerMinute,
            MatchStat::DemosInflicted,
            MatchStat::DemosReceived,
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
