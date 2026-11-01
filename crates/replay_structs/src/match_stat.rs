//! The per-player match stats, shared by the extractor that computes them, the models
//! that read them and the training tools.

use strum::{EnumCount, EnumIter, EnumString, IntoEnumIterator, IntoStaticStr, VariantArray};

/// One per-player match stat. The declaration order is the order of a stat row
/// (`feature_extractor::PlayerMatchStats::values`); the snake-case name is what CSVs and
/// model bundles store.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    EnumString,
    VariantArray,
    IntoStaticStr,
    strum::Display,
)]
#[strum(serialize_all = "snake_case")]
pub enum MatchStat {
    AerialDodgesPerMinute,
    AerialTouchFraction,
    AirborneUprightFraction,
    AirDribbleTouchesPerMinute,
    AirRollFraction,
    BehindBallFraction,
    BigPadsPerMinute,
    BoostCollectedPerMinute,
    BoostEmptyFraction,
    BoostFullFraction,
    BoostingFraction,
    BoostUsedPerMinute,
    BoostWhileSupersonicFraction,
    BumpsInflictedPerMinute,
    BumpsReceivedPerMinute,
    CeilingTouchesPerMinute,
    CleanLandingFraction,
    ClosestOnTeamFraction,
    DefensiveThirdFraction,
    DemosInflicted,
    DemosReceived,
    DiagonalDodgeFraction,
    DodgesPerMinute,
    DoubleCommitFraction,
    DoubleJumpsPerMinute,
    DoubleTapsPerMinute,
    DribbleFraction,
    FastAerialFraction,
    FastDodgeFraction,
    FiftyFiftiesPerMinute,
    FiftyFiftyWinFraction,
    FlicksPerMinute,
    FlipResetsPerMinute,
    FlipTouchFraction,
    ForfeitedLevelOrAhead,
    ForwardTouchFraction,
    Goals,
    GoalSideAfterTouchFraction,
    GoalSideWhenDefendingFraction,
    GroundFraction,
    HalfFlipsPerMinute,
    HighAirFraction,
    JumpsPerMinute,
    JumpsPressedPerMinute,
    KickoffFirstTouchFraction,
    KickoffWinFraction,
    LandingSpeedKeptFraction,
    LastBackGoalDistanceWhenDefending,
    LastBackOnTeamFraction,
    LastBackWhenDefendingFraction,
    LostByForfeit,
    LowAirFraction,
    MeanAbsoluteSteer,
    MeanAirAngularSpeed,
    MeanBallSpeedAfterTouch,
    MeanBoost,
    MeanCanonicalY,
    MeanDistanceToBall,
    MeanDistanceToNearestTeammate,
    MeanDistanceToOwnGoal,
    MeanHeight,
    MeanSpeed,
    MeanTouchSpeedGain,
    MostForwardOnTeamFraction,
    OffensiveThirdFraction,
    OvercommitFraction,
    OwnGoals,
    PitchNegativeDodgeFraction,
    PitchPositiveDodgeFraction,
    PlayersPerTeam,
    PossessionFraction,
    PowerShotsPerMinute,
    PowerslideFraction,
    ReverseFraction,
    RollDodgeFraction,
    ScoreboardAssists,
    ScoreboardGoalsPerShot,
    ScoreboardSaves,
    ScoreboardScorePerMinute,
    ScoreboardShots,
    ShadowDefenseFraction,
    SlowFraction,
    SlowWithBoostFraction,
    SmallPadsPerMinute,
    SpeedFlipsPerMinute,
    SupersonicFraction,
    SupersonicPerBoost,
    SupportSpacingFraction,
    TeammateBumpsPerMinute,
    TornadoSpinsPerMinute,
    TouchesPerMinute,
    WallFraction,
    WallTouchFraction,
    WavedashesPerMinute,
    WavedashLikeDodgesPerMinute,
}

/// Number of stats per player.
pub const MATCH_STAT_COUNT: usize = MatchStat::COUNT;

impl MatchStat {
    /// Every stat, in row order.
    pub fn all() -> impl Iterator<Item = Self> {
        Self::iter()
    }

    /// The stat at row position `index`.
    #[must_use]
    pub fn from_index(index: usize) -> Option<Self> {
        Self::VARIANTS.get(index).copied()
    }

    /// Position in a stat row.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Snake-case name, as stored in CSVs and model bundles.
    #[must_use]
    pub fn name(self) -> &'static str {
        self.into()
    }

    /// The stat named `name`, if there is one.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        name.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_indices_follow_declaration_order() {
        for (index, stat) in MatchStat::iter().enumerate() {
            assert_eq!(stat.index(), index);
            assert_eq!(MatchStat::from_index(index), Some(stat));
            assert_eq!(MatchStat::from_name(stat.name()), Some(stat));
            assert_eq!(stat.to_string(), stat.name());
        }
        assert_eq!(
            MatchStat::FiftyFiftiesPerMinute.name(),
            "fifty_fifties_per_minute"
        );
    }
}
