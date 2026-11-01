//! Rocket League rank definitions and MMR mappings.
//!
//! This module provides a comprehensive enum for all Rocket League ranks
//! with their corresponding MMR (Match Making Rating) values.

use core::fmt;
use core::ops::{Add, Sub};
use core::str::FromStr;

use serde::{Deserialize, Serialize};
use strum::{EnumCount, IntoEnumIterator};

/// Represents a Rocket League competitive rank with division.
///
/// Each rank (except Supersonic Legend) has 4 divisions.
/// MMR values are based on the 3v3 ranked playlist.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, strum::EnumIter, strum::EnumCount,
)]
#[cfg_attr(feature = "database", derive(sqlx::Type))]
#[cfg_attr(feature = "database", sqlx(type_name = "rank_division"))]
pub enum RankDivision {
    // Bronze I
    #[cfg_attr(feature = "database", sqlx(rename = "bronze1_division1"))]
    BronzeIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "bronze1_division2"))]
    BronzeIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "bronze1_division3"))]
    BronzeIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "bronze1_division4"))]
    BronzeIDivision4,
    // Bronze II
    #[cfg_attr(feature = "database", sqlx(rename = "bronze2_division1"))]
    BronzeIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "bronze2_division2"))]
    BronzeIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "bronze2_division3"))]
    BronzeIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "bronze2_division4"))]
    BronzeIIDivision4,
    // Bronze III
    #[cfg_attr(feature = "database", sqlx(rename = "bronze3_division1"))]
    BronzeIIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "bronze3_division2"))]
    BronzeIIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "bronze3_division3"))]
    BronzeIIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "bronze3_division4"))]
    BronzeIIIDivision4,
    // Silver I
    #[cfg_attr(feature = "database", sqlx(rename = "silver1_division1"))]
    SilverIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "silver1_division2"))]
    SilverIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "silver1_division3"))]
    SilverIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "silver1_division4"))]
    SilverIDivision4,
    // Silver II
    #[cfg_attr(feature = "database", sqlx(rename = "silver2_division1"))]
    SilverIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "silver2_division2"))]
    SilverIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "silver2_division3"))]
    SilverIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "silver2_division4"))]
    SilverIIDivision4,
    // Silver III
    #[cfg_attr(feature = "database", sqlx(rename = "silver3_division1"))]
    SilverIIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "silver3_division2"))]
    SilverIIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "silver3_division3"))]
    SilverIIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "silver3_division4"))]
    SilverIIIDivision4,
    // Gold I
    #[cfg_attr(feature = "database", sqlx(rename = "gold1_division1"))]
    GoldIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "gold1_division2"))]
    GoldIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "gold1_division3"))]
    GoldIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "gold1_division4"))]
    GoldIDivision4,
    // Gold II
    #[cfg_attr(feature = "database", sqlx(rename = "gold2_division1"))]
    GoldIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "gold2_division2"))]
    GoldIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "gold2_division3"))]
    GoldIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "gold2_division4"))]
    GoldIIDivision4,
    // Gold III
    #[cfg_attr(feature = "database", sqlx(rename = "gold3_division1"))]
    GoldIIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "gold3_division2"))]
    GoldIIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "gold3_division3"))]
    GoldIIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "gold3_division4"))]
    GoldIIIDivision4,
    // Platinum I
    #[cfg_attr(feature = "database", sqlx(rename = "platinum1_division1"))]
    PlatinumIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "platinum1_division2"))]
    PlatinumIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "platinum1_division3"))]
    PlatinumIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "platinum1_division4"))]
    PlatinumIDivision4,
    // Platinum II
    #[cfg_attr(feature = "database", sqlx(rename = "platinum2_division1"))]
    PlatinumIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "platinum2_division2"))]
    PlatinumIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "platinum2_division3"))]
    PlatinumIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "platinum2_division4"))]
    PlatinumIIDivision4,
    // Platinum III
    #[cfg_attr(feature = "database", sqlx(rename = "platinum3_division1"))]
    PlatinumIIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "platinum3_division2"))]
    PlatinumIIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "platinum3_division3"))]
    PlatinumIIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "platinum3_division4"))]
    PlatinumIIIDivision4,
    // Diamond I
    #[cfg_attr(feature = "database", sqlx(rename = "diamond1_division1"))]
    DiamondIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "diamond1_division2"))]
    DiamondIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "diamond1_division3"))]
    DiamondIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "diamond1_division4"))]
    DiamondIDivision4,
    // Diamond II
    #[cfg_attr(feature = "database", sqlx(rename = "diamond2_division1"))]
    DiamondIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "diamond2_division2"))]
    DiamondIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "diamond2_division3"))]
    DiamondIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "diamond2_division4"))]
    DiamondIIDivision4,
    // Diamond III
    #[cfg_attr(feature = "database", sqlx(rename = "diamond3_division1"))]
    DiamondIIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "diamond3_division2"))]
    DiamondIIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "diamond3_division3"))]
    DiamondIIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "diamond3_division4"))]
    DiamondIIIDivision4,
    // Champion I
    #[cfg_attr(feature = "database", sqlx(rename = "champion1_division1"))]
    ChampionIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "champion1_division2"))]
    ChampionIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "champion1_division3"))]
    ChampionIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "champion1_division4"))]
    ChampionIDivision4,
    // Champion II
    #[cfg_attr(feature = "database", sqlx(rename = "champion2_division1"))]
    ChampionIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "champion2_division2"))]
    ChampionIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "champion2_division3"))]
    ChampionIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "champion2_division4"))]
    ChampionIIDivision4,
    // Champion III
    #[cfg_attr(feature = "database", sqlx(rename = "champion3_division1"))]
    ChampionIIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "champion3_division2"))]
    ChampionIIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "champion3_division3"))]
    ChampionIIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "champion3_division4"))]
    ChampionIIIDivision4,
    // Grand Champion I
    #[cfg_attr(feature = "database", sqlx(rename = "grand_champion1_division1"))]
    GrandChampionIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "grand_champion1_division2"))]
    GrandChampionIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "grand_champion1_division3"))]
    GrandChampionIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "grand_champion1_division4"))]
    GrandChampionIDivision4,
    // Grand Champion II
    #[cfg_attr(feature = "database", sqlx(rename = "grand_champion2_division1"))]
    GrandChampionIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "grand_champion2_division2"))]
    GrandChampionIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "grand_champion2_division3"))]
    GrandChampionIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "grand_champion2_division4"))]
    GrandChampionIIDivision4,
    // Grand Champion III
    #[cfg_attr(feature = "database", sqlx(rename = "grand_champion3_division1"))]
    GrandChampionIIIDivision1,
    #[cfg_attr(feature = "database", sqlx(rename = "grand_champion3_division2"))]
    GrandChampionIIIDivision2,
    #[cfg_attr(feature = "database", sqlx(rename = "grand_champion3_division3"))]
    GrandChampionIIIDivision3,
    #[cfg_attr(feature = "database", sqlx(rename = "grand_champion3_division4"))]
    GrandChampionIIIDivision4,
    // Supersonic Legend
    #[cfg_attr(feature = "database", sqlx(rename = "supersonic_legend"))]
    SupersonicLegend,
}

impl RankDivision {
    /// Converts tier and division to `RankDivision`.
    ///
    /// # Arguments
    ///
    /// * `tier` - Tier number (1 = Bronze1, 21 = GC3, 22 = SSL)
    /// * `division` - Division number (1-4), ignored for SSL (tier 22)
    ///
    /// # Returns
    ///
    /// Returns `Some(RankDivision)` if the tier and division are valid, `None` otherwise.
    #[must_use]
    pub fn from_tier_and_division(tier: i32, division: Option<i32>) -> Option<Self> {
        let division = division.unwrap_or(1).clamp(1, 4);

        match (tier, division) {
            (22, _) => Some(Self::SupersonicLegend),
            // Grand Champion III (tier 21)
            (21, 1) => Some(Self::GrandChampionIIIDivision1),
            (21, 2) => Some(Self::GrandChampionIIIDivision2),
            (21, 3) => Some(Self::GrandChampionIIIDivision3),
            (21, 4) => Some(Self::GrandChampionIIIDivision4),
            // Grand Champion II (tier 20)
            (20, 1) => Some(Self::GrandChampionIIDivision1),
            (20, 2) => Some(Self::GrandChampionIIDivision2),
            (20, 3) => Some(Self::GrandChampionIIDivision3),
            (20, 4) => Some(Self::GrandChampionIIDivision4),
            // Grand Champion I (tier 19)
            (19, 1) => Some(Self::GrandChampionIDivision1),
            (19, 2) => Some(Self::GrandChampionIDivision2),
            (19, 3) => Some(Self::GrandChampionIDivision3),
            (19, 4) => Some(Self::GrandChampionIDivision4),
            // Champion III (tier 18)
            (18, 1) => Some(Self::ChampionIIIDivision1),
            (18, 2) => Some(Self::ChampionIIIDivision2),
            (18, 3) => Some(Self::ChampionIIIDivision3),
            (18, 4) => Some(Self::ChampionIIIDivision4),
            // Champion II (tier 17)
            (17, 1) => Some(Self::ChampionIIDivision1),
            (17, 2) => Some(Self::ChampionIIDivision2),
            (17, 3) => Some(Self::ChampionIIDivision3),
            (17, 4) => Some(Self::ChampionIIDivision4),
            // Champion I (tier 16)
            (16, 1) => Some(Self::ChampionIDivision1),
            (16, 2) => Some(Self::ChampionIDivision2),
            (16, 3) => Some(Self::ChampionIDivision3),
            (16, 4) => Some(Self::ChampionIDivision4),
            // Diamond III (tier 15)
            (15, 1) => Some(Self::DiamondIIIDivision1),
            (15, 2) => Some(Self::DiamondIIIDivision2),
            (15, 3) => Some(Self::DiamondIIIDivision3),
            (15, 4) => Some(Self::DiamondIIIDivision4),
            // Diamond II (tier 14)
            (14, 1) => Some(Self::DiamondIIDivision1),
            (14, 2) => Some(Self::DiamondIIDivision2),
            (14, 3) => Some(Self::DiamondIIDivision3),
            (14, 4) => Some(Self::DiamondIIDivision4),
            // Diamond I (tier 13)
            (13, 1) => Some(Self::DiamondIDivision1),
            (13, 2) => Some(Self::DiamondIDivision2),
            (13, 3) => Some(Self::DiamondIDivision3),
            (13, 4) => Some(Self::DiamondIDivision4),
            // Platinum III (tier 12)
            (12, 1) => Some(Self::PlatinumIIIDivision1),
            (12, 2) => Some(Self::PlatinumIIIDivision2),
            (12, 3) => Some(Self::PlatinumIIIDivision3),
            (12, 4) => Some(Self::PlatinumIIIDivision4),
            // Platinum II (tier 11)
            (11, 1) => Some(Self::PlatinumIIDivision1),
            (11, 2) => Some(Self::PlatinumIIDivision2),
            (11, 3) => Some(Self::PlatinumIIDivision3),
            (11, 4) => Some(Self::PlatinumIIDivision4),
            // Platinum I (tier 10)
            (10, 1) => Some(Self::PlatinumIDivision1),
            (10, 2) => Some(Self::PlatinumIDivision2),
            (10, 3) => Some(Self::PlatinumIDivision3),
            (10, 4) => Some(Self::PlatinumIDivision4),
            // Gold III (tier 9)
            (9, 1) => Some(Self::GoldIIIDivision1),
            (9, 2) => Some(Self::GoldIIIDivision2),
            (9, 3) => Some(Self::GoldIIIDivision3),
            (9, 4) => Some(Self::GoldIIIDivision4),
            // Gold II (tier 8)
            (8, 1) => Some(Self::GoldIIDivision1),
            (8, 2) => Some(Self::GoldIIDivision2),
            (8, 3) => Some(Self::GoldIIDivision3),
            (8, 4) => Some(Self::GoldIIDivision4),
            // Gold I (tier 7)
            (7, 1) => Some(Self::GoldIDivision1),
            (7, 2) => Some(Self::GoldIDivision2),
            (7, 3) => Some(Self::GoldIDivision3),
            (7, 4) => Some(Self::GoldIDivision4),
            // Silver III (tier 6)
            (6, 1) => Some(Self::SilverIIIDivision1),
            (6, 2) => Some(Self::SilverIIIDivision2),
            (6, 3) => Some(Self::SilverIIIDivision3),
            (6, 4) => Some(Self::SilverIIIDivision4),
            // Silver II (tier 5)
            (5, 1) => Some(Self::SilverIIDivision1),
            (5, 2) => Some(Self::SilverIIDivision2),
            (5, 3) => Some(Self::SilverIIDivision3),
            (5, 4) => Some(Self::SilverIIDivision4),
            // Silver I (tier 4)
            (4, 1) => Some(Self::SilverIDivision1),
            (4, 2) => Some(Self::SilverIDivision2),
            (4, 3) => Some(Self::SilverIDivision3),
            (4, 4) => Some(Self::SilverIDivision4),
            // Bronze III (tier 3)
            (3, 1) => Some(Self::BronzeIIIDivision1),
            (3, 2) => Some(Self::BronzeIIIDivision2),
            (3, 3) => Some(Self::BronzeIIIDivision3),
            (3, 4) => Some(Self::BronzeIIIDivision4),
            // Bronze II (tier 2)
            (2, 1) => Some(Self::BronzeIIDivision1),
            (2, 2) => Some(Self::BronzeIIDivision2),
            (2, 3) => Some(Self::BronzeIIDivision3),
            (2, 4) => Some(Self::BronzeIIDivision4),
            // Bronze I (tier 1)
            (1, 1) => Some(Self::BronzeIDivision1),
            (1, 2) => Some(Self::BronzeIDivision2),
            (1, 3) => Some(Self::BronzeIDivision3),
            (1, 4) => Some(Self::BronzeIDivision4),
            _ => None,
        }
    }

    /// Returns the numeric index of this rank division (0-based).
    ///
    /// `BronzeIDivision1` = 0, `SupersonicLegend` = 84.
    #[must_use]
    pub const fn as_index(self) -> usize {
        self as usize
    }

    /// Returns the minimum MMR value for this rank.
    #[must_use]
    #[expect(
        clippy::match_same_arms,
        reason = "Intentional: values from official rank data"
    )]
    pub const fn mmr_min(self) -> i32 {
        match self {
            // Bronze I
            Self::BronzeIDivision1 => 0,
            Self::BronzeIDivision2 => 121,
            Self::BronzeIDivision3 => 138,
            Self::BronzeIDivision4 => 157,
            // Bronze II
            Self::BronzeIIDivision1 => 175,
            Self::BronzeIIDivision2 => 180,
            Self::BronzeIIDivision3 => 203,
            Self::BronzeIIDivision4 => 217,
            // Bronze III
            Self::BronzeIIIDivision1 => 234,
            Self::BronzeIIIDivision2 => 240,
            Self::BronzeIIIDivision3 => 261,
            Self::BronzeIIIDivision4 => 277,
            // Silver I
            Self::SilverIDivision1 => 280,
            Self::SilverIDivision2 => 300,
            Self::SilverIDivision3 => 318,
            Self::SilverIDivision4 => 337,
            // Silver II
            Self::SilverIIDivision1 => 340,
            Self::SilverIIDivision2 => 360,
            Self::SilverIIDivision3 => 378,
            Self::SilverIIDivision4 => 397,
            // Silver III
            Self::SilverIIIDivision1 => 400,
            Self::SilverIIIDivision2 => 419,
            Self::SilverIIIDivision3 => 438,
            Self::SilverIIIDivision4 => 457,
            // Gold I
            Self::GoldIDivision1 => 460,
            Self::GoldIDivision2 => 479,
            Self::GoldIDivision3 => 498,
            Self::GoldIDivision4 => 517,
            // Gold II
            Self::GoldIIDivision1 => 520,
            Self::GoldIIDivision2 => 539,
            Self::GoldIIDivision3 => 558,
            Self::GoldIIDivision4 => 577,
            // Gold III
            Self::GoldIIIDivision1 => 580,
            Self::GoldIIIDivision2 => 580,
            Self::GoldIIIDivision3 => 580,
            Self::GoldIIIDivision4 => 637,
            // Platinum I
            Self::PlatinumIDivision1 => 640,
            Self::PlatinumIDivision2 => 659,
            Self::PlatinumIDivision3 => 640,
            Self::PlatinumIDivision4 => 640,
            // Platinum II
            Self::PlatinumIIDivision1 => 715,
            Self::PlatinumIIDivision2 => 719,
            Self::PlatinumIIDivision3 => 738,
            Self::PlatinumIIDivision4 => 757,
            // Platinum III
            Self::PlatinumIIIDivision1 => 760,
            Self::PlatinumIIIDivision2 => 760,
            Self::PlatinumIIIDivision3 => 760,
            Self::PlatinumIIIDivision4 => 760,
            // Diamond I
            Self::DiamondIDivision1 => 835,
            Self::DiamondIDivision2 => 820,
            Self::DiamondIDivision3 => 820,
            Self::DiamondIDivision4 => 892,
            // Diamond II
            Self::DiamondIIDivision1 => 915,
            Self::DiamondIIDivision2 => 924,
            Self::DiamondIIDivision3 => 900,
            Self::DiamondIIDivision4 => 972,
            // Diamond III
            Self::DiamondIIIDivision1 => 980,
            Self::DiamondIIIDivision2 => 980,
            Self::DiamondIIIDivision3 => 980,
            Self::DiamondIIIDivision4 => 980,
            // Champion I
            Self::ChampionIDivision1 => 1075,
            Self::ChampionIDivision2 => 1095,
            Self::ChampionIDivision3 => 1128,
            Self::ChampionIDivision4 => 1162,
            // Champion II
            Self::ChampionIIDivision1 => 1195,
            Self::ChampionIIDivision2 => 1215,
            Self::ChampionIIDivision3 => 1248,
            Self::ChampionIIDivision4 => 1282,
            // Champion III
            Self::ChampionIIIDivision1 => 1315,
            Self::ChampionIIIDivision2 => 1335,
            Self::ChampionIIIDivision3 => 1369,
            Self::ChampionIIIDivision4 => 1402,
            // Grand Champion I
            Self::GrandChampionIDivision1 => 1436,
            Self::GrandChampionIDivision2 => 1460,
            Self::GrandChampionIDivision3 => 1498,
            Self::GrandChampionIDivision4 => 1537,
            // Grand Champion II
            Self::GrandChampionIIDivision1 => 1575,
            Self::GrandChampionIIDivision2 => 1600,
            Self::GrandChampionIIDivision3 => 1645,
            Self::GrandChampionIIDivision4 => 1677,
            // Grand Champion III
            Self::GrandChampionIIIDivision1 => 1706,
            Self::GrandChampionIIIDivision2 => 1746,
            Self::GrandChampionIIIDivision3 => 1794,
            Self::GrandChampionIIIDivision4 => 1810,
            // Supersonic Legend
            Self::SupersonicLegend => 1883,
        }
    }

    /// Returns the maximum MMR value for this rank.
    #[must_use]
    pub const fn mmr_max(self) -> i32 {
        match self {
            // Bronze I
            Self::BronzeIDivision1 => 117,
            Self::BronzeIDivision2 => 130,
            Self::BronzeIDivision3 => 156,
            Self::BronzeIDivision4 => 174,
            // Bronze II
            Self::BronzeIIDivision1 => 178,
            Self::BronzeIIDivision2 => 194,
            Self::BronzeIIDivision3 => 215,
            Self::BronzeIIDivision4 => 234,
            // Bronze III
            Self::BronzeIIIDivision1 => 238,
            Self::BronzeIIIDivision2 => 257,
            Self::BronzeIIIDivision3 => 275,
            Self::BronzeIIIDivision4 => 294,
            // Silver I
            Self::SilverIDivision1 => 298,
            Self::SilverIDivision2 => 317,
            Self::SilverIDivision3 => 336,
            Self::SilverIDivision4 => 354,
            // Silver II
            Self::SilverIIDivision1 => 358,
            Self::SilverIIDivision2 => 377,
            Self::SilverIIDivision3 => 396,
            Self::SilverIIDivision4 => 415,
            // Silver III
            Self::SilverIIIDivision1 => 418,
            Self::SilverIIIDivision2 => 437,
            Self::SilverIIIDivision3 => 456,
            Self::SilverIIIDivision4 => 474,
            // Gold I
            Self::GoldIDivision1 => 478,
            Self::GoldIDivision2 => 497,
            Self::GoldIDivision3 => 516,
            Self::GoldIDivision4 => 534,
            // Gold II
            Self::GoldIIDivision1 => 538,
            Self::GoldIIDivision2 => 557,
            Self::GoldIIDivision3 => 576,
            Self::GoldIIDivision4 => 590,
            // Gold III
            Self::GoldIIIDivision1 => 598,
            Self::GoldIIIDivision2 => 617,
            Self::GoldIIIDivision3 => 636,
            Self::GoldIIIDivision4 => 654,
            // Platinum I
            Self::PlatinumIDivision1 => 658,
            Self::PlatinumIDivision2 => 677,
            Self::PlatinumIDivision3 => 696,
            Self::PlatinumIDivision4 => 714,
            // Platinum II
            Self::PlatinumIIDivision1 => 718,
            Self::PlatinumIIDivision2 => 737,
            Self::PlatinumIIDivision3 => 756,
            Self::PlatinumIIDivision4 => 774,
            // Platinum III
            Self::PlatinumIIIDivision1 => 778,
            Self::PlatinumIIIDivision2 => 797,
            Self::PlatinumIIIDivision3 => 816,
            Self::PlatinumIIIDivision4 => 834,
            // Diamond I
            Self::DiamondIDivision1 => 843,
            Self::DiamondIDivision2 => 867,
            Self::DiamondIDivision3 => 891,
            Self::DiamondIDivision4 => 914,
            // Diamond II
            Self::DiamondIIDivision1 => 923,
            Self::DiamondIIDivision2 => 947,
            Self::DiamondIIDivision3 => 971,
            Self::DiamondIIDivision4 => 994,
            // Diamond III
            Self::DiamondIIIDivision1 => 1003,
            Self::DiamondIIIDivision2 => 1027,
            Self::DiamondIIIDivision3 => 1051,
            Self::DiamondIIIDivision4 => 1074,
            // Champion I
            Self::ChampionIDivision1 => 1093,
            Self::ChampionIDivision2 => 1127,
            Self::ChampionIDivision3 => 1160,
            Self::ChampionIDivision4 => 1180,
            // Champion II
            Self::ChampionIIDivision1 => 1213,
            Self::ChampionIIDivision2 => 1247,
            Self::ChampionIIDivision3 => 1271,
            Self::ChampionIIDivision4 => 1300,
            // Champion III
            Self::ChampionIIIDivision1 => 1333,
            Self::ChampionIIIDivision2 => 1366,
            Self::ChampionIIIDivision3 => 1401,
            Self::ChampionIIIDivision4 => 1419,
            // Grand Champion I
            Self::GrandChampionIDivision1 => 1458,
            Self::GrandChampionIDivision2 => 1497,
            Self::GrandChampionIDivision3 => 1535,
            Self::GrandChampionIDivision4 => 1561,
            // Grand Champion II
            Self::GrandChampionIIDivision1 => 1598,
            Self::GrandChampionIIDivision2 => 1636,
            Self::GrandChampionIIDivision3 => 1660,
            Self::GrandChampionIIDivision4 => 1708,
            // Grand Champion III
            Self::GrandChampionIIIDivision1 => 1739,
            Self::GrandChampionIIIDivision2 => 1780,
            Self::GrandChampionIIIDivision3 => 1809,
            Self::GrandChampionIIIDivision4 => 1882,
            // Supersonic Legend (unbounded, using 2000 as a practical max)
            Self::SupersonicLegend => 2000,
        }
    }

    /// Returns the middle MMR value for this rank.
    ///
    /// Supersonic Legend uses its listed range like every other division (1883–2000, middle
    /// 1941), although the rank is unbounded above. A higher label (it used to be 2200) sat
    /// out of reach of the tree model, which predicted SSLs hundreds of MMR lower, so their
    /// errors dominated the duels metrics without making SSL cards any more accurate.
    #[must_use]
    pub const fn mmr_middle(self) -> i32 {
        i32::midpoint(self.mmr_min(), self.mmr_max())
    }

    /// Returns an iterator over all possible ranks in order from lowest to highest.
    pub fn all() -> impl Iterator<Item = Self> {
        Self::iter()
    }
}

impl fmt::Display for RankDivision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            // Bronze I
            Self::BronzeIDivision1 => "Bronze I Division 1",
            Self::BronzeIDivision2 => "Bronze I Division 2",
            Self::BronzeIDivision3 => "Bronze I Division 3",
            Self::BronzeIDivision4 => "Bronze I Division 4",
            // Bronze II
            Self::BronzeIIDivision1 => "Bronze II Division 1",
            Self::BronzeIIDivision2 => "Bronze II Division 2",
            Self::BronzeIIDivision3 => "Bronze II Division 3",
            Self::BronzeIIDivision4 => "Bronze II Division 4",
            // Bronze III
            Self::BronzeIIIDivision1 => "Bronze III Division 1",
            Self::BronzeIIIDivision2 => "Bronze III Division 2",
            Self::BronzeIIIDivision3 => "Bronze III Division 3",
            Self::BronzeIIIDivision4 => "Bronze III Division 4",
            // Silver I
            Self::SilverIDivision1 => "Silver I Division 1",
            Self::SilverIDivision2 => "Silver I Division 2",
            Self::SilverIDivision3 => "Silver I Division 3",
            Self::SilverIDivision4 => "Silver I Division 4",
            // Silver II
            Self::SilverIIDivision1 => "Silver II Division 1",
            Self::SilverIIDivision2 => "Silver II Division 2",
            Self::SilverIIDivision3 => "Silver II Division 3",
            Self::SilverIIDivision4 => "Silver II Division 4",
            // Silver III
            Self::SilverIIIDivision1 => "Silver III Division 1",
            Self::SilverIIIDivision2 => "Silver III Division 2",
            Self::SilverIIIDivision3 => "Silver III Division 3",
            Self::SilverIIIDivision4 => "Silver III Division 4",
            // Gold I
            Self::GoldIDivision1 => "Gold I Division 1",
            Self::GoldIDivision2 => "Gold I Division 2",
            Self::GoldIDivision3 => "Gold I Division 3",
            Self::GoldIDivision4 => "Gold I Division 4",
            // Gold II
            Self::GoldIIDivision1 => "Gold II Division 1",
            Self::GoldIIDivision2 => "Gold II Division 2",
            Self::GoldIIDivision3 => "Gold II Division 3",
            Self::GoldIIDivision4 => "Gold II Division 4",
            // Gold III
            Self::GoldIIIDivision1 => "Gold III Division 1",
            Self::GoldIIIDivision2 => "Gold III Division 2",
            Self::GoldIIIDivision3 => "Gold III Division 3",
            Self::GoldIIIDivision4 => "Gold III Division 4",
            // Platinum I
            Self::PlatinumIDivision1 => "Platinum I Division 1",
            Self::PlatinumIDivision2 => "Platinum I Division 2",
            Self::PlatinumIDivision3 => "Platinum I Division 3",
            Self::PlatinumIDivision4 => "Platinum I Division 4",
            // Platinum II
            Self::PlatinumIIDivision1 => "Platinum II Division 1",
            Self::PlatinumIIDivision2 => "Platinum II Division 2",
            Self::PlatinumIIDivision3 => "Platinum II Division 3",
            Self::PlatinumIIDivision4 => "Platinum II Division 4",
            // Platinum III
            Self::PlatinumIIIDivision1 => "Platinum III Division 1",
            Self::PlatinumIIIDivision2 => "Platinum III Division 2",
            Self::PlatinumIIIDivision3 => "Platinum III Division 3",
            Self::PlatinumIIIDivision4 => "Platinum III Division 4",
            // Diamond I
            Self::DiamondIDivision1 => "Diamond I Division 1",
            Self::DiamondIDivision2 => "Diamond I Division 2",
            Self::DiamondIDivision3 => "Diamond I Division 3",
            Self::DiamondIDivision4 => "Diamond I Division 4",
            // Diamond II
            Self::DiamondIIDivision1 => "Diamond II Division 1",
            Self::DiamondIIDivision2 => "Diamond II Division 2",
            Self::DiamondIIDivision3 => "Diamond II Division 3",
            Self::DiamondIIDivision4 => "Diamond II Division 4",
            // Diamond III
            Self::DiamondIIIDivision1 => "Diamond III Division 1",
            Self::DiamondIIIDivision2 => "Diamond III Division 2",
            Self::DiamondIIIDivision3 => "Diamond III Division 3",
            Self::DiamondIIIDivision4 => "Diamond III Division 4",
            // Champion I
            Self::ChampionIDivision1 => "Champion I Division 1",
            Self::ChampionIDivision2 => "Champion I Division 2",
            Self::ChampionIDivision3 => "Champion I Division 3",
            Self::ChampionIDivision4 => "Champion I Division 4",
            // Champion II
            Self::ChampionIIDivision1 => "Champion II Division 1",
            Self::ChampionIIDivision2 => "Champion II Division 2",
            Self::ChampionIIDivision3 => "Champion II Division 3",
            Self::ChampionIIDivision4 => "Champion II Division 4",
            // Champion III
            Self::ChampionIIIDivision1 => "Champion III Division 1",
            Self::ChampionIIIDivision2 => "Champion III Division 2",
            Self::ChampionIIIDivision3 => "Champion III Division 3",
            Self::ChampionIIIDivision4 => "Champion III Division 4",
            // Grand Champion I
            Self::GrandChampionIDivision1 => "Grand Champion I Division 1",
            Self::GrandChampionIDivision2 => "Grand Champion I Division 2",
            Self::GrandChampionIDivision3 => "Grand Champion I Division 3",
            Self::GrandChampionIDivision4 => "Grand Champion I Division 4",
            // Grand Champion II
            Self::GrandChampionIIDivision1 => "Grand Champion II Division 1",
            Self::GrandChampionIIDivision2 => "Grand Champion II Division 2",
            Self::GrandChampionIIDivision3 => "Grand Champion II Division 3",
            Self::GrandChampionIIDivision4 => "Grand Champion II Division 4",
            // Grand Champion III
            Self::GrandChampionIIIDivision1 => "Grand Champion III Division 1",
            Self::GrandChampionIIIDivision2 => "Grand Champion III Division 2",
            Self::GrandChampionIIIDivision3 => "Grand Champion III Division 3",
            Self::GrandChampionIIIDivision4 => "Grand Champion III Division 4",
            // Supersonic Legend
            Self::SupersonicLegend => "Supersonic Legend",
        };
        write!(f, "{name}")
    }
}

/// Error type for rank parsing failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseRankError {
    /// The invalid input string.
    pub input: String,
}

impl fmt::Display for ParseRankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid rank: '{}'", self.input)
    }
}

impl core::error::Error for ParseRankError {}

impl FromStr for RankDivision {
    type Err = ParseRankError;

    /// Parses a rank from a string.
    ///
    /// Accepts formats like:
    /// - `SupersonicLegend` or `Supersonic Legend`
    /// - `GrandChampionIIIDivision1` or `Grand Champion III Division 1`
    /// - `gc3d1` (short form)
    /// - `bronze-1` with division number (external API format)
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Normalize: lowercase and remove spaces/dashes/underscores
        let normalized = s.to_lowercase().replace([' ', '-', '_'], "");

        // Try short forms first (e.g., "gc3d1", "c2d3", "ssld1")
        if let Some(rank) = parse_short_form(&normalized) {
            return Ok(rank);
        }

        // Try full forms
        let rank = match normalized.as_str() {
            // Supersonic Legend
            "supersoniclegend" | "ssl" => Self::SupersonicLegend,

            // Grand Champion III
            "grandchampioniiidivision1" | "grandchampion3division1" => {
                Self::GrandChampionIIIDivision1
            }
            "grandchampioniiidivision2" | "grandchampion3division2" => {
                Self::GrandChampionIIIDivision2
            }
            "grandchampioniiidivision3" | "grandchampion3division3" => {
                Self::GrandChampionIIIDivision3
            }
            "grandchampioniiidivision4" | "grandchampion3division4" => {
                Self::GrandChampionIIIDivision4
            }
            // Grand Champion II
            "grandchampioniidivision1" | "grandchampion2division1" => {
                Self::GrandChampionIIDivision1
            }
            "grandchampioniidivision2" | "grandchampion2division2" => {
                Self::GrandChampionIIDivision2
            }
            "grandchampioniidivision3" | "grandchampion2division3" => {
                Self::GrandChampionIIDivision3
            }
            "grandchampioniidivision4" | "grandchampion2division4" => {
                Self::GrandChampionIIDivision4
            }
            // Grand Champion I
            "grandchampionidivision1" | "grandchampion1division1" => Self::GrandChampionIDivision1,
            "grandchampionidivision2" | "grandchampion1division2" => Self::GrandChampionIDivision2,
            "grandchampionidivision3" | "grandchampion1division3" => Self::GrandChampionIDivision3,
            "grandchampionidivision4" | "grandchampion1division4" => Self::GrandChampionIDivision4,
            // Champion III
            "championiiidivision1" | "champion3division1" => Self::ChampionIIIDivision1,
            "championiiidivision2" | "champion3division2" => Self::ChampionIIIDivision2,
            "championiiidivision3" | "champion3division3" => Self::ChampionIIIDivision3,
            "championiiidivision4" | "champion3division4" => Self::ChampionIIIDivision4,
            // Champion II
            "championiidivision1" | "champion2division1" => Self::ChampionIIDivision1,
            "championiidivision2" | "champion2division2" => Self::ChampionIIDivision2,
            "championiidivision3" | "champion2division3" => Self::ChampionIIDivision3,
            "championiidivision4" | "champion2division4" => Self::ChampionIIDivision4,
            // Champion I
            "championidivision1" | "champion1division1" => Self::ChampionIDivision1,
            "championidivision2" | "champion1division2" => Self::ChampionIDivision2,
            "championidivision3" | "champion1division3" => Self::ChampionIDivision3,
            "championidivision4" | "champion1division4" => Self::ChampionIDivision4,
            // Diamond III
            "diamondiiidivision1" | "diamond3division1" => Self::DiamondIIIDivision1,
            "diamondiiidivision2" | "diamond3division2" => Self::DiamondIIIDivision2,
            "diamondiiidivision3" | "diamond3division3" => Self::DiamondIIIDivision3,
            "diamondiiidivision4" | "diamond3division4" => Self::DiamondIIIDivision4,
            // Diamond II
            "diamondiidivision1" | "diamond2division1" => Self::DiamondIIDivision1,
            "diamondiidivision2" | "diamond2division2" => Self::DiamondIIDivision2,
            "diamondiidivision3" | "diamond2division3" => Self::DiamondIIDivision3,
            "diamondiidivision4" | "diamond2division4" => Self::DiamondIIDivision4,
            // Diamond I
            "diamondidivision1" | "diamond1division1" => Self::DiamondIDivision1,
            "diamondidivision2" | "diamond1division2" => Self::DiamondIDivision2,
            "diamondidivision3" | "diamond1division3" => Self::DiamondIDivision3,
            "diamondidivision4" | "diamond1division4" => Self::DiamondIDivision4,
            // Platinum III
            "platinumiiidivision1" | "platinum3division1" => Self::PlatinumIIIDivision1,
            "platinumiiidivision2" | "platinum3division2" => Self::PlatinumIIIDivision2,
            "platinumiiidivision3" | "platinum3division3" => Self::PlatinumIIIDivision3,
            "platinumiiidivision4" | "platinum3division4" => Self::PlatinumIIIDivision4,
            // Platinum II
            "platinumiidivision1" | "platinum2division1" => Self::PlatinumIIDivision1,
            "platinumiidivision2" | "platinum2division2" => Self::PlatinumIIDivision2,
            "platinumiidivision3" | "platinum2division3" => Self::PlatinumIIDivision3,
            "platinumiidivision4" | "platinum2division4" => Self::PlatinumIIDivision4,
            // Platinum I
            "platinumidivision1" | "platinum1division1" => Self::PlatinumIDivision1,
            "platinumidivision2" | "platinum1division2" => Self::PlatinumIDivision2,
            "platinumidivision3" | "platinum1division3" => Self::PlatinumIDivision3,
            "platinumidivision4" | "platinum1division4" => Self::PlatinumIDivision4,
            // Gold III
            "goldiiidivision1" | "gold3division1" => Self::GoldIIIDivision1,
            "goldiiidivision2" | "gold3division2" => Self::GoldIIIDivision2,
            "goldiiidivision3" | "gold3division3" => Self::GoldIIIDivision3,
            "goldiiidivision4" | "gold3division4" => Self::GoldIIIDivision4,
            // Gold II
            "goldiidivision1" | "gold2division1" => Self::GoldIIDivision1,
            "goldiidivision2" | "gold2division2" => Self::GoldIIDivision2,
            "goldiidivision3" | "gold2division3" => Self::GoldIIDivision3,
            "goldiidivision4" | "gold2division4" => Self::GoldIIDivision4,
            // Gold I
            "goldidivision1" | "gold1division1" => Self::GoldIDivision1,
            "goldidivision2" | "gold1division2" => Self::GoldIDivision2,
            "goldidivision3" | "gold1division3" => Self::GoldIDivision3,
            "goldidivision4" | "gold1division4" => Self::GoldIDivision4,
            // Silver III
            "silveriiidivision1" | "silver3division1" => Self::SilverIIIDivision1,
            "silveriiidivision2" | "silver3division2" => Self::SilverIIIDivision2,
            "silveriiidivision3" | "silver3division3" => Self::SilverIIIDivision3,
            "silveriiidivision4" | "silver3division4" => Self::SilverIIIDivision4,
            // Silver II
            "silveriidivision1" | "silver2division1" => Self::SilverIIDivision1,
            "silveriidivision2" | "silver2division2" => Self::SilverIIDivision2,
            "silveriidivision3" | "silver2division3" => Self::SilverIIDivision3,
            "silveriidivision4" | "silver2division4" => Self::SilverIIDivision4,
            // Silver I
            "silveridivision1" | "silver1division1" => Self::SilverIDivision1,
            "silveridivision2" | "silver1division2" => Self::SilverIDivision2,
            "silveridivision3" | "silver1division3" => Self::SilverIDivision3,
            "silveridivision4" | "silver1division4" => Self::SilverIDivision4,
            // Bronze III
            "bronzeiiidivision1" | "bronze3division1" => Self::BronzeIIIDivision1,
            "bronzeiiidivision2" | "bronze3division2" => Self::BronzeIIIDivision2,
            "bronzeiiidivision3" | "bronze3division3" => Self::BronzeIIIDivision3,
            "bronzeiiidivision4" | "bronze3division4" => Self::BronzeIIIDivision4,
            // Bronze II
            "bronzeiidivision1" | "bronze2division1" => Self::BronzeIIDivision1,
            "bronzeiidivision2" | "bronze2division2" => Self::BronzeIIDivision2,
            "bronzeiidivision3" | "bronze2division3" => Self::BronzeIIDivision3,
            "bronzeiidivision4" | "bronze2division4" => Self::BronzeIIDivision4,
            // Bronze I
            "bronzeidivision1" | "bronze1division1" => Self::BronzeIDivision1,
            "bronzeidivision2" | "bronze1division2" => Self::BronzeIDivision2,
            "bronzeidivision3" | "bronze1division3" => Self::BronzeIDivision3,
            "bronzeidivision4" | "bronze1division4" => Self::BronzeIDivision4,

            _ => {
                return Err(ParseRankError {
                    input: s.to_string(),
                });
            }
        };

        Ok(rank)
    }
}

type RankDivisionStringPatternEntry = (&'static str, fn(i32, i32) -> Option<RankDivision>);

/// Parses short form ranks like `gc3d1`, `c2d3`, `d1d4`, `ssl`.
#[expect(clippy::too_many_lines)]
fn parse_short_form(s: &str) -> Option<RankDivision> {
    use RankDivision::{
        BronzeIDivision1, BronzeIDivision2, BronzeIDivision3, BronzeIDivision4, BronzeIIDivision1,
        BronzeIIDivision2, BronzeIIDivision3, BronzeIIDivision4, BronzeIIIDivision1,
        BronzeIIIDivision2, BronzeIIIDivision3, BronzeIIIDivision4, ChampionIDivision1,
        ChampionIDivision2, ChampionIDivision3, ChampionIDivision4, ChampionIIDivision1,
        ChampionIIDivision2, ChampionIIDivision3, ChampionIIDivision4, ChampionIIIDivision1,
        ChampionIIIDivision2, ChampionIIIDivision3, ChampionIIIDivision4, DiamondIDivision1,
        DiamondIDivision2, DiamondIDivision3, DiamondIDivision4, DiamondIIDivision1,
        DiamondIIDivision2, DiamondIIDivision3, DiamondIIDivision4, DiamondIIIDivision1,
        DiamondIIIDivision2, DiamondIIIDivision3, DiamondIIIDivision4, GoldIDivision1,
        GoldIDivision2, GoldIDivision3, GoldIDivision4, GoldIIDivision1, GoldIIDivision2,
        GoldIIDivision3, GoldIIDivision4, GoldIIIDivision1, GoldIIIDivision2, GoldIIIDivision3,
        GoldIIIDivision4, GrandChampionIDivision1, GrandChampionIDivision2,
        GrandChampionIDivision3, GrandChampionIDivision4, GrandChampionIIDivision1,
        GrandChampionIIDivision2, GrandChampionIIDivision3, GrandChampionIIDivision4,
        GrandChampionIIIDivision1, GrandChampionIIIDivision2, GrandChampionIIIDivision3,
        GrandChampionIIIDivision4, PlatinumIDivision1, PlatinumIDivision2, PlatinumIDivision3,
        PlatinumIDivision4, PlatinumIIDivision1, PlatinumIIDivision2, PlatinumIIDivision3,
        PlatinumIIDivision4, PlatinumIIIDivision1, PlatinumIIIDivision2, PlatinumIIIDivision3,
        PlatinumIIIDivision4, SilverIDivision1, SilverIDivision2, SilverIDivision3,
        SilverIDivision4, SilverIIDivision1, SilverIIDivision2, SilverIIDivision3,
        SilverIIDivision4, SilverIIIDivision1, SilverIIIDivision2, SilverIIIDivision3,
        SilverIIIDivision4, SupersonicLegend,
    };

    // Handle "ssl" specially
    if s == "ssl" {
        return Some(SupersonicLegend);
    }

    // Pattern: <rank_prefix><tier>d<division>
    // gc = Grand Champion, c = Champion, d = Diamond, p = Platinum,
    // g = Gold, s = Silver, b = Bronze
    let patterns: &[RankDivisionStringPatternEntry] = &[
        ("gc", |tier, div| match (tier, div) {
            (3, 1) => Some(GrandChampionIIIDivision1),
            (3, 2) => Some(GrandChampionIIIDivision2),
            (3, 3) => Some(GrandChampionIIIDivision3),
            (3, 4) => Some(GrandChampionIIIDivision4),
            (2, 1) => Some(GrandChampionIIDivision1),
            (2, 2) => Some(GrandChampionIIDivision2),
            (2, 3) => Some(GrandChampionIIDivision3),
            (2, 4) => Some(GrandChampionIIDivision4),
            (1, 1) => Some(GrandChampionIDivision1),
            (1, 2) => Some(GrandChampionIDivision2),
            (1, 3) => Some(GrandChampionIDivision3),
            (1, 4) => Some(GrandChampionIDivision4),
            _ => None,
        }),
        ("c", |tier, div| match (tier, div) {
            (3, 1) => Some(ChampionIIIDivision1),
            (3, 2) => Some(ChampionIIIDivision2),
            (3, 3) => Some(ChampionIIIDivision3),
            (3, 4) => Some(ChampionIIIDivision4),
            (2, 1) => Some(ChampionIIDivision1),
            (2, 2) => Some(ChampionIIDivision2),
            (2, 3) => Some(ChampionIIDivision3),
            (2, 4) => Some(ChampionIIDivision4),
            (1, 1) => Some(ChampionIDivision1),
            (1, 2) => Some(ChampionIDivision2),
            (1, 3) => Some(ChampionIDivision3),
            (1, 4) => Some(ChampionIDivision4),
            _ => None,
        }),
        ("d", |tier, div| match (tier, div) {
            (3, 1) => Some(DiamondIIIDivision1),
            (3, 2) => Some(DiamondIIIDivision2),
            (3, 3) => Some(DiamondIIIDivision3),
            (3, 4) => Some(DiamondIIIDivision4),
            (2, 1) => Some(DiamondIIDivision1),
            (2, 2) => Some(DiamondIIDivision2),
            (2, 3) => Some(DiamondIIDivision3),
            (2, 4) => Some(DiamondIIDivision4),
            (1, 1) => Some(DiamondIDivision1),
            (1, 2) => Some(DiamondIDivision2),
            (1, 3) => Some(DiamondIDivision3),
            (1, 4) => Some(DiamondIDivision4),
            _ => None,
        }),
        ("p", |tier, div| match (tier, div) {
            (3, 1) => Some(PlatinumIIIDivision1),
            (3, 2) => Some(PlatinumIIIDivision2),
            (3, 3) => Some(PlatinumIIIDivision3),
            (3, 4) => Some(PlatinumIIIDivision4),
            (2, 1) => Some(PlatinumIIDivision1),
            (2, 2) => Some(PlatinumIIDivision2),
            (2, 3) => Some(PlatinumIIDivision3),
            (2, 4) => Some(PlatinumIIDivision4),
            (1, 1) => Some(PlatinumIDivision1),
            (1, 2) => Some(PlatinumIDivision2),
            (1, 3) => Some(PlatinumIDivision3),
            (1, 4) => Some(PlatinumIDivision4),
            _ => None,
        }),
        ("g", |tier, div| match (tier, div) {
            (3, 1) => Some(GoldIIIDivision1),
            (3, 2) => Some(GoldIIIDivision2),
            (3, 3) => Some(GoldIIIDivision3),
            (3, 4) => Some(GoldIIIDivision4),
            (2, 1) => Some(GoldIIDivision1),
            (2, 2) => Some(GoldIIDivision2),
            (2, 3) => Some(GoldIIDivision3),
            (2, 4) => Some(GoldIIDivision4),
            (1, 1) => Some(GoldIDivision1),
            (1, 2) => Some(GoldIDivision2),
            (1, 3) => Some(GoldIDivision3),
            (1, 4) => Some(GoldIDivision4),
            _ => None,
        }),
        ("s", |tier, div| match (tier, div) {
            (3, 1) => Some(SilverIIIDivision1),
            (3, 2) => Some(SilverIIIDivision2),
            (3, 3) => Some(SilverIIIDivision3),
            (3, 4) => Some(SilverIIIDivision4),
            (2, 1) => Some(SilverIIDivision1),
            (2, 2) => Some(SilverIIDivision2),
            (2, 3) => Some(SilverIIDivision3),
            (2, 4) => Some(SilverIIDivision4),
            (1, 1) => Some(SilverIDivision1),
            (1, 2) => Some(SilverIDivision2),
            (1, 3) => Some(SilverIDivision3),
            (1, 4) => Some(SilverIDivision4),
            _ => None,
        }),
        ("b", |tier, div| match (tier, div) {
            (3, 1) => Some(BronzeIIIDivision1),
            (3, 2) => Some(BronzeIIIDivision2),
            (3, 3) => Some(BronzeIIIDivision3),
            (3, 4) => Some(BronzeIIIDivision4),
            (2, 1) => Some(BronzeIIDivision1),
            (2, 2) => Some(BronzeIIDivision2),
            (2, 3) => Some(BronzeIIDivision3),
            (2, 4) => Some(BronzeIIDivision4),
            (1, 1) => Some(BronzeIDivision1),
            (1, 2) => Some(BronzeIDivision2),
            (1, 3) => Some(BronzeIDivision3),
            (1, 4) => Some(BronzeIDivision4),
            _ => None,
        }),
    ];

    for (prefix, handler) in patterns {
        if let Some(rest) = s.strip_prefix(prefix)
            && let Some((tier_str, div_str)) = rest.split_once('d')
            && let (Ok(tier), Ok(div)) = (tier_str.parse::<i32>(), div_str.parse::<i32>())
        {
            return handler(tier, div);
        }
    }

    None
}

// ============================================================================
// Rank Models for External APIs
// ============================================================================

/// Rank enum for Rocket League competitive ranks.
///
/// These correspond to rank filter values used by external replay APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum::EnumIter, strum::EnumCount)]
#[cfg_attr(feature = "database", derive(sqlx::Type))]
#[cfg_attr(
    feature = "database",
    sqlx(type_name = "rank", rename_all = "snake_case")
)]
pub enum Rank {
    Unranked,
    Bronze1,
    Bronze2,
    Bronze3,
    Silver1,
    Silver2,
    Silver3,
    Gold1,
    Gold2,
    Gold3,
    Platinum1,
    Platinum2,
    Platinum3,
    Diamond1,
    Diamond2,
    Diamond3,
    Champion1,
    Champion2,
    Champion3,
    GrandChampion1,
    GrandChampion2,
    GrandChampion3,
    SupersonicLegend,
}

impl Rank {
    /// Returns the API string representation for this rank.
    #[must_use]
    pub const fn as_api_string(self) -> &'static str {
        match self {
            Self::Unranked => "unranked",
            Self::Bronze1 => "bronze-1",
            Self::Bronze2 => "bronze-2",
            Self::Bronze3 => "bronze-3",
            Self::Silver1 => "silver-1",
            Self::Silver2 => "silver-2",
            Self::Silver3 => "silver-3",
            Self::Gold1 => "gold-1",
            Self::Gold2 => "gold-2",
            Self::Gold3 => "gold-3",
            Self::Platinum1 => "platinum-1",
            Self::Platinum2 => "platinum-2",
            Self::Platinum3 => "platinum-3",
            Self::Diamond1 => "diamond-1",
            Self::Diamond2 => "diamond-2",
            Self::Diamond3 => "diamond-3",
            Self::Champion1 => "champion-1",
            Self::Champion2 => "champion-2",
            Self::Champion3 => "champion-3",
            Self::GrandChampion1 => "grand-champion-1",
            Self::GrandChampion2 => "grand-champion-2",
            Self::GrandChampion3 => "grand-champion-3",
            Self::SupersonicLegend => "supersonic-legend",
        }
    }

    /// Short name for tables, e.g. "B1", "GC3", "SSL".
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::Unranked => "-",
            Self::Bronze1 => "B1",
            Self::Bronze2 => "B2",
            Self::Bronze3 => "B3",
            Self::Silver1 => "S1",
            Self::Silver2 => "S2",
            Self::Silver3 => "S3",
            Self::Gold1 => "G1",
            Self::Gold2 => "G2",
            Self::Gold3 => "G3",
            Self::Platinum1 => "P1",
            Self::Platinum2 => "P2",
            Self::Platinum3 => "P3",
            Self::Diamond1 => "D1",
            Self::Diamond2 => "D2",
            Self::Diamond3 => "D3",
            Self::Champion1 => "C1",
            Self::Champion2 => "C2",
            Self::Champion3 => "C3",
            Self::GrandChampion1 => "GC1",
            Self::GrandChampion2 => "GC2",
            Self::GrandChampion3 => "GC3",
            Self::SupersonicLegend => "SSL",
        }
    }

    /// Display name, e.g. "Gold III", "Champ I", "GC II", "SSL".
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Unranked => "Unranked",
            Self::Bronze1 => "Bronze I",
            Self::Bronze2 => "Bronze II",
            Self::Bronze3 => "Bronze III",
            Self::Silver1 => "Silver I",
            Self::Silver2 => "Silver II",
            Self::Silver3 => "Silver III",
            Self::Gold1 => "Gold I",
            Self::Gold2 => "Gold II",
            Self::Gold3 => "Gold III",
            Self::Platinum1 => "Platinum I",
            Self::Platinum2 => "Platinum II",
            Self::Platinum3 => "Platinum III",
            Self::Diamond1 => "Diamond I",
            Self::Diamond2 => "Diamond II",
            Self::Diamond3 => "Diamond III",
            Self::Champion1 => "Champ I",
            Self::Champion2 => "Champ II",
            Self::Champion3 => "Champ III",
            Self::GrandChampion1 => "GC I",
            Self::GrandChampion2 => "GC II",
            Self::GrandChampion3 => "GC III",
            Self::SupersonicLegend => "SSL",
        }
    }

    /// Plural display name, e.g. "Gold IIIs", "SSLs".
    #[must_use]
    pub const fn plural_name(self) -> &'static str {
        match self {
            Self::Unranked => "Unranked",
            Self::Bronze1 => "Bronze Is",
            Self::Bronze2 => "Bronze IIs",
            Self::Bronze3 => "Bronze IIIs",
            Self::Silver1 => "Silver Is",
            Self::Silver2 => "Silver IIs",
            Self::Silver3 => "Silver IIIs",
            Self::Gold1 => "Gold Is",
            Self::Gold2 => "Gold IIs",
            Self::Gold3 => "Gold IIIs",
            Self::Platinum1 => "Platinum Is",
            Self::Platinum2 => "Platinum IIs",
            Self::Platinum3 => "Platinum IIIs",
            Self::Diamond1 => "Diamond Is",
            Self::Diamond2 => "Diamond IIs",
            Self::Diamond3 => "Diamond IIIs",
            Self::Champion1 => "Champ Is",
            Self::Champion2 => "Champ IIs",
            Self::Champion3 => "Champ IIIs",
            Self::GrandChampion1 => "GC Is",
            Self::GrandChampion2 => "GC IIs",
            Self::GrandChampion3 => "GC IIIs",
            Self::SupersonicLegend => "SSLs",
        }
    }

    /// The ranked tier at `tier` (0 = Bronze I … 21 = SSL), saturating at SSL.
    #[must_use]
    pub fn from_tier_index(tier: usize) -> Self {
        Self::from_numeric_index_saturating(
            i32::try_from(tier).unwrap_or(i32::MAX).saturating_add(1),
        )
    }

    /// Returns the folder name for storing replays of this rank.
    #[must_use]
    pub const fn as_folder_name(self) -> &'static str {
        self.as_api_string()
    }

    /// Returns an iterator over all ranked tiers (excluding unranked).
    pub fn all_ranked() -> impl Iterator<Item = Self> {
        [
            Self::Bronze1,
            Self::Bronze2,
            Self::Bronze3,
            Self::Silver1,
            Self::Silver2,
            Self::Silver3,
            Self::Gold1,
            Self::Gold2,
            Self::Gold3,
            Self::Platinum1,
            Self::Platinum2,
            Self::Platinum3,
            Self::Diamond1,
            Self::Diamond2,
            Self::Diamond3,
            Self::Champion1,
            Self::Champion2,
            Self::Champion3,
            Self::GrandChampion1,
            Self::GrandChampion2,
            Self::GrandChampion3,
            Self::SupersonicLegend,
        ]
        .into_iter()
    }

    /// Returns an iterator over all ranks including unranked.
    pub fn all() -> impl Iterator<Item = Self> {
        core::iter::once(Self::Unranked).chain(Self::all_ranked())
    }
}

impl core::fmt::Display for Rank {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.as_api_string())
    }
}

impl FromStr for Rank {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_lowercase().replace('_', "-").as_str() {
            "unranked" => Ok(Self::Unranked),
            "bronze-1" => Ok(Self::Bronze1),
            "bronze-2" => Ok(Self::Bronze2),
            "bronze-3" => Ok(Self::Bronze3),
            "silver-1" => Ok(Self::Silver1),
            "silver-2" => Ok(Self::Silver2),
            "silver-3" => Ok(Self::Silver3),
            "gold-1" => Ok(Self::Gold1),
            "gold-2" => Ok(Self::Gold2),
            "gold-3" => Ok(Self::Gold3),
            "platinum-1" => Ok(Self::Platinum1),
            "platinum-2" => Ok(Self::Platinum2),
            "platinum-3" => Ok(Self::Platinum3),
            "diamond-1" => Ok(Self::Diamond1),
            "diamond-2" => Ok(Self::Diamond2),
            "diamond-3" => Ok(Self::Diamond3),
            "champion-1" => Ok(Self::Champion1),
            "champion-2" => Ok(Self::Champion2),
            "champion-3" => Ok(Self::Champion3),
            "grand-champion-1" => Ok(Self::GrandChampion1),
            "grand-champion-2" => Ok(Self::GrandChampion2),
            "grand-champion-3" => Ok(Self::GrandChampion3),
            "supersonic-legend" => Ok(Self::SupersonicLegend),
            _ => Err(anyhow::anyhow!("Invalid rank: {s}")),
        }
    }
}

impl Rank {
    /// Maximum valid index for ranks.
    const MAX_INDEX: usize = Self::COUNT - 1;

    /// Returns the numeric index of this rank (0 = Unranked, 1 = Bronze1, ..., 22 = SupersonicLegend).
    #[must_use]
    pub const fn as_numeric_index(self) -> i32 {
        self as i32
    }

    /// Creates a `Rank` from a numeric index, saturating at boundaries.
    ///
    /// Index 0 = Unranked, 1 = Bronze1, ..., 22 = SupersonicLegend.
    /// Values below 0 return Unranked, values above 22 return SupersonicLegend.
    #[must_use]
    pub fn from_numeric_index_saturating(index: i32) -> Self {
        if index <= 0 {
            return Self::Unranked;
        }
        let clamped = (index as usize).min(Self::MAX_INDEX);
        Self::iter().nth(clamped).unwrap_or(Self::SupersonicLegend)
    }

    /// Adds an offset to this rank, saturating at Unranked (min) and SupersonicLegend (max).
    ///
    /// # Examples
    ///
    /// ```
    /// use replay_structs::Rank;
    ///
    /// assert_eq!(Rank::Bronze1.saturating_add(3), Rank::Silver1);
    /// assert_eq!(Rank::SupersonicLegend.saturating_add(5), Rank::SupersonicLegend);
    /// assert_eq!(Rank::Bronze1.saturating_add(-5), Rank::Unranked);
    /// ```
    #[must_use]
    pub fn saturating_add(self, offset: i32) -> Self {
        let new_index = self.as_numeric_index().saturating_add(offset);
        Self::from_numeric_index_saturating(new_index)
    }

    /// Subtracts an offset from this rank, saturating at Unranked (min) and SupersonicLegend (max).
    ///
    /// # Examples
    ///
    /// ```
    /// use replay_structs::Rank;
    ///
    /// assert_eq!(Rank::Silver1.saturating_sub(3), Rank::Bronze1);
    /// assert_eq!(Rank::Bronze1.saturating_sub(5), Rank::Unranked);
    /// assert_eq!(Rank::Unranked.saturating_sub(1), Rank::Unranked);
    /// ```
    #[must_use]
    pub fn saturating_sub(self, offset: i32) -> Self {
        let new_index = self.as_numeric_index().saturating_sub(offset);
        Self::from_numeric_index_saturating(new_index)
    }
}

impl Add<i32> for Rank {
    type Output = Self;

    /// Adds an offset to this rank, saturating at boundaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use replay_structs::Rank;
    ///
    /// assert_eq!(Rank::Bronze1 + 3, Rank::Silver1);
    /// assert_eq!(Rank::SupersonicLegend + 1, Rank::SupersonicLegend);
    /// ```
    fn add(self, offset: i32) -> Self::Output {
        self.saturating_add(offset)
    }
}

impl Sub<i32> for Rank {
    type Output = Self;

    /// Subtracts an offset from this rank, saturating at boundaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use replay_structs::Rank;
    ///
    /// assert_eq!(Rank::Silver1 - 3, Rank::Bronze1);
    /// assert_eq!(Rank::Bronze1 - 5, Rank::Unranked);
    /// ```
    fn sub(self, offset: i32) -> Self::Output {
        self.saturating_sub(offset)
    }
}

impl Sub<Self> for Rank {
    type Output = i32;

    /// Returns the difference in rank levels between two ranks.
    ///
    /// # Examples
    ///
    /// ```
    /// use replay_structs::Rank;
    ///
    /// assert_eq!(Rank::Silver1 - Rank::Bronze1, 3);
    /// assert_eq!(Rank::Bronze1 - Rank::Silver1, -3);
    /// ```
    fn sub(self, other: Self) -> Self::Output {
        self.as_numeric_index() - other.as_numeric_index()
    }
}

impl From<crate::RankInfo> for RankDivision {
    fn from(rank_info: crate::RankInfo) -> Self {
        if let Some(tier) = rank_info.tier
            && let Some(division) = Self::from_tier_and_division(tier, rank_info.division)
        {
            return division;
        }

        match rank_info.id.as_str() {
            "grand-champion" => Self::GrandChampionIDivision2,
            "grand-champion-1" => Self::from_tier_and_division(19, rank_info.division)
                .unwrap_or(Self::GrandChampionIDivision1),
            "grand-champion-2" => Self::from_tier_and_division(20, rank_info.division)
                .unwrap_or(Self::GrandChampionIIDivision1),
            "grand-champion-3" => Self::from_tier_and_division(21, rank_info.division)
                .unwrap_or(Self::GrandChampionIIIDivision1),
            "supersonic-legend" => Self::SupersonicLegend,
            _ => {
                let tier = rank_info.tier.unwrap_or(1);
                let division = rank_info.division;
                Self::from_tier_and_division(tier, division).unwrap_or(Self::BronzeIDivision1)
            }
        }
    }
}

impl From<i32> for RankDivision {
    /// Converts an MMR value to the corresponding `RankDivision`.
    ///
    /// Uses each division's `mmr_min` and `mmr_max`. When the published tables leave a gap
    /// between one division's maximum and the next division's minimum, the MMR is mapped to the
    /// lower division so [`RankDivision::from`] does not jump backward for increasing MMR.
    ///
    /// # Returns
    ///
    /// - If MMR is below the minimum for Bronze I Division 1, returns `BronzeIDivision1`
    /// - If MMR is above the maximum for Supersonic Legend, returns `SupersonicLegend`
    /// - Otherwise returns the matching rank division, or the last division passed when in a gap
    fn from(mmr: i32) -> Self {
        if mmr < Self::BronzeIDivision1.mmr_min() {
            return Self::BronzeIDivision1;
        }

        let mut chosen = Self::BronzeIDivision1;

        for rank in Self::iter() {
            let rank_min = rank.mmr_min();
            let rank_max = rank.mmr_max();

            if mmr < rank_min {
                return chosen;
            }

            if mmr <= rank_max {
                return rank;
            }

            chosen = rank;
        }

        chosen
    }
}

impl From<f32> for RankDivision {
    /// Converts an MMR value to the corresponding `RankDivision`.
    ///
    /// Same rules as `From<i32>` for `RankDivision`.
    fn from(mmr: f32) -> Self {
        let mmr = mmr as i32;
        Self::from(mmr)
    }
}

impl From<RankDivision> for Rank {
    fn from(division: RankDivision) -> Self {
        match division {
            RankDivision::BronzeIDivision1
            | RankDivision::BronzeIDivision2
            | RankDivision::BronzeIDivision3
            | RankDivision::BronzeIDivision4 => Self::Bronze1,
            RankDivision::BronzeIIDivision1
            | RankDivision::BronzeIIDivision2
            | RankDivision::BronzeIIDivision3
            | RankDivision::BronzeIIDivision4 => Self::Bronze2,
            RankDivision::BronzeIIIDivision1
            | RankDivision::BronzeIIIDivision2
            | RankDivision::BronzeIIIDivision3
            | RankDivision::BronzeIIIDivision4 => Self::Bronze3,
            RankDivision::SilverIDivision1
            | RankDivision::SilverIDivision2
            | RankDivision::SilverIDivision3
            | RankDivision::SilverIDivision4 => Self::Silver1,
            RankDivision::SilverIIDivision1
            | RankDivision::SilverIIDivision2
            | RankDivision::SilverIIDivision3
            | RankDivision::SilverIIDivision4 => Self::Silver2,
            RankDivision::SilverIIIDivision1
            | RankDivision::SilverIIIDivision2
            | RankDivision::SilverIIIDivision3
            | RankDivision::SilverIIIDivision4 => Self::Silver3,
            RankDivision::GoldIDivision1
            | RankDivision::GoldIDivision2
            | RankDivision::GoldIDivision3
            | RankDivision::GoldIDivision4 => Self::Gold1,
            RankDivision::GoldIIDivision1
            | RankDivision::GoldIIDivision2
            | RankDivision::GoldIIDivision3
            | RankDivision::GoldIIDivision4 => Self::Gold2,
            RankDivision::GoldIIIDivision1
            | RankDivision::GoldIIIDivision2
            | RankDivision::GoldIIIDivision3
            | RankDivision::GoldIIIDivision4 => Self::Gold3,
            RankDivision::PlatinumIDivision1
            | RankDivision::PlatinumIDivision2
            | RankDivision::PlatinumIDivision3
            | RankDivision::PlatinumIDivision4 => Self::Platinum1,
            RankDivision::PlatinumIIDivision1
            | RankDivision::PlatinumIIDivision2
            | RankDivision::PlatinumIIDivision3
            | RankDivision::PlatinumIIDivision4 => Self::Platinum2,
            RankDivision::PlatinumIIIDivision1
            | RankDivision::PlatinumIIIDivision2
            | RankDivision::PlatinumIIIDivision3
            | RankDivision::PlatinumIIIDivision4 => Self::Platinum3,
            RankDivision::DiamondIDivision1
            | RankDivision::DiamondIDivision2
            | RankDivision::DiamondIDivision3
            | RankDivision::DiamondIDivision4 => Self::Diamond1,
            RankDivision::DiamondIIDivision1
            | RankDivision::DiamondIIDivision2
            | RankDivision::DiamondIIDivision3
            | RankDivision::DiamondIIDivision4 => Self::Diamond2,
            RankDivision::DiamondIIIDivision1
            | RankDivision::DiamondIIIDivision2
            | RankDivision::DiamondIIIDivision3
            | RankDivision::DiamondIIIDivision4 => Self::Diamond3,
            RankDivision::ChampionIDivision1
            | RankDivision::ChampionIDivision2
            | RankDivision::ChampionIDivision3
            | RankDivision::ChampionIDivision4 => Self::Champion1,
            RankDivision::ChampionIIDivision1
            | RankDivision::ChampionIIDivision2
            | RankDivision::ChampionIIDivision3
            | RankDivision::ChampionIIDivision4 => Self::Champion2,
            RankDivision::ChampionIIIDivision1
            | RankDivision::ChampionIIIDivision2
            | RankDivision::ChampionIIIDivision3
            | RankDivision::ChampionIIIDivision4 => Self::Champion3,
            RankDivision::GrandChampionIDivision1
            | RankDivision::GrandChampionIDivision2
            | RankDivision::GrandChampionIDivision3
            | RankDivision::GrandChampionIDivision4 => Self::GrandChampion1,
            RankDivision::GrandChampionIIDivision1
            | RankDivision::GrandChampionIIDivision2
            | RankDivision::GrandChampionIIDivision3
            | RankDivision::GrandChampionIIDivision4 => Self::GrandChampion2,
            RankDivision::GrandChampionIIIDivision1
            | RankDivision::GrandChampionIIIDivision2
            | RankDivision::GrandChampionIIIDivision3
            | RankDivision::GrandChampionIIIDivision4 => Self::GrandChampion3,
            RankDivision::SupersonicLegend => Self::SupersonicLegend,
        }
    }
}

/// Rank information.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RankInfo {
    /// Rank ID (e.g., "grand-champion-3")
    pub id: String,

    /// Tier number
    pub tier: Option<i32>,

    /// Division number (1-4)
    pub division: Option<i32>,

    /// Human-readable name
    pub name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmr_values() {
        // Check some known MMR values
        assert_eq!(RankDivision::SupersonicLegend.mmr_min(), 1883);
        assert_eq!(RankDivision::SupersonicLegend.mmr_max(), 2000);

        assert_eq!(RankDivision::GrandChampionIIIDivision1.mmr_min(), 1706);
        assert_eq!(RankDivision::GrandChampionIIIDivision1.mmr_max(), 1739);

        assert_eq!(RankDivision::BronzeIDivision1.mmr_min(), 0);
        assert_eq!(RankDivision::BronzeIDivision1.mmr_max(), 117);
    }

    #[test]
    fn test_mmr_middle() {
        // BronzeIDivision1: min=0, max=117, middle=58
        assert_eq!(RankDivision::BronzeIDivision1.mmr_middle(), 58);
        assert_eq!(RankDivision::SupersonicLegend.mmr_middle(), 1941);
    }

    #[test]
    fn test_from_tier_and_division() {
        assert_eq!(
            RankDivision::from_tier_and_division(22, None),
            Some(RankDivision::SupersonicLegend)
        );
        assert_eq!(
            RankDivision::from_tier_and_division(21, Some(1)),
            Some(RankDivision::GrandChampionIIIDivision1)
        );
        assert_eq!(
            RankDivision::from_tier_and_division(21, Some(2)),
            Some(RankDivision::GrandChampionIIIDivision2)
        );
        assert_eq!(
            RankDivision::from_tier_and_division(1, Some(1)),
            Some(RankDivision::BronzeIDivision1)
        );
        assert_eq!(RankDivision::from_tier_and_division(0, Some(1)), None);
        assert_eq!(RankDivision::from_tier_and_division(23, Some(1)), None);
    }

    #[test]
    fn test_from_rank_info() {
        use crate::RankInfo;
        assert_eq!(
            RankDivision::from(RankInfo {
                id: "supersonic-legend".to_string(),
                tier: Some(22),
                division: None,
                name: None,
            }),
            RankDivision::SupersonicLegend
        );
        assert_eq!(
            RankDivision::from(RankInfo {
                id: "grand-champion-3".to_string(),
                tier: Some(21),
                division: Some(1),
                name: None,
            }),
            RankDivision::GrandChampionIIIDivision1
        );
        assert_eq!(
            RankDivision::from(RankInfo {
                id: "bronze-1".to_string(),
                tier: Some(1),
                division: Some(1),
                name: None,
            }),
            RankDivision::BronzeIDivision1
        );
    }

    #[test]
    fn test_as_index() {
        assert_eq!(RankDivision::BronzeIDivision1.as_index(), 0);
        assert_eq!(RankDivision::SupersonicLegend.as_index(), 84);
    }

    #[test]
    fn test_from_str() {
        // Full form
        assert_eq!(
            "SupersonicLegend".parse::<RankDivision>().unwrap(),
            RankDivision::SupersonicLegend
        );
        assert_eq!(
            "Supersonic Legend".parse::<RankDivision>().unwrap(),
            RankDivision::SupersonicLegend
        );
        assert_eq!(
            "Grand Champion III Division 1"
                .parse::<RankDivision>()
                .unwrap(),
            RankDivision::GrandChampionIIIDivision1
        );

        // Short form
        assert_eq!(
            "ssl".parse::<RankDivision>().unwrap(),
            RankDivision::SupersonicLegend
        );
        assert_eq!(
            "gc3d1".parse::<RankDivision>().unwrap(),
            RankDivision::GrandChampionIIIDivision1
        );
        assert_eq!(
            "c2d3".parse::<RankDivision>().unwrap(),
            RankDivision::ChampionIIDivision3
        );
        assert_eq!(
            "d1d4".parse::<RankDivision>().unwrap(),
            RankDivision::DiamondIDivision4
        );
        assert_eq!(
            "b1d1".parse::<RankDivision>().unwrap(),
            RankDivision::BronzeIDivision1
        );
    }

    #[test]
    fn test_from_str_error() {
        assert!("invalid_rank".parse::<RankDivision>().is_err());
        assert!("".parse::<RankDivision>().is_err());
    }

    #[test]
    fn test_display() {
        assert_eq!(
            RankDivision::SupersonicLegend.to_string(),
            "Supersonic Legend"
        );
        assert_eq!(
            RankDivision::GrandChampionIIIDivision1.to_string(),
            "Grand Champion III Division 1"
        );
        assert_eq!(
            RankDivision::BronzeIDivision1.to_string(),
            "Bronze I Division 1"
        );
    }

    #[test]
    fn test_all_ranks_ordered() {
        let ranks: Vec<_> = RankDivision::all().collect();
        assert_eq!(ranks.len(), 85); // 21 ranks * 4 divisions + 1 SSL

        // First should be Bronze I Div 1
        assert_eq!(ranks[0], RankDivision::BronzeIDivision1);
        // Last should be SSL
        assert_eq!(ranks[84], RankDivision::SupersonicLegend);
    }

    #[test]
    fn test_ord() {
        assert!(RankDivision::BronzeIDivision1 < RankDivision::BronzeIDivision2);
        assert!(RankDivision::BronzeIDivision4 < RankDivision::BronzeIIDivision1);
        assert!(RankDivision::GrandChampionIIIDivision4 < RankDivision::SupersonicLegend);
    }

    #[test]
    fn test_from_mmr() {
        // Test exact matches within ranges
        assert_eq!(RankDivision::from(0), RankDivision::BronzeIDivision1);
        assert_eq!(RankDivision::from(100), RankDivision::BronzeIDivision1);
        assert_eq!(RankDivision::from(1900), RankDivision::SupersonicLegend);
        assert_eq!(RankDivision::from(2000), RankDivision::SupersonicLegend);

        // Test edge cases: below minimum
        assert_eq!(RankDivision::from(-100), RankDivision::BronzeIDivision1);

        // Test edge cases: above maximum
        assert_eq!(RankDivision::from(3000), RankDivision::SupersonicLegend);

        // Test mid-range values
        assert_eq!(RankDivision::from(1200), RankDivision::ChampionIIDivision1);
        assert_eq!(
            RankDivision::from(1500),
            RankDivision::GrandChampionIDivision3
        );
    }

    #[test]
    fn test_rank_add() {
        // Basic addition (Bronze1 has index 1)
        assert_eq!(Rank::Bronze1 + 3, Rank::Silver1); // 1 + 3 = 4 = Silver1
        assert_eq!(Rank::Bronze1 + 1, Rank::Bronze2); // 1 + 1 = 2 = Bronze2
        assert_eq!(Rank::Bronze1 + 5, Rank::Silver3); // 1 + 5 = 6 = Silver3
        assert_eq!(Rank::Bronze1 + 6, Rank::Gold1); // 1 + 6 = 7 = Gold1

        // Saturation at max
        assert_eq!(Rank::SupersonicLegend + 1, Rank::SupersonicLegend);
        assert_eq!(Rank::Champion3 + 5, Rank::SupersonicLegend);

        // Adding zero
        assert_eq!(Rank::Gold1 + 0, Rank::Gold1);

        // Adding negative (same as subtraction)
        assert_eq!(Rank::Silver1 + (-3), Rank::Bronze1); // 4 - 3 = 1 = Bronze1
    }

    #[test]
    fn test_rank_sub() {
        // Basic subtraction
        assert_eq!(Rank::Silver1 - 3, Rank::Bronze1);
        assert_eq!(Rank::Silver1 - 1, Rank::Bronze3);

        // Saturation at min
        assert_eq!(Rank::Bronze1 - 5, Rank::Unranked);
        assert_eq!(Rank::Unranked - 1, Rank::Unranked);

        // Subtracting zero
        assert_eq!(Rank::Gold1 - 0, Rank::Gold1);

        // Subtracting negative (same as addition)
        assert_eq!(Rank::Bronze1 - (-3), Rank::Silver1);
    }

    #[test]
    fn test_rank_difference() {
        // Difference between ranks
        assert_eq!(Rank::Silver1 - Rank::Bronze1, 3);
        assert_eq!(Rank::Bronze1 - Rank::Silver1, -3);
        assert_eq!(Rank::SupersonicLegend - Rank::Bronze1, 21);
        assert_eq!(Rank::Bronze1 - Rank::Bronze1, 0);
        assert_eq!(Rank::SupersonicLegend - Rank::Unranked, 22);
    }

    #[test]
    fn test_rank_numeric_index() {
        assert_eq!(Rank::Unranked.as_numeric_index(), 0);
        assert_eq!(Rank::Bronze1.as_numeric_index(), 1);
        assert_eq!(Rank::Bronze2.as_numeric_index(), 2);
        assert_eq!(Rank::Bronze3.as_numeric_index(), 3);
        assert_eq!(Rank::Silver1.as_numeric_index(), 4);
        assert_eq!(Rank::SupersonicLegend.as_numeric_index(), 22);
    }

    #[test]
    fn test_rank_from_numeric_index() {
        assert_eq!(Rank::from_numeric_index_saturating(0), Rank::Unranked);
        assert_eq!(Rank::from_numeric_index_saturating(1), Rank::Bronze1);
        assert_eq!(Rank::from_numeric_index_saturating(4), Rank::Silver1);
        assert_eq!(
            Rank::from_numeric_index_saturating(22),
            Rank::SupersonicLegend
        );

        // Saturation
        assert_eq!(Rank::from_numeric_index_saturating(-5), Rank::Unranked);
        assert_eq!(
            Rank::from_numeric_index_saturating(100),
            Rank::SupersonicLegend
        );
    }

    #[test]
    fn test_rank_divison_from_mmr() {
        assert_eq!(RankDivision::from(0), RankDivision::BronzeIDivision1);
        assert_eq!(RankDivision::from(1395), RankDivision::ChampionIIIDivision3);

        assert_eq!(
            RankDivision::from(1489),
            RankDivision::GrandChampionIDivision2
        );
        assert_eq!(RankDivision::from(1900), RankDivision::SupersonicLegend);
        assert_eq!(RankDivision::from(2000), RankDivision::SupersonicLegend);

        // Test edge cases: below minimum
        assert_eq!(RankDivision::from(-100), RankDivision::BronzeIDivision1);
    }

    #[test]
    fn test_rank_division_from_mmr_monotonic_through_full_range() {
        let mut previous = RankDivision::from(0);
        for mmr in 1..=2000 {
            let current = RankDivision::from(mmr);
            assert!(
                current >= previous,
                "MMR {mmr}: expected rank >= previous rank ({previous:?}); got {current:?}"
            );
            previous = current;
        }
    }
}
