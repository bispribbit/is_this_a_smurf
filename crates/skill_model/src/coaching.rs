//! One roast per player: what the next rank up does that you don't.
//!
//! Each player is compared with the **median player [`TARGET_TIER_STEP`] divisions above
//! their predicted division** (Diamond I → Diamond II, Champion III → Grand Champion I) on
//! a curated list of stats that a person can act on. The stat with the largest shortfall,
//! measured in that division's own spread, picks the line. Because the reference is always
//! just above, a bronze is told to hit the ball forward and a champ to
//! fast-aerial — never the reverse.
//!
//! Only stats that actually improve from the player's tier to the target tier are eligible
//! (see [`MINIMUM_PROGRESSION`]); otherwise everyone would be told to do whatever varies
//! most between individual players, whether or not it is what separates the ranks.
//!
//! The reference table ([`CoachingTable`]) is built from training data by the exporter and
//! shipped inside the model bundle.

use feature_extractor::MatchStat;
use serde::{Deserialize, Serialize};

mod lines;

/// How many tiers above the player's predicted tier the comparison targets.
///
/// The next one (Diamond I → Diamond II): the roast is about the rank the card shows, so it
/// names the step right above it rather than one the player was never placed near.
pub const TARGET_TIER_STEP: usize = 1;

/// How many tiers above the player's tier a stat must improve over to count.
///
/// One division is too close: neighbouring medians barely move and their differences are
/// mostly noise.
pub const PROGRESSION_TIER_STEP: usize = 3;

/// A stat must move by at least this many spreads between the player's tier and
/// [`PROGRESSION_TIER_STEP`] tiers above to count as something that separates the ranks.
pub const MINIMUM_PROGRESSION: f32 = 0.15;

/// The player must trail the target tier's median by at least this many spreads.
pub const MINIMUM_SHORTFALL: f32 = 0.25;

/// Which way is better for a stat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Better {
    Higher,
    Lower,
}

/// How a stat value is written in a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// A 0–1 fraction shown as a percentage.
    Percent,
    /// A rate shown as `x.x/min`.
    PerMinute,
    /// A whole-match count.
    Count,
    /// Ball speed normalised by 6,000 uu/s, shown in km/h.
    BallSpeed,
}

impl Unit {
    /// Smallest gap worth roasting someone over, in this unit's raw value. Below it the two
    /// numbers in the line look the same to a reader ("0% vs 1%"), whatever the spread says.
    #[must_use]
    pub const fn minimum_gap(self) -> f32 {
        match self {
            Self::Percent => 0.05,
            Self::PerMinute => 0.8,
            Self::Count => 1.0,
            // ≈ 6 km/h.
            Self::BallSpeed => 0.03,
        }
    }

    /// Formats one value for a line.
    #[must_use]
    pub fn format(self, value: f32) -> String {
        match self {
            Self::Percent => format!("{:.0}%", value * 100.0),
            Self::PerMinute => format!("{value:.1}/min"),
            Self::Count => format!("{value:.0}"),
            // 6,000 uu/s × 0.036 km/h per uu/s.
            Self::BallSpeed => format!("{:.0} km/h", value * 216.0),
        }
    }
}

/// Placeholder for the player's gamertag in lines.
const NAME_PLACEHOLDER: &str = "{name}";

/// One coachable stat and its lines.
///
/// Lines use `{name}`, `{yours}`, `{theirs}` and `{next}` (plural division name, e.g.
/// "Gold IIs").
#[derive(Debug, Clone, Copy)]
pub struct CoachingTip {
    pub stat: MatchStat,
    pub better: Better,
    pub unit: Unit,
    pub lines: &'static [&'static str],
}

/// The curated, actionable stats. Order is the table's column order.
pub const COACHING_TIPS: &[CoachingTip] = &[
    CoachingTip {
        stat: MatchStat::TouchesPerMinute,
        better: Better::Higher,
        unit: Unit::PerMinute,
        lines: lines::TOUCHES_PER_MINUTE,
    },
    CoachingTip {
        stat: MatchStat::ForwardTouchFraction,
        better: Better::Higher,
        unit: Unit::Percent,
        lines: lines::FORWARD_TOUCH_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::MeanBallSpeedAfterTouch,
        better: Better::Higher,
        unit: Unit::BallSpeed,
        lines: lines::MEAN_BALL_SPEED_AFTER_TOUCH,
    },
    CoachingTip {
        stat: MatchStat::BoostEmptyFraction,
        better: Better::Lower,
        unit: Unit::Percent,
        lines: lines::BOOST_EMPTY_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::SmallPadsPerMinute,
        better: Better::Higher,
        unit: Unit::PerMinute,
        lines: lines::SMALL_PADS_PER_MINUTE,
    },
    CoachingTip {
        stat: MatchStat::SupersonicFraction,
        better: Better::Higher,
        unit: Unit::Percent,
        lines: lines::SUPERSONIC_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::SlowFraction,
        better: Better::Lower,
        unit: Unit::Percent,
        lines: lines::SLOW_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::OvercommitFraction,
        better: Better::Lower,
        unit: Unit::Percent,
        lines: lines::OVERCOMMIT_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::GoalSideWhenDefendingFraction,
        better: Better::Higher,
        unit: Unit::Percent,
        lines: lines::GOAL_SIDE_WHEN_DEFENDING_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::DoubleCommitFraction,
        better: Better::Lower,
        unit: Unit::Percent,
        lines: lines::DOUBLE_COMMIT_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::GoalSideAfterTouchFraction,
        better: Better::Higher,
        unit: Unit::Percent,
        lines: lines::GOAL_SIDE_AFTER_TOUCH_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::KickoffWinFraction,
        better: Better::Higher,
        unit: Unit::Percent,
        lines: lines::KICKOFF_WIN_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::DemosReceived,
        better: Better::Lower,
        unit: Unit::Count,
        lines: lines::DEMOS_RECEIVED,
    },
    CoachingTip {
        stat: MatchStat::DodgesPerMinute,
        better: Better::Higher,
        unit: Unit::PerMinute,
        lines: lines::DODGES_PER_MINUTE,
    },
    CoachingTip {
        stat: MatchStat::HighAirFraction,
        better: Better::Higher,
        unit: Unit::Percent,
        lines: lines::HIGH_AIR_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::AerialTouchFraction,
        better: Better::Higher,
        unit: Unit::Percent,
        lines: lines::AERIAL_TOUCH_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::FastAerialFraction,
        better: Better::Higher,
        unit: Unit::Percent,
        lines: lines::FAST_AERIAL_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::WavedashLikeDodgesPerMinute,
        better: Better::Higher,
        unit: Unit::PerMinute,
        lines: lines::WAVEDASH_LIKE_DODGES_PER_MINUTE,
    },
    CoachingTip {
        stat: MatchStat::PowerslideFraction,
        better: Better::Higher,
        unit: Unit::Percent,
        lines: lines::POWERSLIDE_FRACTION,
    },
    CoachingTip {
        stat: MatchStat::CleanLandingFraction,
        better: Better::Higher,
        unit: Unit::Percent,
        lines: lines::CLEAN_LANDING_FRACTION,
    },
];

/// Lines for a player with nothing left to learn from the rank above (or already SSL).
pub const NOTHING_TO_FIX_LINES: &[&str] = lines::NOTHING_TO_FIX;

/// Lines for a flagged smurf, who gets a roast instead of a stat tip.
pub const SMURF_LINES: &[&str] = lines::SMURF;

/// Renders smurf line `variant` (wrapped to the available lines) about player `name`.
#[must_use]
pub fn smurf_line(variant: usize, name: &str) -> String {
    SMURF_LINES
        .get(variant % SMURF_LINES.len())
        .copied()
        .unwrap_or_default()
        .replace(NAME_PLACEHOLDER, name)
}

/// Reference values of one tier for every tip, aligned with [`COACHING_TIPS`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TierReference {
    /// Median of each tip's stat among players labelled with this tier.
    pub medians: Vec<f32>,
    /// Robust spread (inter-quartile range / 1.349) of each tip's stat in this tier.
    pub spreads: Vec<f32>,
}

/// Reference stats for every tier, Bronze I (index 0) to Supersonic Legend (index 21).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CoachingTable {
    pub tiers: Vec<TierReference>,
    /// Per tip: whether the data agrees with its declared direction (medians move the
    /// declared way from Bronze to SSL). Tips the data contradicts are never used.
    pub enabled: Vec<bool>,
}

/// One [`CoachingTable`] per playlist size: what the next rank does differs between duels,
/// doubles and standard (a duel player touches the ball far more often).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CoachingTables {
    /// Index = players per team − 1 (duels, doubles, standard). A table without tiers means
    /// that playlist had no training data.
    pub by_players_per_team: Vec<CoachingTable>,
}

impl CoachingTables {
    /// The table for a playlist size, or the standard table when that size has none.
    #[must_use]
    pub fn for_players_per_team(&self, players_per_team: usize) -> Option<&CoachingTable> {
        let has_tiers = |table: &&CoachingTable| !table.tiers.is_empty();
        players_per_team
            .checked_sub(1)
            .and_then(|index| self.by_players_per_team.get(index))
            .filter(has_tiers)
            .or_else(|| self.by_players_per_team.get(2).filter(has_tiers))
    }
}

/// Tier index (0 = Bronze I … 21 = SSL) for an MMR value.
#[must_use]
pub fn tier_index(mmr: f32) -> usize {
    let rank: replay_structs::Rank = replay_structs::RankDivision::from(mmr).into();
    usize::try_from(rank.as_numeric_index() - 1)
        .unwrap_or(0)
        .min(21)
}

/// The chosen advice for one player.
#[derive(Debug, Clone)]
pub struct CoachingAdvice {
    /// The tip, or `None` when there is nothing to fix.
    pub tip: Option<&'static CoachingTip>,
    /// The player's value of the tip's stat.
    pub yours: f32,
    /// The target tier's median.
    pub theirs: f32,
    /// Plural name of the target division.
    pub next: &'static str,
    /// Shortfall in target-tier spreads (0 when there is nothing to fix).
    pub shortfall: f32,
}

impl CoachingAdvice {
    /// Renders the advice about player `name` with line variant `variant` (wrapped to the
    /// available lines).
    #[must_use]
    pub fn line(&self, variant: usize, name: &str) -> String {
        let Some(tip) = self.tip else {
            return NOTHING_TO_FIX_LINES
                .get(variant % NOTHING_TO_FIX_LINES.len())
                .copied()
                .unwrap_or_default()
                .replace(NAME_PLACEHOLDER, name);
        };
        tip.lines
            .get(variant % tip.lines.len().max(1))
            .copied()
            .unwrap_or_default()
            .replace(NAME_PLACEHOLDER, name)
            .replace("{yours}", &tip.unit.format(self.yours))
            .replace("{theirs}", &tip.unit.format(self.theirs))
            .replace("{next}", self.next)
    }
}

impl CoachingTable {
    /// Picks the advice for a player predicted at `predicted_mmr` with match stats `stats`.
    ///
    /// Returns `None` only when the table is empty (no reference data shipped).
    #[must_use]
    pub fn advise(&self, predicted_mmr: f32, stats: &[f32]) -> Option<CoachingAdvice> {
        let last = self.tiers.len().checked_sub(1)?;
        let current = tier_index(predicted_mmr).min(last);
        let target = (current + TARGET_TIER_STEP).min(last);
        let next = replay_structs::Rank::from_tier_index(target).plural_name();
        let nothing = CoachingAdvice {
            tip: None,
            yours: 0.0,
            theirs: 0.0,
            next,
            shortfall: 0.0,
        };
        if target == current {
            return Some(nothing);
        }
        let current_reference = self.tiers.get(current)?;
        let target_reference = self.tiers.get(target)?;
        let progression_reference = self
            .tiers
            .get((current + PROGRESSION_TIER_STEP).min(last))?;

        let mut best = nothing;
        for (tip_index, tip) in COACHING_TIPS.iter().enumerate() {
            if !self.enabled.get(tip_index).copied().unwrap_or(false) {
                continue;
            }
            let Some(&yours) = stats.get(tip.stat.index()) else {
                continue;
            };
            let Some(&theirs) = target_reference.medians.get(tip_index) else {
                continue;
            };
            let Some(&spread) = target_reference.spreads.get(tip_index) else {
                continue;
            };
            let Some(&current_median) = current_reference.medians.get(tip_index) else {
                continue;
            };
            let Some(&progression_median) = progression_reference.medians.get(tip_index) else {
                continue;
            };
            let Some(&progression_spread) = progression_reference.spreads.get(tip_index) else {
                continue;
            };
            if spread <= f32::EPSILON || progression_spread <= f32::EPSILON {
                continue;
            }
            let sign = match tip.better {
                Better::Higher => 1.0,
                Better::Lower => -1.0,
            };
            let progression = sign * (progression_median - current_median) / progression_spread;
            let shortfall = sign * (theirs - yours) / spread;
            let visible = (theirs - yours).abs() >= tip.unit.minimum_gap()
                && tip.unit.format(theirs) != tip.unit.format(yours);
            if visible
                && progression >= MINIMUM_PROGRESSION
                && shortfall >= MINIMUM_SHORTFALL
                && shortfall > best.shortfall
            {
                best = CoachingAdvice {
                    tip: Some(tip),
                    yours,
                    theirs,
                    next,
                    shortfall,
                };
            }
        }
        Some(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tip_column(stat: MatchStat) -> usize {
        COACHING_TIPS
            .iter()
            .position(|tip| tip.stat == stat)
            .unwrap_or(usize::MAX)
    }

    /// A table where only touches/min and air time improve with rank; air time only
    /// starts mattering from Gold up.
    fn table() -> CoachingTable {
        let tiers = (0..22)
            .map(|tier| {
                let mut medians = vec![0.5; COACHING_TIPS.len()];
                let spreads = vec![0.1; COACHING_TIPS.len()];
                medians[tip_column(MatchStat::TouchesPerMinute)] = 1.0f32.mul_add(tier as f32, 2.0);
                medians[tip_column(MatchStat::HighAirFraction)] = if tier < 6 {
                    0.5
                } else {
                    0.1f32.mul_add((tier - 6) as f32, 0.5)
                };
                TierReference { medians, spreads }
            })
            .collect();
        CoachingTable {
            tiers,
            enabled: vec![true; COACHING_TIPS.len()],
        }
    }

    /// Several players in one match often get the same tip; enough variety keeps them from
    /// all reading the same sentence.
    #[test]
    fn every_tip_has_enough_distinct_lines() {
        for tip in COACHING_TIPS {
            assert!(tip.lines.len() >= 20, "{} has too few lines", tip.stat);
            for line in tip.lines {
                for placeholder in ["{yours}", "{theirs}", "{next}"] {
                    assert!(line.contains(placeholder), "{line:?} lacks {placeholder}");
                }
            }
            let distinct: std::collections::HashSet<_> = tip.lines.iter().collect();
            assert_eq!(
                distinct.len(),
                tip.lines.len(),
                "{} repeats a line",
                tip.stat
            );
        }
        assert!(NOTHING_TO_FIX_LINES.len() >= 20);
        assert!(SMURF_LINES.len() >= 20);
    }

    /// Lines describe one of several players, so they never talk to the reader.
    #[test]
    fn lines_never_address_the_reader() {
        let all_lines = COACHING_TIPS
            .iter()
            .flat_map(|tip| tip.lines.iter())
            .chain(NOTHING_TO_FIX_LINES)
            .chain(SMURF_LINES);
        for line in all_lines {
            let addresses_reader = ["{name}", "{yours}", "{theirs}", "{next}"]
                .iter()
                .fold(line.to_string(), |text, placeholder| {
                    text.replace(placeholder, "")
                })
                .split(|character: char| !character.is_alphanumeric() && character != '\'')
                .any(|word| {
                    let word = word.to_lowercase();
                    word == "you" || word.starts_with("you'") || word.starts_with("your")
                });
            assert!(!addresses_reader, "{line:?} addresses the reader");
        }
    }

    /// A low-ranked player who never jumps is still told to touch the ball: air time does
    /// not separate their tier from the target tier, however far behind they are.
    #[test]
    fn advice_targets_what_separates_the_next_rank() {
        let mut stats = vec![0.5; feature_extractor::MATCH_STAT_COUNT];
        stats[MatchStat::TouchesPerMinute.index()] = 1.0;
        stats[MatchStat::HighAirFraction.index()] = 0.0;
        let bronze = table().advise(100.0, &stats).expect("table is not empty");
        assert_eq!(
            bronze.tip.map(|tip| tip.stat),
            Some(MatchStat::TouchesPerMinute)
        );
        assert_eq!(bronze.next, "Bronze IIs");
        assert!(bronze.line(0, "Player").contains("Bronze IIs"));
    }

    #[test]
    fn nothing_to_fix_when_ahead_of_the_next_rank() {
        let stats = vec![10.0; feature_extractor::MATCH_STAT_COUNT];
        let advice = table().advise(100.0, &stats).expect("table is not empty");
        assert!(advice.tip.is_none());
        assert!(NOTHING_TO_FIX_LINES.contains(&advice.line(0, "Player").as_str()));
    }

    #[test]
    fn units_format_for_humans() {
        assert_eq!(Unit::Percent.format(0.123), "12%");
        assert_eq!(Unit::PerMinute.format(7.04), "7.0/min");
        assert_eq!(Unit::BallSpeed.format(0.5), "108 km/h");
    }
}
