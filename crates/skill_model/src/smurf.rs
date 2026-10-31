//! The smurf flag: the single definition the app's badge, its verdict copy and the offline
//! evaluation tools all use.
//!
//! A player is flagged for one of three reasons:
//!
//! * **ranked above the lobby** — their whole-match prediction is
//!   [`SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR`] over the lobby's median. It mostly finds
//!   parties carrying a weaker friend in standard; in duels and doubles it almost never
//!   fires, because the labels it learned from hardly ever differ inside those lobbies;
//! * **plays above the others** — per stat group ([`crate::plays_like`]), the division the
//!   player's own stats play like, over the median of the *other* players (each scored half
//!   by that group and half by the whole-match prediction); the groups' margins are
//!   averaged by weight, and the player is flagged above [`smurf_margin_over_others_tiers`].
//!   Counted in divisions rather than MMR because divisions are much wider at the top (in
//!   MMR, a third of these flags landed on Diamond-and-up players), and against the others
//!   because in a duel the lobby median is half way to the smurf, which halved their margin;
//! * **mechanics above the lobby** — [`MINIMUM_MECHANICS_ABOVE_LOBBY`] or more named
//!   mechanics done more than the 95th percentile of the other players' tier
//!   ([`crate::mechanics`]). On its own it flags too many strong players, so it only adds
//!   to the others: in duels it caught three more of 14 mechanical and four more of 61
//!   dominant smurfs for one more percent flagged.
//!
//! `docs/model.md` has how both margins were chosen.

use feature_extractor::TOTAL_SLOTS;

use crate::coaching::tier_index;
use crate::plays_like::PlaysLikePredictions;

/// Margin above the lobby's median whole-match prediction at which a player is flagged.
///
/// Chosen from the precision/recall table `train` prints for the evaluation split: about
/// 0.4 % of players flagged, ~60 % of them genuinely ranked well above their lobby.
pub const SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR: f32 = 100.0;

/// Weighted plays-like margin, in divisions over the other players, that a player must
/// exceed to be flagged, for a playlist of `players_per_team`.
///
/// Chosen from the flag-rate table `train` prints so each playlist flags about what the
/// weighted groups were measured at (~6.5 % of duels and doubles players, ~4.5 % of
/// standard), a rate set for recall: a smurf detector and coach would rather flag a strong
/// player than miss a smurf. Doubles teammates share the ball, so the groups spread wider
/// there and need the widest margin. `docs/model.md` has the flag rates and recall.
#[must_use]
pub const fn smurf_margin_over_others_tiers(players_per_team: usize) -> f32 {
    match players_per_team {
        1 => 3.75,
        2 => 5.0,
        _ => 4.25,
    }
}

/// Named mechanics above the lobby's tier that flag a player on their own.
pub const MINIMUM_MECHANICS_ABOVE_LOBBY: usize = 4;

/// Why a player is flagged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::enum_variant_names,
    reason = "every reason is a way of standing above the lobby"
)]
pub enum SmurfReason {
    /// Whole-match prediction well above the lobby's.
    RankedAboveLobby,
    /// Plays several divisions above the other players (and is not already ranked above
    /// the lobby).
    PlaysAboveLobby,
    /// Does several named mechanics far more than the other players' rank (and neither of
    /// the above).
    MechanicsAboveLobby,
}

fn median(values: &mut [f32]) -> Option<f32> {
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    let upper = *values.get(middle)?;
    Some(if values.len().is_multiple_of(2) {
        let lower = values.get(middle.wrapping_sub(1)).copied().unwrap_or(upper);
        f32::midpoint(lower, upper)
    } else {
        upper
    })
}

fn tier(mmr: f32) -> f32 {
    tier_index(mmr) as f32
}

/// One group's margin for every slot: the slot's division over the median of the other
/// players, each scored with the midpoint of their division and the whole-match
/// prediction's. The whole-match model reads the whole lobby, so one opponent's off game
/// moves the baseline half as much (in duels, a third of the flags on Diamond-and-up
/// players went away).
fn group_margins(
    plays_like_mmr: &[Option<f32>; TOTAL_SLOTS],
    match_mmr: &[Option<f32>; TOTAL_SLOTS],
) -> [Option<f32>; TOTAL_SLOTS] {
    let tiers: [Option<f32>; TOTAL_SLOTS] = plays_like_mmr.map(|mmr| mmr.map(tier));
    let others_tiers: [Option<f32>; TOTAL_SLOTS] = core::array::from_fn(|slot| {
        let plays_like = tiers.get(slot).copied().flatten()?;
        Some(
            match_mmr
                .get(slot)
                .copied()
                .flatten()
                .map_or(plays_like, |mmr| f32::midpoint(plays_like, tier(mmr))),
        )
    });
    core::array::from_fn(|slot| {
        let own = tiers.get(slot).copied().flatten()?;
        let mut others: Vec<f32> = others_tiers
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != slot)
            .filter_map(|(_, tier)| *tier)
            .collect();
        median(&mut others).map(|others_median| own - others_median)
    })
}

/// Weighted plays-like margin per slot, in divisions over the other players: what
/// [`smurf_suspicions`] compares with [`smurf_margin_over_others_tiers`].
#[must_use]
pub fn plays_like_margins(
    match_mmr: &[Option<f32>; TOTAL_SLOTS],
    plays_like: &PlaysLikePredictions,
) -> [Option<f32>; TOTAL_SLOTS] {
    let margins: Vec<[Option<f32>; TOTAL_SLOTS]> = plays_like
        .groups
        .iter()
        .map(|group| group_margins(&group.mmr, match_mmr))
        .collect();
    core::array::from_fn(|slot| {
        let mut weighted = 0.0;
        let mut weights = 0.0;
        for (group, group_margins) in plays_like.groups.iter().zip(&margins) {
            if let Some(Some(margin)) = group_margins.get(slot) {
                weighted += group.weight * margin;
                weights += group.weight;
            }
        }
        (weights > 0.0).then(|| weighted / weights)
    })
}

/// Flags per slot from the whole-match and plays-like predictions and, per slot, how many
/// named mechanics the player does above the lobby's tier
/// ([`crate::mechanics::mechanics_above_lobby`]).
///
/// `players_per_team` picks the playlist's margin. A lobby with fewer than two scored
/// players has no one to compare with and flags nobody.
#[must_use]
pub fn smurf_suspicions(
    match_mmr: &[Option<f32>; TOTAL_SLOTS],
    plays_like: &PlaysLikePredictions,
    mechanics_above: &[usize; TOTAL_SLOTS],
    players_per_team: usize,
) -> [Option<SmurfReason>; TOTAL_SLOTS] {
    let mut predictions: Vec<f32> = match_mmr.iter().flatten().copied().collect();
    if predictions.len() < 2 {
        return [None; TOTAL_SLOTS];
    }
    let prediction_median = median(&mut predictions);
    let margins = plays_like_margins(match_mmr, plays_like);
    let tier_margin = smurf_margin_over_others_tiers(players_per_team);
    let mut flags = [None; TOTAL_SLOTS];
    for (((flag, mmr), margin), mechanics) in flags
        .iter_mut()
        .zip(match_mmr)
        .zip(margins)
        .zip(mechanics_above)
    {
        let ranked_above = mmr
            .zip(prediction_median)
            .is_some_and(|(mmr, median)| mmr > median + SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR);
        let plays_above = margin.is_some_and(|margin| margin > tier_margin);
        *flag = if ranked_above {
            Some(SmurfReason::RankedAboveLobby)
        } else if plays_above {
            Some(SmurfReason::PlaysAboveLobby)
        } else if *mechanics >= MINIMUM_MECHANICS_ABOVE_LOBBY {
            Some(SmurfReason::MechanicsAboveLobby)
        } else {
            None
        };
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plays_like::GroupPrediction;

    fn slots(values: &[f32]) -> [Option<f32>; TOTAL_SLOTS] {
        core::array::from_fn(|slot| values.get(slot).copied())
    }

    /// One group's weight and per-slot MMRs.
    struct TestGroup<'values> {
        weight: f32,
        mmr: &'values [f32],
    }

    fn groups(groups: &[TestGroup<'_>]) -> PlaysLikePredictions {
        PlaysLikePredictions {
            groups: groups
                .iter()
                .map(|group| GroupPrediction {
                    weight: group.weight,
                    mmr: slots(group.mmr),
                })
                .collect(),
        }
    }

    fn one_group(mmr: &[f32]) -> PlaysLikePredictions {
        groups(&[TestGroup { weight: 1.0, mmr }])
    }

    #[test]
    fn a_duel_flags_the_player_who_plays_far_above() {
        // Both ranked Bronze I by the match model; one plays like a Champion I.
        let flags = smurf_suspicions(
            &slots(&[100.0, 110.0]),
            &one_group(&[1100.0, 100.0]),
            &[0; TOTAL_SLOTS],
            1,
        );
        assert_eq!(flags[0], Some(SmurfReason::PlaysAboveLobby));
        assert_eq!(flags[1], None);
    }

    #[test]
    fn a_strong_group_alone_is_averaged_down() {
        // Champion I aerials (15 divisions up), Bronze I everything else, weighted 1 to 3.
        let plays_like = groups(&[
            TestGroup {
                weight: 1.0,
                mmr: &[1100.0, 100.0],
            },
            TestGroup {
                weight: 3.0,
                mmr: &[100.0, 100.0],
            },
        ]);
        let margins = plays_like_margins(&slots(&[100.0, 100.0]), &plays_like);
        let margin = margins[0].unwrap_or_default();
        assert!((margin - 15.0 / 4.0).abs() < 1e-4, "margin {margin}");
    }

    #[test]
    fn ranked_above_the_lobby_wins_over_plays_above() {
        let flags = smurf_suspicions(
            &slots(&[300.0, 300.0, 300.0, 600.0]),
            &one_group(&[290.0, 290.0, 290.0, 1100.0]),
            &[0; TOTAL_SLOTS],
            2,
        );
        assert_eq!(flags[3], Some(SmurfReason::RankedAboveLobby));
    }

    #[test]
    fn a_lone_player_is_never_flagged() {
        let flags = smurf_suspicions(
            &slots(&[2000.0]),
            &one_group(&[2000.0]),
            &[MINIMUM_MECHANICS_ABOVE_LOBBY; TOTAL_SLOTS],
            1,
        );
        assert!(flags.iter().all(Option::is_none));
    }

    #[test]
    fn enough_mechanics_above_the_lobby_flag_on_their_own() {
        let mut mechanics = [0; TOTAL_SLOTS];
        mechanics[1] = MINIMUM_MECHANICS_ABOVE_LOBBY;
        let flags = smurf_suspicions(
            &slots(&[500.0, 500.0]),
            &one_group(&[500.0, 500.0]),
            &mechanics,
            1,
        );
        assert_eq!(flags[0], None);
        assert_eq!(flags[1], Some(SmurfReason::MechanicsAboveLobby));
    }
}
