//! The smurf flag: the single definition the app's badge, its verdict copy and the offline
//! evaluation tools all use.
//!
//! A player is flagged for either of two reasons:
//!
//! * **ranked above the lobby** — their whole-match prediction is
//!   [`SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR`] over the lobby's median. It mostly finds
//!   parties carrying a weaker friend in standard; in duels and doubles it almost never
//!   fires, because the labels it learned from hardly ever differ inside those lobbies;
//! * **plays above the lobby** — the rank their own stats play like
//!   ([`crate::PlaysLikeModel`]) is more than [`SMURF_MARGIN_OVER_LOBBY_MEDIAN_TIERS`]
//!   divisions over the lobby's median. Counted in divisions rather than MMR because divisions are much
//!   wider at the top: in MMR, a third of these flags landed on Diamond-and-up players.
//!
//! `docs/model.md` has how both margins were chosen.

use feature_extractor::TOTAL_SLOTS;

use crate::coaching::tier_index;

/// Margin above the lobby's median whole-match prediction at which a player is flagged.
///
/// Chosen from the precision/recall table `train` prints for the evaluation split: about
/// 0.4 % of players flagged, ~60 % of them genuinely ranked well above their lobby.
pub const SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR: f32 = 100.0;

/// Divisions (tiers) above the lobby's median plays-like division that a player must
/// exceed to be flagged: from a Gold I lobby, playing like a Platinum III or higher.
///
/// On the evaluation split it flags ~2 % of standard and ~4 % of doubles players, catches
/// half the obvious smurfs there, and under 10 % of its flags land on Diamond-and-up labels.
pub const SMURF_MARGIN_OVER_LOBBY_MEDIAN_TIERS: f32 = 4.0;

/// Why a player is flagged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmurfReason {
    /// Whole-match prediction well above the lobby's.
    RankedAboveLobby,
    /// Plays several divisions above the lobby (and is not already ranked above it).
    PlaysAboveLobby,
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

/// Flags per slot from the whole-match predictions and the plays-like predictions. A lobby
/// with fewer than two scored players has no one to compare with and flags nobody.
#[must_use]
pub fn smurf_suspicions(
    match_mmr: &[Option<f32>; TOTAL_SLOTS],
    plays_like_mmr: &[Option<f32>; TOTAL_SLOTS],
) -> [Option<SmurfReason>; TOTAL_SLOTS] {
    let mut predictions: Vec<f32> = match_mmr.iter().flatten().copied().collect();
    let mut plays_like_tiers: Vec<f32> = plays_like_mmr
        .iter()
        .flatten()
        .map(|&mmr| tier_index(mmr) as f32)
        .collect();
    if predictions.len() < 2 {
        return [None; TOTAL_SLOTS];
    }
    let prediction_median = median(&mut predictions);
    let tier_median = median(&mut plays_like_tiers);
    let mut flags = [None; TOTAL_SLOTS];
    for ((flag, mmr), plays_like) in flags.iter_mut().zip(match_mmr).zip(plays_like_mmr) {
        let ranked_above = mmr
            .zip(prediction_median)
            .is_some_and(|(mmr, median)| mmr > median + SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR);
        let plays_above = plays_like.zip(tier_median).is_some_and(|(mmr, median)| {
            tier_index(mmr) as f32 > median + SMURF_MARGIN_OVER_LOBBY_MEDIAN_TIERS
        });
        *flag = if ranked_above {
            Some(SmurfReason::RankedAboveLobby)
        } else if plays_above {
            Some(SmurfReason::PlaysAboveLobby)
        } else {
            None
        };
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slots(values: &[f32]) -> [Option<f32>; TOTAL_SLOTS] {
        core::array::from_fn(|slot| values.get(slot).copied())
    }

    #[test]
    fn a_duel_flags_the_player_who_plays_far_above() {
        // Both ranked Bronze I by the match model; one plays like a Champion I.
        let flags = smurf_suspicions(&slots(&[100.0, 110.0]), &slots(&[1100.0, 100.0]));
        assert_eq!(flags[0], Some(SmurfReason::PlaysAboveLobby));
        assert_eq!(flags[1], None);
    }

    #[test]
    fn four_divisions_up_is_not_enough() {
        // Gold II (580) over three Silver Is (320): exactly four divisions.
        let flags = smurf_suspicions(
            &slots(&[300.0, 300.0, 300.0, 300.0]),
            &slots(&[320.0, 320.0, 320.0, 580.0]),
        );
        assert!(flags.iter().all(Option::is_none));
    }

    #[test]
    fn ranked_above_the_lobby_wins_over_plays_above() {
        let flags = smurf_suspicions(
            &slots(&[300.0, 300.0, 300.0, 600.0]),
            &slots(&[290.0, 290.0, 290.0, 1100.0]),
        );
        assert_eq!(flags[3], Some(SmurfReason::RankedAboveLobby));
    }

    #[test]
    fn a_lone_player_is_never_flagged() {
        let flags = smurf_suspicions(&slots(&[2000.0]), &slots(&[2000.0]));
        assert!(flags.iter().all(Option::is_none));
    }
}
