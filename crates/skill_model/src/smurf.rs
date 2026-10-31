//! The smurf flag: the single definition the app's badge, its verdict copy and the offline
//! evaluation tools all use.
//!
//! A player is flagged for either of two reasons:
//!
//! * **ranked above the lobby** — their whole-match prediction is
//!   [`SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR`] over the lobby's median. It mostly finds
//!   parties carrying a weaker friend in standard; in duels and doubles it almost never
//!   fires, because the labels it learned from hardly ever differ inside those lobbies;
//! * **plays above the others** — the rank their own stats play like
//!   ([`crate::PlaysLikeModel`]) is more than [`smurf_margin_over_others_tiers`] divisions
//!   over the median of the *other* players. Counted in divisions rather than MMR because
//!   divisions are much wider at the top (in MMR, a third of these flags landed on
//!   Diamond-and-up players), and against the others because in a duel the lobby median is
//!   half way to the smurf, which halved their margin.
//!
//! `docs/model.md` has how both margins were chosen.

use feature_extractor::TOTAL_SLOTS;

use crate::coaching::tier_index;

/// Margin above the lobby's median whole-match prediction at which a player is flagged.
///
/// Chosen from the precision/recall table `train` prints for the evaluation split: about
/// 0.4 % of players flagged, ~60 % of them genuinely ranked well above their lobby.
pub const SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR: f32 = 100.0;

/// Divisions (tiers) over the other players' median plays-like division that a player
/// must exceed to be flagged, for a playlist of `players_per_team`.
///
/// Smaller lobbies compare against fewer players, so their median is noisier and needs a
/// wider margin: more than five divisions in duels and doubles (from a Silver I opponent,
/// playing like a Platinum I or higher), more than four in standard.
///
/// Set for recall: a smurf detector and coach would rather flag a strong player than miss a
/// smurf. One division tighter flags a third fewer players but misses more smurfs; on the
/// evaluation split this flags ~4 % of duels, ~5 % of doubles and ~4.5 % of standard players.
#[must_use]
pub const fn smurf_margin_over_others_tiers(players_per_team: usize) -> f32 {
    if players_per_team >= 3 { 4.0 } else { 5.0 }
}

/// Why a player is flagged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmurfReason {
    /// Whole-match prediction well above the lobby's.
    RankedAboveLobby,
    /// Plays several divisions above the other players (and is not already ranked above
    /// the lobby).
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

/// Flags per slot from the whole-match and plays-like predictions.
///
/// `players_per_team` picks the playlist's margin. A lobby with fewer than two scored
/// players has no one to compare with and flags nobody.
#[must_use]
pub fn smurf_suspicions(
    match_mmr: &[Option<f32>; TOTAL_SLOTS],
    plays_like_mmr: &[Option<f32>; TOTAL_SLOTS],
    players_per_team: usize,
) -> [Option<SmurfReason>; TOTAL_SLOTS] {
    let mut predictions: Vec<f32> = match_mmr.iter().flatten().copied().collect();
    if predictions.len() < 2 {
        return [None; TOTAL_SLOTS];
    }
    let prediction_median = median(&mut predictions);
    let tiers: [Option<f32>; TOTAL_SLOTS] =
        plays_like_mmr.map(|mmr| mmr.map(|mmr| tier_index(mmr) as f32));
    let tier_margin = smurf_margin_over_others_tiers(players_per_team);
    let mut flags = [None; TOTAL_SLOTS];
    for (slot, (flag, mmr)) in flags.iter_mut().zip(match_mmr).enumerate() {
        let ranked_above = mmr
            .zip(prediction_median)
            .is_some_and(|(mmr, median)| mmr > median + SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR);
        let mut others: Vec<f32> = tiers
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != slot)
            .filter_map(|(_, tier)| *tier)
            .collect();
        let plays_above = tiers
            .get(slot)
            .copied()
            .flatten()
            .zip(median(&mut others))
            .is_some_and(|(tier, others_median)| tier > others_median + tier_margin);
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
        let flags = smurf_suspicions(&slots(&[100.0, 110.0]), &slots(&[1100.0, 100.0]), 1);
        assert_eq!(flags[0], Some(SmurfReason::PlaysAboveLobby));
        assert_eq!(flags[1], None);
    }

    #[test]
    fn four_divisions_over_the_others_is_not_enough_in_standard() {
        // Gold II (580) over three Silver Is (320): exactly four divisions.
        let flags = smurf_suspicions(
            &slots(&[300.0, 300.0, 300.0, 300.0]),
            &slots(&[320.0, 320.0, 320.0, 580.0]),
            3,
        );
        assert!(flags.iter().all(Option::is_none));
    }

    #[test]
    fn ranked_above_the_lobby_wins_over_plays_above() {
        let flags = smurf_suspicions(
            &slots(&[300.0, 300.0, 300.0, 600.0]),
            &slots(&[290.0, 290.0, 290.0, 1100.0]),
            2,
        );
        assert_eq!(flags[3], Some(SmurfReason::RankedAboveLobby));
    }

    #[test]
    fn a_lone_player_is_never_flagged() {
        let flags = smurf_suspicions(&slots(&[2000.0]), &slots(&[2000.0]), 1);
        assert!(flags.iter().all(Option::is_none));
    }
}
