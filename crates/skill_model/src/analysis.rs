#![expect(
    clippy::indexing_slicing,
    reason = "per-slot arrays of length TOTAL_SLOTS indexed by slot < TOTAL_SLOTS"
)]

//! Turns a parsed replay into everything the app shows.
//!
//! * **Whole-match scores** come from the match model — the cards.
//! * **Smurf flags** compare the match model and the plays-like model with the lobby
//!   ([`crate::smurf_suspicions`]) — the badges and the verdict.
//! * **The timeline** shows *form*: each player's whole-match score, moved up or down by how
//!   much better or worse they did in a window than in their own average window, times
//!   [`SkillModelBundle::timeline_emphasis`]. Window-level lobby estimates jump by ±100 MMR
//!   from one minute to the next for the whole lobby at once (see `docs/model.md`),
//!   so the timeline never shows them directly.
//! * **One roast per player** from [`crate::coaching`].

use std::ops::Range;

use feature_extractor::{
    MATCH_STAT_COUNT, PlayerMatchStats, PlayerRoster, SLOTS_PER_TEAM, TOTAL_SLOTS,
    compute_player_match_stats, compute_player_window_stats, time_windows,
};
use replay_structs::{ParsedReplay, Team};

use crate::SkillModelBundle;
use crate::coaching::{CoachingAdvice, smurf_line};
use crate::mechanics::{MechanicEvidence, mechanics_above_lobby};
use crate::smurf::{SmurfReason, smurf_suspicions};

/// Players with less live time than this in a window get no timeline score there. Same
/// cut-off the window model was trained with.
pub const MINIMUM_WINDOW_SECONDS: f32 = 10.0;

/// Live seconds a player needs over the whole match to be scored.
///
/// Shorter players do not count towards the lobby either: mostly casual players who joined
/// for the last few seconds. Same cut-off the match model was trained with.
pub const MINIMUM_MATCH_SECONDS: f32 = 30.0;

/// One timeline window: its frame range and each slot's displayed MMR.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerTimeline {
    /// Raw frame range of the window.
    pub frames: Range<usize>,
    /// Displayed MMR per slot, `None` when the player was (nearly) absent.
    pub player_mmr: [Option<f32>; TOTAL_SLOTS],
}

/// Everything the app needs for one replay.
#[derive(Debug, Clone)]
pub struct MatchAnalysis {
    /// Player names in slot order (blue sorted by name, then orange); empty when unused.
    pub names: [String; TOTAL_SLOTS],
    /// Whole-match MMR per slot.
    pub match_mmr: [Option<f32>; TOTAL_SLOTS],
    /// MMR each player's own stats play like, per slot (the stat groups' weighted mean).
    pub plays_like_mmr: [Option<f32>; TOTAL_SLOTS],
    /// Why each slot is flagged as a smurf, if it is.
    pub smurf: [Option<SmurfReason>; TOTAL_SLOTS],
    /// Per slot: the named mechanics done far more than the other players' tier, highest
    /// typical tier first.
    pub mechanics: [Vec<MechanicEvidence>; TOTAL_SLOTS],
    /// Stat tip per slot; [`Self::roast`] replaces it with a smurf line for flagged slots.
    pub coaching: [Option<CoachingAdvice>; TOTAL_SLOTS],
    /// Timeline windows in order.
    pub timeline: Vec<PlayerTimeline>,
}

impl MatchAnalysis {
    /// Team of a slot (the first [`SLOTS_PER_TEAM`] slots are blue).
    #[must_use]
    pub const fn team(slot: usize) -> Team {
        if slot < SLOTS_PER_TEAM {
            Team::Blue
        } else {
            Team::Orange
        }
    }

    /// The roast shown for a slot, line `variant`: a smurf line for a flagged player (a
    /// smurf gets no help), otherwise their stat tip.
    #[must_use]
    pub fn roast(&self, slot: usize, variant: usize) -> Option<String> {
        let name = self.names.get(slot)?;
        if self.smurf.get(slot)?.is_some() {
            return Some(smurf_line(variant, name));
        }
        self.coaching
            .get(slot)?
            .as_ref()
            .map(|advice| advice.line(variant, name))
    }
}

/// Mean over present slots.
fn present_mean(values: &[Option<f32>; TOTAL_SLOTS]) -> f32 {
    let present: Vec<f32> = values.iter().flatten().copied().collect();
    present.iter().sum::<f32>() / present.len().max(1) as f32
}

impl SkillModelBundle {
    /// Scores one replay.
    #[must_use]
    pub fn analyze(&self, parsed: &ParsedReplay) -> MatchAnalysis {
        let roster = PlayerRoster::from_parsed(parsed);
        let mut match_stats = compute_player_match_stats(parsed, &roster);
        for slot_stats in &mut match_stats {
            if slot_stats
                .as_ref()
                .is_some_and(|s| s.live_seconds < MINIMUM_MATCH_SECONDS)
            {
                *slot_stats = None;
            }
        }
        let match_mmr = self.match_model.predict(&match_stats);
        let plays_like = self.plays_like_groups.predict(&match_stats);
        let plays_like_mmr = plays_like.combined_mmr();
        let stat_values: [Option<&[f32; MATCH_STAT_COUNT]>; TOTAL_SLOTS] =
            core::array::from_fn(|slot| match_stats[slot].as_ref().map(|stats| &stats.values));
        let mechanics = mechanics_above_lobby(
            self.mechanics
                .for_players_per_team(parsed.match_format.players_per_team),
            &stat_values,
            &match_mmr,
        );
        let smurf = smurf_suspicions(
            &match_mmr,
            &plays_like,
            &crate::smurf::SlotEvidence::per_slot(&mechanics, &stat_values),
            parsed.match_format.players_per_team,
        );

        let coaching_table = self
            .coaching
            .for_players_per_team(parsed.match_format.players_per_team);
        let coaching = core::array::from_fn(|slot| {
            let stats: &PlayerMatchStats = match_stats[slot].as_ref()?;
            coaching_table?.advise(match_mmr[slot]?, &stats.values)
        });
        // A player flagged for how they play is shown at the level they play like, so the
        // timeline agrees with the flag instead of sitting at the lobby's rank.
        let timeline_level: [Option<f32>; TOTAL_SLOTS] =
            core::array::from_fn(|slot| match smurf[slot] {
                Some(SmurfReason::PlaysAboveLobby | SmurfReason::DerankingAboveLobby) => {
                    plays_like_mmr[slot].or(match_mmr[slot])
                }
                _ => match_mmr[slot],
            });
        let timeline = self.timeline(parsed, &roster, &timeline_level);
        MatchAnalysis {
            names: roster.names,
            match_mmr,
            plays_like_mmr,
            smurf,
            mechanics,
            coaching,
            timeline,
        }
    }

    fn timeline(
        &self,
        parsed: &ParsedReplay,
        roster: &PlayerRoster,
        level: &[Option<f32>; TOTAL_SLOTS],
    ) -> Vec<PlayerTimeline> {
        let windows = time_windows(parsed, self.window_seconds);
        // Each window: every present player's deviation from that window's lobby, and how
        // long they played in it.
        let mut deviations: Vec<[Option<f32>; TOTAL_SLOTS]> = Vec::with_capacity(windows.len());
        let mut weights: Vec<[f32; TOTAL_SLOTS]> = Vec::with_capacity(windows.len());
        for range in &windows {
            let mut stats = compute_player_window_stats(parsed, roster, range.clone());
            for slot_stats in &mut stats {
                if slot_stats
                    .as_ref()
                    .is_some_and(|s| s.live_seconds < MINIMUM_WINDOW_SECONDS)
                {
                    *slot_stats = None;
                }
            }
            let predictions = self.window_model.predict(&stats);
            let lobby = present_mean(&predictions);
            deviations.push(core::array::from_fn(|slot| {
                predictions[slot].map(|value| value - lobby)
            }));
            weights.push(core::array::from_fn(|slot| {
                stats[slot].as_ref().map_or(0.0, |s| s.live_seconds)
            }));
        }

        // A player's average window deviation, so the timeline shows form around their own
        // `level` rather than a second, competing estimate of it.
        let average_deviation: [f32; TOTAL_SLOTS] = core::array::from_fn(|slot| {
            let mut weighted = 0.0;
            let mut total = 0.0;
            for (window, weight) in deviations.iter().zip(&weights) {
                if let Some(deviation) = window[slot] {
                    weighted = deviation.mul_add(weight[slot], weighted);
                    total += weight[slot];
                }
            }
            if total > 0.0 { weighted / total } else { 0.0 }
        });

        windows
            .into_iter()
            .zip(deviations)
            .map(|(frames, window)| PlayerTimeline {
                frames,
                player_mmr: core::array::from_fn(|slot| {
                    let form = window[slot]? - average_deviation[slot];
                    Some(self.timeline_emphasis.mul_add(form, level[slot]?))
                }),
            })
            .collect()
    }
}
