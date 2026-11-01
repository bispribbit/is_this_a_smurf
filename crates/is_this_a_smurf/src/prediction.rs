#![expect(
    clippy::indexing_slicing,
    reason = "per-slot arrays of length TOTAL_SLOTS indexed by slot < TOTAL_SLOTS"
)]

//! Turns a [`MatchAnalysis`] into what the timeline and the cards display.

use feature_extractor::TOTAL_SLOTS;
use rand::{RngExt, rng};
use replay_structs::{GameFrame, MatchFormat, ParsedReplay, RankDivision, Team};
use skill_model::mechanics::{MechanicEvidence, TIER_SINGULARS};
use skill_model::{MatchAnalysis, PlayerTimeline, SmurfReason};

use crate::app_state::{
    GoalMarkerDisplay, PlayerAverage, PredictionResults, SegmentStepInfo, StepStatus,
};

/// Cached regulation-clock parameters derived once from the frame list.
///
/// Rocket League replays store an absolute `time` (seconds from replay start) and a
/// `seconds_remaining` game-clock countdown (starts at 300 for a 5-minute match).
/// After the clock reaches 0 the ball stays live (ball-on-ground rule) and overtime
/// also plays with `seconds_remaining == 0`.
///
/// Play-time is what a viewer sees on ballchasing.com: 0:00 at kickoff, 5:00 at the
/// end of regulation, and beyond 5:00 for overtime / ball-on-ground.
pub(crate) struct PlayTimeClock {
    pub(crate) regulation_seconds: f32,
    replay_time_at_clock_zero: Option<f32>,
}

impl PlayTimeClock {
    pub(crate) fn from_frames(frames: &[GameFrame]) -> Self {
        let regulation_seconds = frames
            .first()
            .map_or(300.0, |frame| frame.seconds_remaining as f32);
        let replay_time_at_clock_zero = frames
            .iter()
            .find(|frame| frame.seconds_remaining <= 0)
            .map(|frame| frame.time);
        Self {
            regulation_seconds,
            replay_time_at_clock_zero,
        }
    }

    pub(crate) fn play_time(&self, frame: &GameFrame) -> f32 {
        if frame.seconds_remaining > 0 {
            self.regulation_seconds - frame.seconds_remaining as f32
        } else if let Some(zero_time) = self.replay_time_at_clock_zero {
            self.regulation_seconds + (frame.time - zero_time).max(0.0)
        } else {
            self.regulation_seconds
        }
    }
}

/// One step per timeline window, with its replay-clock time range, ranks not yet revealed.
pub(crate) fn segment_step_infos(
    frames: &[GameFrame],
    timeline: &[PlayerTimeline],
) -> Vec<SegmentStepInfo> {
    timeline
        .iter()
        .map(|window| SegmentStepInfo {
            start_time: frames
                .get(window.frames.start)
                .map_or(0.0, |frame| frame.time),
            end_time: frames
                .get(window.frames.end.saturating_sub(1))
                .map_or(0.0, |frame| frame.time),
            status: StepStatus::Pending,
            player_segment_ranks: None,
        })
        .collect()
}

/// Game-clock time at each window boundary (start of every window, then the end of the
/// last). Used only for the labels under the timeline axis.
pub(crate) fn compute_segment_boundary_play_times(
    frames: &[GameFrame],
    timeline: &[PlayerTimeline],
) -> Vec<f32> {
    let clock = PlayTimeClock::from_frames(frames);
    let mut play_times: Vec<f32> = timeline
        .iter()
        .map(|window| {
            frames
                .get(window.frames.start)
                .map_or(0.0, |frame| clock.play_time(frame))
        })
        .collect();
    if let Some(last) = timeline.last() {
        play_times.push(
            frames
                .get(last.frames.end.saturating_sub(1))
                .map_or(0.0, |frame| clock.play_time(frame)),
        );
    }
    play_times
}

/// Rank badges for one window; absent players stay `None`.
pub(crate) fn ranks_from_player_mmr(
    player_mmr: &[Option<f32>; TOTAL_SLOTS],
) -> [Option<RankDivision>; TOTAL_SLOTS] {
    core::array::from_fn(|slot| player_mmr[slot].map(RankDivision::from))
}

/// Times (seconds) at each segment boundary: start of segment 0, then starts of 1..n, then end of last segment.
pub(crate) fn compute_segment_boundary_times(segment_steps: &[SegmentStepInfo]) -> Vec<f32> {
    let Some(first_step) = segment_steps.first() else {
        return Vec::new();
    };
    let mut boundary_times_seconds = Vec::with_capacity(segment_steps.len() + 1);
    boundary_times_seconds.push(first_step.start_time);
    for step in segment_steps.iter().skip(1) {
        boundary_times_seconds.push(step.start_time);
    }
    if let Some(last_step) = segment_steps.last() {
        boundary_times_seconds.push(last_step.end_time);
    }
    boundary_times_seconds
}

/// One timeline lane per player present in the match, blue first, with each lane's model
/// slot.
pub(crate) struct TimelinePlayers {
    pub(crate) names: Vec<String>,
    pub(crate) teams: Vec<Team>,
    pub(crate) slots: Vec<usize>,
}

/// Lanes for the players present in the match (a duel has two, a casual lobby can have
/// more than six).
pub(crate) fn prepare_players_for_timeline(analysis: &MatchAnalysis) -> TimelinePlayers {
    let slots: Vec<usize> = (0..TOTAL_SLOTS)
        .filter(|&slot| !analysis.names[slot].is_empty())
        .collect();
    TimelinePlayers {
        names: slots
            .iter()
            .map(|&slot| analysis.names[slot].clone())
            .collect(),
        teams: slots
            .iter()
            .map(|&slot| MatchAnalysis::team(slot))
            .collect(),
        slots,
    }
}

/// Goal markers for the timeline, derived from replay header goals and frame times.
/// Uses absolute replay time for positioning on the equal-width segment timeline.
pub(crate) fn build_goal_markers(
    parsed: &ParsedReplay,
    player_names: &[String],
) -> Vec<GoalMarkerDisplay> {
    let mut markers = Vec::new();
    for goal in &parsed.goals {
        let Some(time_seconds) = parsed.frames.get(goal.frame).map(|frame| frame.time) else {
            continue;
        };
        let player_lane_index = player_names
            .iter()
            .position(|name| name == &goal.player_name);
        markers.push(GoalMarkerDisplay {
            time_seconds,
            scorer_name: goal.player_name.clone(),
            team: goal.player_team,
            player_lane_index,
        });
    }
    markers
}

/// Median of the lobby's whole-match predictions: the rank the verdict's no-smurf lines are
/// written for.
pub(crate) fn lobby_median_mmr(player_averages: &[PlayerAverage]) -> f32 {
    let mut values: Vec<f32> = player_averages.iter().map(|player| player.mmr).collect();
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    let upper = values.get(middle).copied().unwrap_or(0.0);
    if values.len().is_multiple_of(2) {
        let lower = values.get(middle.wrapping_sub(1)).copied().unwrap_or(upper);
        f32::midpoint(lower, upper)
    } else {
        upper
    }
}

/// Formats seconds as `m:ss` (no leading zero on minutes) for boundary labels under the timeline.
pub(crate) fn format_timeline_boundary_label(seconds: f32) -> String {
    let total_seconds = seconds.max(0.0) as u32;
    let minutes = total_seconds / 60;
    let secs = total_seconds % 60;
    format!("{minutes}:{secs:02}")
}

/// Cards and verdict input: every player the match model scored, in slot order.
/// Mechanics a card shows at most.
const MECHANICS_SHOWN: usize = 3;

/// Mechanics an unflagged player must do far above the lobby before the card names them.
const MECHANICS_FOR_UNFLAGGED: usize = 3;

/// "Speed flips like a Champ I · flip resets like a GC I", or `None` when there is nothing
/// worth showing.
fn mechanics_line(evidence: &[MechanicEvidence], flagged: bool) -> Option<String> {
    if evidence.is_empty() || (!flagged && evidence.len() < MECHANICS_FOR_UNFLAGGED) {
        return None;
    }
    let parts: Vec<String> = evidence
        .iter()
        .take(MECHANICS_SHOWN)
        .map(|item| {
            let tier = TIER_SINGULARS
                .get(item.typical_tier)
                .copied()
                .unwrap_or("SSL");
            format!("{} like a {tier}", item.mechanic.label)
        })
        .collect();
    let line = parts.join(" · ");
    let mut characters = line.chars();
    characters
        .next()
        .map(|first| first.to_uppercase().chain(characters).collect())
}

pub(crate) fn build_prediction_results(
    analysis: &MatchAnalysis,
    match_format: MatchFormat,
) -> PredictionResults {
    let mut random = rng();
    let player_averages = (0..TOTAL_SLOTS)
        .filter(|&slot| !analysis.names[slot].is_empty())
        .filter_map(|slot| {
            let mmr = analysis.match_mmr[slot]?;
            Some(PlayerAverage {
                name: analysis.names[slot].clone(),
                team: MatchAnalysis::team(slot),
                mmr,
                rank: RankDivision::from(mmr),
                next_rank_roast: analysis.coaching[slot].as_ref().map(|advice| {
                    advice.line(random.random_range(0..usize::MAX), &analysis.names[slot])
                }),
                smurf: analysis.smurf[slot],
                plays_like: analysis.plays_like_mmr[slot]
                    .filter(|_| {
                        matches!(
                            analysis.smurf[slot],
                            Some(SmurfReason::PlaysAboveLobby | SmurfReason::DerankingAboveLobby)
                        )
                    })
                    .map(RankDivision::from),
                mechanics: mechanics_line(
                    &analysis.mechanics[slot],
                    analysis.smurf[slot].is_some(),
                ),
            })
        })
        .collect();
    PredictionResults {
        player_averages,
        match_format,
    }
}
