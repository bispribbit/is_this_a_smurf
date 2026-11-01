//! Application state, progress tracking, and result payloads.

use feature_extractor::TOTAL_SLOTS;
use replay_structs::{MatchFormat, RankDivision, Team, UnsupportedReplayMatch};

/// Prediction results for the entire replay.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PredictionResults {
    /// Whole-match result per player.
    pub(crate) player_averages: Vec<PlayerAverage>,
    /// Team size and queue of the match; ranks are always the competitive rank for that
    /// team size, also for casual matches.
    pub(crate) match_format: MatchFormat,
}

/// Whole-match result for a single player.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlayerAverage {
    /// Player name.
    pub(crate) name: String,
    /// Player team.
    pub(crate) team: Team,
    /// Whole-match predicted MMR.
    pub(crate) mmr: f32,
    /// Rank derived from [`Self::mmr`].
    pub(crate) rank: RankDivision,
    /// One-line roast: what the next rank up does better.
    pub(crate) next_rank_roast: Option<String>,
    /// Why the player is flagged as a smurf, if they are.
    pub(crate) smurf: Option<skill_model::SmurfReason>,
    /// The rank their own stats play like, when that is why they are flagged.
    pub(crate) plays_like: Option<RankDivision>,
    /// "Forfeited while ahead (3–1)" when they are flagged for forfeiting while level or ahead.
    pub(crate) forfeit: Option<String>,
    /// The mechanics they do far more than the lobby's rank, e.g. "Speed flips like a
    /// Champ I · flip resets like a GC I": shown for flagged players, and for anyone doing
    /// several.
    pub(crate) mechanics: Option<String>,
}

/// Status of a single step in the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StepStatus {
    Pending,
    Processing,
    Done(String),
}

/// Progress for one segment (time range + status).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SegmentStepInfo {
    pub(crate) start_time: f32,
    pub(crate) end_time: f32,
    pub(crate) status: StepStatus,
    /// Filled when the window is revealed; a slot is `None` when that player was (nearly)
    /// absent from the window.
    pub(crate) player_segment_ranks: Option<[Option<RankDivision>; TOTAL_SLOTS]>,
}

/// One goal shown on the analysis timeline (replay-derived).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GoalMarkerDisplay {
    pub(crate) time_seconds: f32,
    pub(crate) scorer_name: String,
    pub(crate) team: Team,
    pub(crate) player_lane_index: Option<usize>,
}

/// State for the animated match timeline during processing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TimelineTrackState {
    pub(crate) match_duration_seconds: f32,
    /// Absolute replay-clock boundary times (used for positioning).
    pub(crate) boundary_times_seconds: Vec<f32>,
    /// Game-clock (play-time) at each boundary (used for display labels only).
    pub(crate) boundary_play_times_seconds: Vec<f32>,
    pub(crate) goals: Vec<GoalMarkerDisplay>,
    /// One lane per player present in the match.
    pub(crate) player_names: Vec<String>,
    pub(crate) player_teams: Vec<Team>,
    /// Model slot of each lane, to read that player's window ranks.
    pub(crate) player_slots: Vec<usize>,
    pub(crate) num_segments: usize,
}

/// Live progress during analysis (parsing, model load, timeline windows).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProgressState {
    /// Reading file bytes from the browser file API (can take a moment for large replays).
    pub(crate) reading_file: StepStatus,
    /// Copying the `ArrayBuffer` into a Rust `Vec` (synchronous; often the slowest step after the browser read).
    pub(crate) copying_into_memory: StepStatus,
    pub(crate) parsing: StepStatus,
    pub(crate) loading_model: StepStatus,
    pub(crate) segments: Vec<SegmentStepInfo>,
    /// Present while inferring segments (and briefly for the global rank reveal).
    pub(crate) timeline: Option<TimelineTrackState>,
}

/// The different states the application can be in.
#[derive(Debug, Clone)]
pub(crate) enum AppState {
    /// Waiting for the user to upload a replay file (processing also
    /// happens while in this state, with progress shown via [`LocalProcessing`]).
    WaitingForUpload,
    /// An error occurred (error message).
    Error(String),
    /// Replay parsed but the match type is not supported (e.g. Hoops or 4v4).
    UnsupportedReplay(UnsupportedReplayMatch),
}

/// Processing state shown by `UploadPage`; owned by `App` together with the pipeline future,
/// so a replay dropped on any screen can restart the analysis.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LocalProcessing {
    /// Name of the file being processed.
    pub(crate) filename: String,
    /// Current progress.
    pub(crate) progress: ProgressState,
    /// Filled when inference completes; keeps the timeline and summary on one screen without routing.
    pub(crate) results: Option<PredictionResults>,
}
