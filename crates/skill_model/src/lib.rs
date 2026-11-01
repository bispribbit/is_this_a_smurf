//! Per-player skill model shipped in the "Is this a smurf?" app.
//!
//! Burn-free on purpose: everything here runs unchanged in the WASM app. A parsed replay is
//! summarised per player with [`feature_extractor::compute_player_match_stats`], gradient
//! boosted trees turn the summaries into MMR estimates ([`TabularSkillModel`]), a second
//! model does the same per 60-second window for the timeline, one model per stat group scores
//! the rank each player's own stats play like ([`plays_like`]) for the smurf flag
//! ([`smurf_suspicions`]), [`mechanics`] names the mechanics a player does far more than
//! their lobby's rank, and [`coaching`] picks one "here's what the next rank does
//! better" roast per player.
//!
//! Everything ships as one [`SkillModelBundle`], stored in a compact binary format
//! ([`SkillModelBundle::to_bytes`]).
//!
//! See `docs/model.md` for how the model was chosen and measured.

mod analysis;
pub mod coaching;
mod encoding;
pub mod mechanics;
mod model;
pub mod plays_like;
mod smurf;

pub use analysis::{MINIMUM_MATCH_SECONDS, MINIMUM_WINDOW_SECONDS, MatchAnalysis, PlayerTimeline};
pub use encoding::{BundleDecodeError, BundleEncodeError, SkillModelBundle};
pub use model::{
    FrozenEnsemble, FrozenNode, LobbySummary, OPPONENT_INDEPENDENT_STATS, PlaysLikeModel,
    TabularLayout, TabularSkillModel,
};
pub use plays_like::{
    GroupPrediction, PlaysLikeGroup, PlaysLikeGroups, PlaysLikePredictions, STAT_GROUPS, StatGroup,
};
pub use smurf::{
    DERANKING_MINIMUM_MARGIN_TIERS, MINIMUM_MECHANICS_ABOVE_LOBBY,
    SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR, SlotEvidence, SmurfReason, plays_like_margins,
    smurf_margin_over_others_tiers, smurf_suspicions,
};
