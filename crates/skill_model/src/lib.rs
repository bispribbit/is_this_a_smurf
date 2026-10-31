//! Per-player skill model shipped in the "Is this a smurf?" app.
//!
//! Burn-free on purpose: everything here runs unchanged in the WASM app. A parsed replay is
//! summarised per player with [`feature_extractor::compute_player_match_stats`], gradient
//! boosted trees turn the summaries into MMR estimates ([`TabularSkillModel`]), a second
//! model does the same per 60-second window for the timeline, a third scores the rank each
//! player's own stats play like ([`PlaysLikeModel`]) for the smurf flag
//! ([`smurf_suspicions`]), and [`coaching`] picks one "here's what the next rank does
//! better" roast per player.
//!
//! Everything ships as one [`SkillModelBundle`], stored in a compact binary format
//! ([`SkillModelBundle::to_bytes`]).
//!
//! See `docs/model.md` for how the model was chosen and measured.

mod analysis;
pub mod coaching;
mod encoding;
mod model;
mod smurf;

pub use analysis::{MINIMUM_MATCH_SECONDS, MINIMUM_WINDOW_SECONDS, MatchAnalysis, PlayerTimeline};
pub use encoding::{BundleDecodeError, BundleEncodeError, SkillModelBundle};
pub use model::{
    FrozenEnsemble, FrozenNode, LobbySummary, OPPONENT_INDEPENDENT_STATS, PlaysLikeModel,
    TabularLayout, TabularSkillModel,
};
pub use smurf::{
    SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR, SmurfReason, smurf_margin_over_others_tiers,
    smurf_suspicions,
};
