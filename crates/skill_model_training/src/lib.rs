//! Training for the shipped skill model ([`skill_model`]).
//!
//! Two main binaries: `extract_stats` parses every downloaded replay into per-player stats
//! CSVs (whole match and 60-second windows), and `train` drops likely smurf labels, fits
//! the gradient-boosted trees, prints the evaluation, builds the coaching table and writes
//! the bundle the app embeds. The other binaries are experiments.
//! See `docs/model.md`.

pub mod account_cleaning;
pub mod coaching_table;
pub mod dataset;
pub mod evaluation;
pub mod gradient_boosting;
pub mod label_cleaning;
pub mod mechanic_table;
pub mod smurf_references;
