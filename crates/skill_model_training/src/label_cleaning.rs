#![expect(
    clippy::indexing_slicing,
    reason = "slot arrays indexed by slots below TOTAL_SLOTS, tier counts by tier_index"
)]

//! Drops training labels that are most likely smurf accounts.
//!
//! A smurf is labelled with the smurf account's rank, so a Champion on a Bronze account
//! teaches every model that Champion stats happen in Bronze (about a tenth of Bronze I
//! duelists play like Champions). Each training player gets an out-of-fold plays-like
//! prediction (own stats only, from a model that never saw their replay); a label at least
//! [`CLEAN_MARGIN_MMR`] below it is dropped, and the shipped models train on what is left.
//! Evaluation labels are never touched.

use std::collections::{HashMap, HashSet};

use feature_extractor::TOTAL_SLOTS;
use skill_model::coaching::tier_index;
use skill_model::{PlaysLikeModel, TabularLayout};
use uuid::Uuid;

use crate::dataset::{Lobby, Partition, train_absolute_ensemble};
use crate::evaluation::stable_replay_hash;
use crate::gradient_boosting::GradientBoostingConfig;

/// A label this far below the player's out-of-fold plays-like prediction is dropped.
pub const CLEAN_MARGIN_MMR: f32 = 300.0;

/// Cleaning rounds before the match and window models train. A second round cost duels a
/// quarter of its within-lobby correlation.
pub const MATCH_MODEL_CLEANING_ROUNDS: usize = 1;

/// Cleaning rounds before the plays-like model trains: the second round lets it catch
/// smurfs the first round's models had learned to see as normal low-rank play.
pub const PLAYS_LIKE_CLEANING_ROUNDS: usize = 2;

/// Folds for the out-of-fold predictions.
const FOLDS: u64 = 2;

/// One player in one replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerKey {
    pub replay_id: Uuid,
    pub slot: usize,
}

/// Plays-like predictions per slot of one replay.
pub type SlotScores = [Option<f32>; TOTAL_SLOTS];

/// Plays-like predictions per replay.
#[derive(Debug, Default)]
pub struct PlaysLikeScores {
    pub by_replay: HashMap<Uuid, SlotScores>,
}

/// Labels dropped as likely smurfs.
#[derive(Debug, Default)]
pub struct DroppedLabels {
    pub players: HashSet<PlayerKey>,
}

impl DroppedLabels {
    /// Drops the same players' labels from other lobbies of their replays (the windows).
    pub fn apply(&self, lobbies: &mut [Lobby]) {
        for lobby in lobbies {
            let replay_id = lobby.replay_id;
            for (slot, target) in lobby.targets.iter_mut().enumerate() {
                if self.players.contains(&PlayerKey { replay_id, slot }) {
                    *target = 0.0;
                }
            }
        }
    }
}

/// Trains a plays-like model on `layout`'s columns and the training partition of `lobbies`.
#[must_use]
pub fn train_plays_like_model(
    lobbies: &[Lobby],
    layout: &TabularLayout,
    config: &GradientBoostingConfig,
) -> PlaysLikeModel {
    PlaysLikeModel {
        layout: layout.clone(),
        ensemble: train_absolute_ensemble(lobbies, layout, config),
    }
}

/// RMSE of a plays-like model against the labels of early-stopping players: how well it
/// estimates rank, which sets its group's weight in the smurf flag.
#[must_use]
pub fn early_stopping_rmse(lobbies: &[Lobby], model: &PlaysLikeModel) -> f64 {
    let mut squared_errors = Vec::new();
    for lobby in lobbies
        .iter()
        .filter(|lobby| lobby.partition == Partition::EarlyStopping)
    {
        let scores = model.predict_stats(&lobby.stat_references());
        for slot in lobby.labelled_slots() {
            if let Some(Some(score)) = scores.get(slot) {
                let target = lobby.targets.get(slot).copied().unwrap_or_default();
                squared_errors.push(f64::from(score - target).powi(2));
            }
        }
    }
    (squared_errors.iter().sum::<f64>() / squared_errors.len().max(1) as f64).sqrt()
}

/// Plays-like predictions for every training and early-stopping replay, each from a model
/// trained without that replay ([`FOLDS`] folds by replay hash).
#[must_use]
pub fn out_of_fold_plays_like(
    lobbies: &[Lobby],
    layout: &TabularLayout,
    config: &GradientBoostingConfig,
) -> PlaysLikeScores {
    // Independent of the early-stopping split, which uses the hash's low digits.
    let fold = |lobby: &Lobby| (stable_replay_hash(lobby.replay_id) / 10_000) % FOLDS;
    let mut scores = PlaysLikeScores::default();
    for held_out in 0..FOLDS {
        let mut subset = lobbies.to_vec();
        for lobby in &mut subset {
            if lobby.partition != Partition::Evaluation && fold(lobby) == held_out {
                lobby.partition = Partition::Evaluation;
            }
        }
        let model = train_plays_like_model(&subset, layout, config);
        for lobby in lobbies
            .iter()
            .filter(|lobby| lobby.partition != Partition::Evaluation && fold(lobby) == held_out)
        {
            scores.by_replay.insert(
                lobby.replay_id,
                model.predict_stats(&lobby.stat_references()),
            );
        }
    }
    scores
}

/// Training lobbies with likely smurf labels dropped.
pub struct CleanedLabels {
    pub lobbies: Vec<Lobby>,
    /// Every dropped label, to drop from the window lobbies too.
    pub dropped: DroppedLabels,
    pub counts: CleaningCounts,
}

/// Labels seen and dropped.
#[derive(Debug, Default)]
pub struct CleaningCounts {
    /// Labels seen per playlist size (index = players per team − 1).
    pub labelled_by_playlist: [usize; 3],
    /// Labels dropped per playlist size.
    pub dropped_by_playlist: [usize; 3],
    /// Labels dropped per label tier.
    pub dropped_by_tier: [usize; 22],
}

impl CleaningCounts {
    /// Adds another round's drops. Labels seen are kept from the first round.
    fn add(&mut self, round: &Self) {
        if self.labelled_by_playlist == [0; 3] {
            self.labelled_by_playlist = round.labelled_by_playlist;
        }
        for (total, count) in self
            .dropped_by_playlist
            .iter_mut()
            .zip(round.dropped_by_playlist)
        {
            *total += count;
        }
        for (total, count) in self.dropped_by_tier.iter_mut().zip(round.dropped_by_tier) {
            *total += count;
        }
    }

    /// Prints how many labels were dropped, per playlist and per label tier.
    pub fn print_report(&self) {
        for (name, (dropped, labelled)) in ["duels", "doubles", "standard"].iter().zip(
            self.dropped_by_playlist
                .iter()
                .zip(&self.labelled_by_playlist),
        ) {
            println!(
                "  {name}: dropped {dropped} of {labelled} training labels ({:.1}%)",
                100.0 * *dropped as f64 / (*labelled).max(1) as f64
            );
        }
        const TIER_NAMES: [&str; 22] = [
            "B1", "B2", "B3", "S1", "S2", "S3", "G1", "G2", "G3", "P1", "P2", "P3", "D1", "D2",
            "D3", "C1", "C2", "C3", "GC1", "GC2", "GC3", "SSL",
        ];
        let by_tier: Vec<String> = self
            .dropped_by_tier
            .iter()
            .zip(TIER_NAMES)
            .filter(|(count, _)| **count > 0)
            .map(|(count, name)| format!("{name} {count}"))
            .collect();
        println!("  dropped by label tier: {}", by_tier.join(", "));
    }
}

/// [`clean_labels`] repeated `rounds` times, each round scoring the labels left by the one
/// before with a fresh out-of-fold model on `layout`. Prints each round's report.
///
/// A second round helps because the first round's models learned from the very smurf
/// labels they are trying to find, which pulls their low-rank predictions down.
#[must_use]
pub fn clean_labels_in_rounds(
    lobbies: &[Lobby],
    layout: &TabularLayout,
    config: &GradientBoostingConfig,
    rounds: usize,
    margin: f32,
) -> CleanedLabels {
    let mut cleaned = CleanedLabels {
        lobbies: lobbies.to_vec(),
        dropped: DroppedLabels::default(),
        counts: CleaningCounts::default(),
    };
    for round in 1..=rounds {
        println!("  round {round}:");
        let out_of_fold = out_of_fold_plays_like(&cleaned.lobbies, layout, config);
        let round_cleaned = clean_labels(&cleaned.lobbies, &out_of_fold, margin);
        round_cleaned.counts.print_report();
        cleaned.lobbies = round_cleaned.lobbies;
        cleaned
            .dropped
            .players
            .extend(round_cleaned.dropped.players);
        cleaned.counts.add(&round_cleaned.counts);
    }
    cleaned
}

/// Drops every label at least `margin` below its out-of-fold plays-like prediction.
#[must_use]
pub fn clean_labels(
    lobbies: &[Lobby],
    out_of_fold: &PlaysLikeScores,
    margin: f32,
) -> CleanedLabels {
    clean_labels_where(lobbies, out_of_fold, |_, gap| gap >= margin)
}

/// Drops every label for which `should_drop(player, gap)` holds, `gap` being the
/// out-of-fold plays-like prediction minus the label.
#[must_use]
pub fn clean_labels_where(
    lobbies: &[Lobby],
    out_of_fold: &PlaysLikeScores,
    should_drop: impl Fn(PlayerKey, f32) -> bool,
) -> CleanedLabels {
    let mut cleaned = CleanedLabels {
        lobbies: lobbies.to_vec(),
        dropped: DroppedLabels::default(),
        counts: CleaningCounts::default(),
    };
    for lobby in &mut cleaned.lobbies {
        let Some(scores) = out_of_fold.by_replay.get(&lobby.replay_id) else {
            continue;
        };
        let playlist = lobby.players_per_team().unwrap_or(3).clamp(1, 3) - 1;
        for (slot, (target, score)) in lobby.targets.iter_mut().zip(scores).enumerate() {
            let Some(score) = score else {
                continue;
            };
            if *target <= 0.0 {
                continue;
            }
            cleaned.counts.labelled_by_playlist[playlist] += 1;
            let player = PlayerKey {
                replay_id: lobby.replay_id,
                slot,
            };
            if should_drop(player, score - *target) {
                cleaned.counts.dropped_by_playlist[playlist] += 1;
                cleaned.counts.dropped_by_tier[tier_index(*target)] += 1;
                cleaned.dropped.players.insert(player);
                *target = 0.0;
            }
        }
    }
    cleaned
}
