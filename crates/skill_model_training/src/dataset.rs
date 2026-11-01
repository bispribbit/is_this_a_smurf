#![expect(
    clippy::indexing_slicing,
    reason = "fixed-size stat and slot arrays indexed by bounded layout indices"
)]

//! Loading per-player stats and turning them into training rows.
//!
//! A *lobby* is one replay (whole-match stats) or one window of a replay (window stats).
//! Lobbies are split by a stable hash of the replay id, so every window of a replay lands in
//! the same partition and the split never changes between runs.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use feature_extractor::{MATCH_STAT_COUNT, MatchStat, TOTAL_SLOTS};
use skill_model::{
    FrozenEnsemble, LobbySummary, MINIMUM_MATCH_SECONDS, MINIMUM_WINDOW_SECONDS, TabularLayout,
    TabularSkillModel,
};
use uuid::Uuid;

use crate::evaluation::{PlayerPrediction, stable_replay_hash};
use crate::gradient_boosting::{
    Dataset, FeatureMatrix, GradientBoostingConfig, GradientBoostingModel,
};

/// Share of training replays held out for early stopping and calibration.
pub const EARLY_STOPPING_FRACTION: f64 = 0.1;

/// Index of the first stat column in a stats CSV.
const FIRST_STAT_COLUMN: usize = 5;

/// Which part of the data a lobby belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Partition {
    /// Fits the trees.
    Training,
    /// Stops tree growth and fits the deviation scale.
    EarlyStopping,
    /// The database's evaluation split; only ever scored.
    Evaluation,
}

/// One replay, or one window of a replay.
#[derive(Debug, Clone)]
pub struct Lobby {
    pub replay_id: Uuid,
    /// Window index within the replay; `0` for whole-match rows.
    pub window: usize,
    pub partition: Partition,
    /// Stats per slot; `None` when the slot is empty (or too short in a window).
    pub stats: [Option<[f32; MATCH_STAT_COUNT]>; TOTAL_SLOTS],
    /// Label MMR per slot; `0` when the rank is unknown.
    pub targets: [f32; TOTAL_SLOTS],
    /// Live seconds per slot.
    pub live_seconds: [f32; TOTAL_SLOTS],
}

/// Identifies one lobby while loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct LobbyKey {
    replay_id: Uuid,
    window: usize,
}

impl Lobby {
    /// Stats per slot, by reference, as the model takes them.
    #[must_use]
    pub fn stat_references(&self) -> [Option<&[f32; MATCH_STAT_COUNT]>; TOTAL_SLOTS] {
        core::array::from_fn(|slot| self.stats[slot].as_ref())
    }

    /// Slots with stats and a known label.
    pub fn labelled_slots(&self) -> impl Iterator<Item = usize> + '_ {
        (0..TOTAL_SLOTS).filter(|&slot| self.stats[slot].is_some() && self.targets[slot] > 0.0)
    }

    /// Playlist size (1 duels, 2 doubles, 3 standard), read from the stat every player
    /// carries; `None` for a lobby without players.
    #[must_use]
    pub fn players_per_team(&self) -> Option<usize> {
        self.stats
            .iter()
            .flatten()
            .next()
            .map(|stats| stats[MatchStat::PlayersPerTeam.index()].round() as usize)
    }

    /// Mean label over labelled slots.
    #[must_use]
    pub fn mean_label(&self) -> Option<f32> {
        let labels: Vec<f32> = self
            .labelled_slots()
            .map(|slot| self.targets[slot])
            .collect();
        (!labels.is_empty()).then(|| labels.iter().sum::<f32>() / labels.len() as f32)
    }
}

/// Reads a stats CSV written by `extract_stats`.
///
/// A trailing `window_index` column marks window rows. Players with less than
/// [`MINIMUM_WINDOW_SECONDS`] in a window, or less than [`MINIMUM_MATCH_SECONDS`] in a whole
/// match, are dropped (as the app does).
///
/// # Errors
///
/// When the file is missing or its stat columns differ from this build's stats.
pub fn read_lobbies(path: &Path) -> Result<Vec<Lobby>> {
    let mut reader =
        csv::Reader::from_path(path).with_context(|| format!("opening {}", path.display()))?;
    let headers = reader.headers()?.clone();
    for (index, stat) in MatchStat::all().enumerate() {
        let name = stat.name();
        anyhow::ensure!(
            headers.get(FIRST_STAT_COLUMN + index) == Some(name),
            "{}: column {} should be {name} — re-run extract_stats",
            path.display(),
            FIRST_STAT_COLUMN + index,
        );
    }
    let window_column = headers.iter().position(|header| header == "window_index");

    let mut lobbies: HashMap<LobbyKey, Lobby> = HashMap::new();
    for record in reader.records() {
        let record = record?;
        let field = |index: usize| record.get(index).unwrap_or_default();
        let replay_id = Uuid::parse_str(field(0))?;
        let slot: usize = field(2).parse()?;
        let live_seconds: f32 = field(4).parse()?;
        let window = match window_column {
            Some(column) => field(column).parse()?,
            None => 0,
        };
        let minimum_seconds = if window_column.is_some() {
            MINIMUM_WINDOW_SECONDS
        } else {
            MINIMUM_MATCH_SECONDS
        };
        if slot >= TOTAL_SLOTS || live_seconds < minimum_seconds {
            continue;
        }
        let mut stats = [0.0; MATCH_STAT_COUNT];
        for (index, value) in stats.iter_mut().enumerate() {
            *value = field(FIRST_STAT_COLUMN + index).parse()?;
        }
        let partition = if field(1) == "evaluation" {
            Partition::Evaluation
        } else if (stable_replay_hash(replay_id) % 10_000) as f64 / 10_000.0
            < EARLY_STOPPING_FRACTION
        {
            Partition::EarlyStopping
        } else {
            Partition::Training
        };
        let lobby = lobbies
            .entry(LobbyKey { replay_id, window })
            .or_insert_with(|| Lobby {
                replay_id,
                window,
                partition,
                stats: [None; TOTAL_SLOTS],
                targets: [0.0; TOTAL_SLOTS],
                live_seconds: [0.0; TOTAL_SLOTS],
            });
        lobby.stats[slot] = Some(stats);
        lobby.targets[slot] = field(3).parse()?;
        lobby.live_seconds[slot] = live_seconds;
    }

    let mut lobbies: Vec<Lobby> = lobbies.into_values().collect();
    lobbies.sort_by_key(|lobby| LobbyKey {
        replay_id: lobby.replay_id,
        window: lobby.window,
    });
    Ok(lobbies)
}

/// Training rows for one ensemble.
struct TrainingRows {
    features: FeatureMatrix,
    targets: Vec<f32>,
}

/// Rows of one partition. `deviation_target` trains on label − lobby mean label, which
/// needs at least two labelled players.
fn rows(
    lobbies: &[Lobby],
    partition: Partition,
    layout: &TabularLayout,
    deviation_target: bool,
) -> TrainingRows {
    let mut values = Vec::new();
    let mut targets = Vec::new();
    let mut columns = 0;
    for lobby in lobbies.iter().filter(|l| l.partition == partition) {
        let Some(mean_label) = lobby.mean_label() else {
            continue;
        };
        if deviation_target && lobby.labelled_slots().count() < 2 {
            continue;
        }
        let summary = LobbySummary::new(&lobby.stat_references());
        for slot in lobby.labelled_slots() {
            let Some(stats) = lobby.stats[slot].as_ref() else {
                continue;
            };
            let before = values.len();
            layout.push_player_features(&summary, slot, stats, &mut values);
            columns = values.len() - before;
            targets.push(if deviation_target {
                lobby.targets[slot] - mean_label
            } else {
                lobby.targets[slot]
            });
        }
    }
    TrainingRows {
        features: FeatureMatrix { values, columns },
        targets,
    }
}

fn fit(
    lobbies: &[Lobby],
    layout: &TabularLayout,
    deviation_target: bool,
    config: &GradientBoostingConfig,
) -> GradientBoostingModel {
    let training = rows(lobbies, Partition::Training, layout, deviation_target);
    let early = rows(lobbies, Partition::EarlyStopping, layout, deviation_target);
    GradientBoostingModel::fit(
        &Dataset {
            features: &training.features,
            targets: &training.targets,
        },
        Some(&Dataset {
            features: &early.features,
            targets: &early.targets,
        }),
        config,
    )
}

/// Trains one ensemble predicting each player's label from `layout`'s features.
#[must_use]
pub fn train_absolute_ensemble(
    lobbies: &[Lobby],
    layout: &TabularLayout,
    config: &GradientBoostingConfig,
) -> FrozenEnsemble {
    fit(lobbies, layout, false, config).freeze()
}

/// OLS slope of true on predicted within-lobby deviation over the early-stopping lobbies,
/// using the model's own centred deviations exactly as it produces them.
fn fit_deviation_scale(lobbies: &[Lobby], model: &TabularSkillModel) -> f32 {
    let unscaled = TabularSkillModel {
        deviation_scale: 1.0,
        ..model.clone()
    };
    let mut covariance = 0.0f64;
    let mut variance = 0.0f64;
    for lobby in lobbies
        .iter()
        .filter(|l| l.partition == Partition::EarlyStopping)
    {
        let labelled: Vec<usize> = lobby.labelled_slots().collect();
        if labelled.len() < 2 {
            continue;
        }
        let predictions = unscaled.predict_stats(&lobby.stat_references());
        let count = labelled.len() as f64;
        let mean_prediction = labelled
            .iter()
            .filter_map(|&slot| predictions[slot])
            .map(f64::from)
            .sum::<f64>()
            / count;
        let mean_label = labelled
            .iter()
            .map(|&slot| f64::from(lobby.targets[slot]))
            .sum::<f64>()
            / count;
        for &slot in &labelled {
            let Some(prediction) = predictions[slot] else {
                continue;
            };
            let predicted = f64::from(prediction) - mean_prediction;
            covariance = predicted.mul_add(f64::from(lobby.targets[slot]) - mean_label, covariance);
            variance = predicted.mul_add(predicted, variance);
        }
    }
    if variance > 0.0 {
        (covariance / variance) as f32
    } else {
        1.0
    }
}

/// Trains the two ensembles on `lobbies` and fits the deviation scale.
#[must_use]
pub fn train_model(
    lobbies: &[Lobby],
    layout: &TabularLayout,
    config: &GradientBoostingConfig,
) -> TabularSkillModel {
    let within_layout = TabularLayout {
        own: false,
        ..layout.clone()
    };
    let absolute = fit(lobbies, layout, false, config);
    let within = fit(lobbies, &within_layout, true, config);
    let mut model = TabularSkillModel {
        stat_names: MatchStat::all().map(|stat| stat.to_string()).collect(),
        absolute_layout: layout.clone(),
        absolute: absolute.freeze(),
        within_layout,
        within: within.freeze(),
        deviation_scale: 1.0,
    };
    model.deviation_scale = fit_deviation_scale(lobbies, &model);
    model
}

/// One player's prediction in one evaluation lobby, with what the timeline needs.
#[derive(Debug, Clone, Copy)]
pub struct ScoredPlayer {
    pub prediction: PlayerPrediction,
    pub window: usize,
    pub live_seconds: f32,
    /// Prediction minus the lobby's mean prediction over present players.
    pub deviation: f32,
}

/// Scores every labelled player of the evaluation lobbies.
#[must_use]
pub fn evaluate(lobbies: &[Lobby], model: &TabularSkillModel) -> Vec<ScoredPlayer> {
    let mut scored = Vec::new();
    for lobby in lobbies
        .iter()
        .filter(|l| l.partition == Partition::Evaluation)
    {
        let predictions = model.predict_stats(&lobby.stat_references());
        let present: Vec<f32> = predictions.iter().flatten().copied().collect();
        let mean = present.iter().sum::<f32>() / present.len().max(1) as f32;
        for slot in lobby.labelled_slots() {
            if let Some(prediction) = predictions[slot] {
                scored.push(ScoredPlayer {
                    prediction: PlayerPrediction {
                        replay_id: lobby.replay_id,
                        slot,
                        prediction_mmr: prediction,
                        target_mmr: lobby.targets[slot],
                    },
                    window: lobby.window,
                    live_seconds: lobby.live_seconds[slot],
                    deviation: prediction - mean,
                });
            }
        }
    }
    scored
}
