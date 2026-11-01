#![expect(
    clippy::indexing_slicing,
    reason = "report tables indexed by bounded rank-group and tier indices"
)]

//! Trains the shipped skill model bundle and reports how good it is.
//!
//! 1. Label cleaning: drops training labels far below an out-of-fold plays-like prediction
//!    (likely smurf accounts; see [`skill_model_training::label_cleaning`]).
//! 2. Whole-match model (cards) on `extract_stats`'s match CSV.
//! 3. Plays-like model (own stats only) and the smurf flag it feeds.
//! 4. Window model (timeline) on the window CSV, plus the timeline emphasis.
//! 5. Coaching table (the "next rank" roast) from the training players.
//! 6. Writes `data/skill_model.bin` and checks that it reloads to identical predictions.
//!
//! Every metric is computed on the database's evaluation split, which no part of training
//! sees, per playlist (duels, doubles, standard). One model serves all three: the
//! `players_per_team` stat tells the trees which playlist a lobby is. See `docs/model.md`
//! for what the numbers mean.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin train

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use feature_extractor::MATCH_STAT_COUNT;
use skill_model::coaching::{COACHING_TIPS, CoachingTable, CoachingTables, tier_index};
use skill_model::mechanics::{MECHANIC_HIGH_QUANTILE, MechanicTables, mechanics_above_lobby};
use skill_model::{
    PlaysLikeGroup, PlaysLikeGroups, PlaysLikeModel, SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR,
    STAT_GROUPS, SkillModelBundle, SmurfReason, TabularLayout, TabularSkillModel,
    plays_like_margins, smurf_margin_over_others_tiers, smurf_suspicions,
};
use skill_model_training::coaching_table::build_coaching_tables;
use skill_model_training::dataset::{
    Lobby, Partition, ScoredPlayer, evaluate, read_lobbies, train_model,
};
use skill_model_training::evaluation::{
    MINIMUM_PLAYERS_PER_LOBBY, PROXY_POSITIVE_MARGIN_MMR, PlayerPrediction,
    compute_lobby_metrics_with_minimum, flag_metrics,
};
use skill_model_training::gradient_boosting::GradientBoostingConfig;
use skill_model_training::label_cleaning::{
    CLEAN_MARGIN_MMR, CleanedLabels, MATCH_MODEL_CLEANING_ROUNDS, PLAYS_LIKE_CLEANING_ROUNDS,
    clean_labels_in_rounds, early_stopping_rmse, train_plays_like_model,
};
use skill_model_training::mechanic_table::build_mechanic_tables;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(about = "Train the skill model bundle the app embeds")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

    #[arg(long, default_value = "data/window_stats.csv")]
    window_stats: PathBuf,

    /// Window length the window CSV was extracted with.
    #[arg(long, default_value_t = 60.0)]
    window_seconds: f32,

    /// Typical timeline swing to aim for, in MMR (about one tier).
    #[arg(long, default_value_t = 70.0)]
    timeline_swing_mmr: f32,

    #[arg(long, default_value = "data/skill_model.bin")]
    out: PathBuf,

    /// Seed for the trees' row subsampling.
    #[arg(long, default_value_t = 0x5EED)]
    seed: u64,

    /// Skip training: load `--out` and only print the roast report. For iterating on the
    /// roast lines, which live in code and are not part of the trained bundle.
    #[arg(long)]
    reuse_bundle: bool,
}

/// Largest prediction change allowed between the trained model and its on-disk form.
const MAXIMUM_RELOAD_CHANGE_MMR: f32 = 0.5;

/// Rank names for the coaching report, per tier index.
const TIER_NAMES: [&str; 22] = [
    "Bronze I",
    "Bronze II",
    "Bronze III",
    "Silver I",
    "Silver II",
    "Silver III",
    "Gold I",
    "Gold II",
    "Gold III",
    "Platinum I",
    "Platinum II",
    "Platinum III",
    "Diamond I",
    "Diamond II",
    "Diamond III",
    "Champion I",
    "Champion II",
    "Champion III",
    "Grand Champion I",
    "Grand Champion II",
    "Grand Champion III",
    "Supersonic Legend",
];

/// Playlists by players per team, with their report name.
const PLAYLISTS: [Playlist; 3] = [
    Playlist {
        players_per_team: 1,
        name: "duels",
    },
    Playlist {
        players_per_team: 2,
        name: "doubles",
    },
    Playlist {
        players_per_team: 3,
        name: "standard",
    },
];

/// One playlist size and its report name.
struct Playlist {
    players_per_team: usize,
    name: &'static str,
}

impl Playlist {
    /// Within-lobby metrics need this many rank-known players: the full lobby for duels and
    /// doubles, [`MINIMUM_PLAYERS_PER_LOBBY`] for standard.
    fn minimum_players_per_lobby(&self) -> usize {
        MINIMUM_PLAYERS_PER_LOBBY.min(self.players_per_team * 2)
    }
}

/// Metrics of each playlist present in `lobbies`.
fn print_playlist_metrics(lobbies: &[Lobby], model: &skill_model::TabularSkillModel) {
    for playlist in &PLAYLISTS {
        let playlist_lobbies: Vec<Lobby> = lobbies
            .iter()
            .filter(|lobby| lobby.players_per_team() == Some(playlist.players_per_team))
            .cloned()
            .collect();
        let predictions: Vec<PlayerPrediction> = evaluate(&playlist_lobbies, model)
            .iter()
            .map(|scored| scored.prediction)
            .collect();
        if predictions.is_empty() {
            continue;
        }
        print_metrics(
            &format!("evaluation {}", playlist.name),
            &predictions,
            playlist.minimum_players_per_lobby(),
        );
    }
}

/// The shipped smurf flag on each playlist's evaluation split: how often it fires, why, and
/// how many label-based party positives (ranked ≥ 150 over their lobby) it confirms.
fn print_smurf_report(
    lobbies: &[Lobby],
    match_model: &TabularSkillModel,
    plays_like_groups: &PlaysLikeGroups,
    mechanic_tables: &MechanicTables,
) {
    for playlist in &PLAYLISTS {
        let mut margins: Vec<f32> = Vec::new();
        let mut players = 0usize;
        let mut ranked_above = 0usize;
        let mut plays_above = 0usize;
        let mut mechanics_above = 0usize;
        let mut deranking = 0usize;
        let mut positives = 0usize;
        let mut flagged_positives = 0usize;
        for lobby in lobbies.iter().filter(|lobby| {
            lobby.partition == Partition::Evaluation
                && lobby.players_per_team() == Some(playlist.players_per_team)
        }) {
            let stats = lobby.stat_references();
            let match_mmr = match_model.predict_stats(&stats);
            let plays_like = plays_like_groups.predict_stats(&stats);
            margins.extend(plays_like_margins(&match_mmr, &plays_like).iter().flatten());
            let mechanics = mechanics_above_lobby(
                mechanic_tables.for_players_per_team(playlist.players_per_team),
                &stats,
                &match_mmr,
            );
            let flags = smurf_suspicions(
                &match_mmr,
                &plays_like,
                &skill_model::SlotEvidence::per_slot(&mechanics, &stats),
                playlist.players_per_team,
            );
            let mut labels: Vec<f32> = lobby
                .labelled_slots()
                .map(|slot| lobby.targets[slot])
                .collect();
            labels.sort_by(f32::total_cmp);
            let label_median = (labels.len() >= 2).then(|| {
                let middle = labels.len() / 2;
                if labels.len().is_multiple_of(2) {
                    f32::midpoint(labels[middle - 1], labels[middle])
                } else {
                    labels[middle]
                }
            });
            for (slot, flag) in flags.iter().enumerate() {
                if lobby.stats[slot].is_none() {
                    continue;
                }
                players += 1;
                match flag {
                    Some(SmurfReason::RankedAboveLobby) => ranked_above += 1,
                    Some(SmurfReason::PlaysAboveLobby) => plays_above += 1,
                    Some(SmurfReason::MechanicsAboveLobby) => mechanics_above += 1,
                    Some(SmurfReason::DerankingAboveLobby) => deranking += 1,
                    None => {}
                }
                let positive = lobby.targets[slot] > 0.0
                    && label_median.is_some_and(|median| {
                        lobby.targets[slot] - median >= PROXY_POSITIVE_MARGIN_MMR
                    });
                positives += usize::from(positive);
                flagged_positives += usize::from(positive && flag.is_some());
            }
        }
        if players == 0 {
            continue;
        }
        println!(
            "  {:<9} flags {:.2}% of players ({:.2}% ranked above the lobby, {:.2}% play above it, {:.2}% mechanics above it, {:.2}% deranking)  party positives caught {flagged_positives}/{positives}",
            playlist.name,
            100.0 * (ranked_above + plays_above + mechanics_above + deranking) as f64
                / players as f64,
            100.0 * ranked_above as f64 / players as f64,
            100.0 * plays_above as f64 / players as f64,
            100.0 * mechanics_above as f64 / players as f64,
            100.0 * deranking as f64 / players as f64,
        );
        let rates: Vec<String> = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
            .iter()
            .map(|threshold| {
                let above = margins
                    .iter()
                    .filter(|margin| **margin > *threshold)
                    .count();
                format!(
                    "+{threshold:.1}: {:.1}%",
                    100.0 * above as f64 / margins.len().max(1) as f64
                )
            })
            .collect();
        println!(
            "            plays-above rate by weighted margin (shipped +{:.1}): {}",
            smurf_margin_over_others_tiers(playlist.players_per_team),
            rates.join("  ")
        );
    }
}

fn print_metrics(label: &str, predictions: &[PlayerPrediction], minimum_players_per_lobby: usize) {
    let metrics = compute_lobby_metrics_with_minimum(predictions, minimum_players_per_lobby);
    println!(
        "  {label:<22} within_r={:.3} ±{:.3}  lobby_rmse={:.0}  player_rmse={:.0}  concordance={:.3}  players={}",
        metrics.within_r,
        metrics.within_r_standard_error,
        metrics.lobby_rmse,
        metrics.player_rmse,
        metrics.concordance,
        metrics.player_count,
    );
}

/// Typical size of a player's window-to-window form swing, in MMR (RMS of each window's
/// deviation around the player's own time-weighted mean deviation).
fn typical_form_swing(windows: &[ScoredPlayer]) -> f32 {
    let mut by_player: HashMap<PlayerKey, Vec<ScoredPlayer>> = HashMap::new();
    for window in windows {
        by_player
            .entry(PlayerKey {
                replay_id: window.prediction.replay_id,
                slot: window.prediction.slot,
            })
            .or_default()
            .push(*window);
    }
    let mut square_sum = 0.0f64;
    let mut count = 0.0f64;
    for player_windows in by_player.values().filter(|w| w.len() >= 2) {
        let seconds: f32 = player_windows.iter().map(|w| w.live_seconds).sum();
        let mean = player_windows
            .iter()
            .map(|w| w.deviation * w.live_seconds)
            .sum::<f32>()
            / seconds.max(1e-6);
        for window in player_windows {
            square_sum += f64::from(window.deviation - mean).powi(2);
            count += 1.0;
        }
    }
    (square_sum / count.max(1.0)).sqrt() as f32
}

/// Identifies one player in one replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PlayerKey {
    replay_id: Uuid,
    slot: usize,
}

/// Which tips the roast picks on the evaluation split, per predicted rank group.
fn print_coaching_report(lobbies: &[Lobby], bundle: &SkillModelBundle) {
    let mut counts: Vec<HashMap<&'static str, usize>> = vec![HashMap::new(); 8];
    let mut examples: Vec<Option<String>> = vec![None; 8];
    for lobby in lobbies
        .iter()
        .filter(|l| l.partition == Partition::Evaluation)
    {
        let Some(coaching) = lobby
            .players_per_team()
            .and_then(|players_per_team| bundle.coaching.for_players_per_team(players_per_team))
        else {
            continue;
        };
        let predictions = bundle.match_model.predict_stats(&lobby.stat_references());
        for (slot, prediction) in predictions.iter().enumerate() {
            let Some(prediction) = prediction else {
                continue;
            };
            let Some(Some(stats)) = lobby.stats.get(slot) else {
                continue;
            };
            let Some(advice) = coaching.advise(*prediction, stats) else {
                continue;
            };
            let group = (tier_index(*prediction) / 3).min(7);
            let key = advice.tip.map_or("(nothing to fix)", |tip| tip.stat);
            *counts[group].entry(key).or_default() += 1;
            if examples[group].is_none() && advice.tip.is_some() {
                examples[group] = Some(advice.line(slot, "Player"));
            }
        }
    }
    let groups = [
        "Bronze", "Silver", "Gold", "Platinum", "Diamond", "Champion", "GC", "SSL",
    ];
    for (group, name) in groups.iter().enumerate() {
        let total: usize = counts[group].values().sum();
        if total == 0 {
            continue;
        }
        let mut ranked: Vec<&str> = counts[group].keys().copied().collect();
        ranked
            .sort_by_key(|stat| core::cmp::Reverse(counts[group].get(stat).copied().unwrap_or(0)));
        let top: Vec<String> = ranked
            .iter()
            .take(4)
            .map(|stat| {
                let count = counts[group].get(stat).copied().unwrap_or(0);
                format!("{stat} {:.0}%", 100.0 * count as f64 / total as f64)
            })
            .collect();
        println!("  {name:<9} n={total:<5} {}", top.join(", "));
        if let Some(example) = &examples[group] {
            println!("            e.g. \"{example}\"");
        }
    }
}

fn print_all_enabled_tips(coaching: &CoachingTables) {
    for (playlist, table) in PLAYLISTS.iter().zip(&coaching.by_players_per_team) {
        if table.tiers.is_empty() {
            println!("  {}: no training data", playlist.name);
            continue;
        }
        println!("  {}:", playlist.name);
        print_enabled_tips(table);
    }
}

fn print_enabled_tips(coaching: &CoachingTable) {
    let disabled: Vec<&str> = COACHING_TIPS
        .iter()
        .zip(&coaching.enabled)
        .filter(|(_, enabled)| !**enabled)
        .map(|(tip, _)| tip.stat)
        .collect();
    if disabled.is_empty() {
        println!("  all {} tips agree with the data", COACHING_TIPS.len());
    } else {
        println!("  tips disabled because the data disagrees with their direction: {disabled:?}");
    }
    if let (Some(bronze), Some(champion)) = (coaching.tiers.first(), coaching.tiers.get(15)) {
        for (tip_index, tip) in COACHING_TIPS.iter().enumerate().take(6) {
            println!(
                "    {:<36} {} median {:.3} → {} median {:.3}",
                tip.stat,
                TIER_NAMES[0],
                bronze.medians.get(tip_index).copied().unwrap_or(0.0),
                TIER_NAMES[15],
                champion.medians.get(tip_index).copied().unwrap_or(0.0),
            );
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = GradientBoostingConfig {
        seed: args.seed,
        ..GradientBoostingConfig::default()
    };
    let layout = TabularLayout {
        own: true,
        deviation: true,
        lobby_mean: true,
        team_context: true,
        stats: (0..MATCH_STAT_COUNT).collect(),
    };

    if args.reuse_bundle {
        let bundle = SkillModelBundle::from_bytes(
            &std::fs::read(&args.out).with_context(|| format!("reading {}", args.out.display()))?,
        )
        .map_err(|error| anyhow::anyhow!("decoding bundle: {error}"))?;
        let match_lobbies = read_lobbies(&args.match_stats)?;
        println!("== roast report for {} (no retraining)", args.out.display());
        print_coaching_report(&match_lobbies, &bundle);
        return Ok(());
    }

    println!(
        "== label cleaning (labels ≥ {CLEAN_MARGIN_MMR} MMR below an out-of-fold plays-like prediction)"
    );
    let raw_match_lobbies = read_lobbies(&args.match_stats)?;
    // Cleaning uses every stat (the better rank estimate); the shipped plays-like model
    // sees only opponent-independent stats (the better smurf flag). See docs/model.md.
    let CleanedLabels {
        lobbies: match_lobbies,
        dropped,
        counts,
    } = clean_labels_in_rounds(
        &raw_match_lobbies,
        &PlaysLikeModel::own_stats_layout(),
        &config,
        MATCH_MODEL_CLEANING_ROUNDS,
        CLEAN_MARGIN_MMR,
    );
    drop(raw_match_lobbies);
    if MATCH_MODEL_CLEANING_ROUNDS > 1 {
        println!("  all rounds:");
        counts.print_report();
    }

    println!("== whole-match model ({})", args.match_stats.display());
    let match_model = train_model(&match_lobbies, &layout, &config);
    let match_scored = evaluate(&match_lobbies, &match_model);
    let match_predictions: Vec<PlayerPrediction> =
        match_scored.iter().map(|s| s.prediction).collect();
    print_playlist_metrics(&match_lobbies, &match_model);
    println!(
        "  ranked-above-the-lobby rule (prediction > lobby median + margin) vs players ranked ≥150 over their lobby:"
    );
    for margin in [50.0, 75.0, 100.0, 150.0, 200.0] {
        let flags = flag_metrics(&match_predictions, margin);
        let shipped = if (margin - SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR).abs() < f32::EPSILON {
            "  ← shipped"
        } else {
            ""
        };
        println!(
            "    +{margin:<4.0} flags {:.2}% of players  precision {:.0}%  recall {:.0}%{shipped}",
            100.0 * flags.flag_rate(),
            100.0 * flags.precision(),
            100.0 * flags.recall(),
        );
    }

    println!("== plays-like model and smurf flag (evaluation split)");
    println!("  further cleaning for the plays-like model:");
    let plays_like_lobbies = clean_labels_in_rounds(
        &match_lobbies,
        &PlaysLikeModel::own_stats_layout(),
        &config,
        PLAYS_LIKE_CLEANING_ROUNDS - MATCH_MODEL_CLEANING_ROUNDS,
        CLEAN_MARGIN_MMR,
    )
    .lobbies;
    let mut plays_like_groups = PlaysLikeGroups::default();
    for group in &STAT_GROUPS {
        let model = train_plays_like_model(&plays_like_lobbies, &group.layout(), &config);
        let rmse = early_stopping_rmse(&plays_like_lobbies, &model);
        println!(
            "  {:<16} {} stats  early-stopping rmse={rmse:.0}",
            group.name,
            group.stats.len()
        );
        plays_like_groups.groups.push(PlaysLikeGroup {
            name: group.name.to_owned(),
            weight: (1.0 / (rmse * rmse)) as f32,
            model,
        });
    }
    let mechanics = build_mechanic_tables(&match_lobbies, MECHANIC_HIGH_QUANTILE);
    print_smurf_report(&match_lobbies, &match_model, &plays_like_groups, &mechanics);

    println!(
        "== window model ({}, {} s windows)",
        args.window_stats.display(),
        args.window_seconds
    );
    let mut window_lobbies = read_lobbies(&args.window_stats)?;
    dropped.apply(&mut window_lobbies);
    let window_model = train_model(&window_lobbies, &layout, &config);
    let window_scored = evaluate(&window_lobbies, &window_model);
    let swing = typical_form_swing(&window_scored);
    let timeline_emphasis = args.timeline_swing_mmr / swing.max(1.0);
    println!(
        "  typical form swing {swing:.1} MMR → timeline emphasis ×{timeline_emphasis:.1} (aims for ±{:.0} MMR)",
        args.timeline_swing_mmr
    );

    println!("== coaching table");
    let coaching = build_coaching_tables(&match_lobbies);
    print_all_enabled_tips(&coaching);

    let bundle = SkillModelBundle {
        match_model,
        window_model,
        plays_like_groups,
        window_seconds: args.window_seconds,
        timeline_emphasis,
        coaching,
        mechanics,
    };
    println!("  advice picked on the evaluation split, by predicted rank:");
    print_coaching_report(&match_lobbies, &bundle);

    let bytes = bundle
        .to_bytes()
        .map_err(|error| anyhow::anyhow!("encoding bundle: {error}"))?;
    std::fs::write(&args.out, &bytes).with_context(|| format!("writing {}", args.out.display()))?;
    let reloaded = SkillModelBundle::from_bytes(&std::fs::read(&args.out)?)
        .map_err(|error| anyhow::anyhow!("reloading bundle: {error}"))?;
    let reloaded_predictions: Vec<PlayerPrediction> =
        evaluate(&match_lobbies, &reloaded.match_model)
            .iter()
            .map(|s| s.prediction)
            .collect();
    // Leaf values are quantised to 16 bits on disk, so predictions move by a hair; anything
    // visible means the encoding is broken.
    let largest_plays_like_change = match_lobbies
        .iter()
        .filter(|lobby| lobby.partition == Partition::Evaluation)
        .flat_map(|lobby| {
            let stats = lobby.stat_references();
            let before = bundle
                .plays_like_groups
                .predict_stats(&stats)
                .combined_mmr();
            let after = reloaded
                .plays_like_groups
                .predict_stats(&stats)
                .combined_mmr();
            before
                .into_iter()
                .zip(after)
                .filter_map(|(before, after)| Some((before? - after?).abs()))
                .collect::<Vec<f32>>()
        })
        .fold(0.0_f32, f32::max);
    let largest_change = reloaded_predictions
        .iter()
        .zip(&match_predictions)
        .map(|(a, b)| (a.prediction_mmr - b.prediction_mmr).abs())
        .fold(largest_plays_like_change, f32::max);
    anyhow::ensure!(
        largest_change < MAXIMUM_RELOAD_CHANGE_MMR,
        "reloaded bundle predicts differently (by up to {largest_change:.3} MMR)"
    );
    print_metrics(
        "reloaded from disk",
        &reloaded_predictions,
        MINIMUM_PLAYERS_PER_LOBBY,
    );
    println!(
        "== wrote {} ({:.2} MB), reload verified (largest prediction change {largest_change:.4} MMR)",
        args.out.display(),
        bytes.len() as f64 / 1_048_576.0
    );
    Ok(())
}
