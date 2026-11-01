//! One model for every playlist, or one model per playlist?
//!
//! For each seed, trains the whole-match model once on every playlist (what `train` ships)
//! and once per playlist on that playlist alone, then scores both on each playlist's
//! evaluation split. Prints each arm's metrics averaged over the seeds, with the spread
//! between seeds.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin compare_playlist_models -- --seeds 1,2,3

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use feature_extractor::MATCH_STAT_COUNT;
use replay_structs::TeamSize;
use skill_model::{TabularLayout, TabularSkillModel};
use skill_model_training::dataset::{Lobby, evaluate, read_lobbies, train_model};
use skill_model_training::evaluation::{
    LobbyMetrics, MINIMUM_PLAYERS_PER_LOBBY, PlayerPrediction, compute_lobby_metrics_with_minimum,
};
use skill_model_training::gradient_boosting::GradientBoostingConfig;

#[derive(Parser, Debug)]
#[command(about = "Compare a joint model against one model per playlist")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

    /// Seeds for the trees' row subsampling; each runs every arm.
    #[arg(long, value_delimiter = ',', default_value = "1,2,3")]
    seeds: Vec<u64>,
}

/// Metrics of one arm on one playlist, one entry per seed.
#[derive(Default)]
struct ArmResults {
    runs: Vec<LobbyMetrics>,
}

/// Both arms on one playlist.
#[derive(Default)]
struct PlaylistResults {
    joint: ArmResults,
    separate: ArmResults,
}

fn playlist_lobbies(lobbies: &[Lobby], players_per_team: usize) -> Vec<Lobby> {
    lobbies
        .iter()
        .filter(|lobby| lobby.players_per_team() == Some(players_per_team))
        .cloned()
        .collect()
}

fn score(lobbies: &[Lobby], model: &TabularSkillModel, players_per_team: usize) -> LobbyMetrics {
    let predictions: Vec<PlayerPrediction> = evaluate(lobbies, model)
        .iter()
        .map(|scored| scored.prediction)
        .collect();
    compute_lobby_metrics_with_minimum(
        &predictions,
        MINIMUM_PLAYERS_PER_LOBBY.min(players_per_team * 2),
    )
}

/// Mean and half the range over seeds.
struct Summary {
    mean: f64,
    spread: f64,
}

fn summarise(runs: &[LobbyMetrics], metric: fn(&LobbyMetrics) -> f64) -> Summary {
    let values: Vec<f64> = runs.iter().map(metric).collect();
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Summary {
        mean: values.iter().sum::<f64>() / values.len().max(1) as f64,
        spread: (maximum - minimum) / 2.0,
    }
}

fn print_arm(label: &str, arm: &ArmResults) {
    let within_r = summarise(&arm.runs, |metrics| metrics.within_r);
    let concordance = summarise(&arm.runs, |metrics| metrics.concordance);
    let lobby_rmse = summarise(&arm.runs, |metrics| metrics.lobby_rmse);
    let player_rmse = summarise(&arm.runs, |metrics| metrics.player_rmse);
    let standard_error = summarise(&arm.runs, |metrics| metrics.within_r_standard_error);
    println!(
        "    {label:<9} within_r={:.3} ±{:.3} (bootstrap ±{:.3})  concordance={:.3} ±{:.3}  lobby_rmse={:.1} ±{:.1}  player_rmse={:.1} ±{:.1}",
        within_r.mean,
        within_r.spread,
        standard_error.mean,
        concordance.mean,
        concordance.spread,
        lobby_rmse.mean,
        lobby_rmse.spread,
        player_rmse.mean,
        player_rmse.spread,
    );
}

fn main() -> Result<()> {
    let args = Args::parse();
    let lobbies = read_lobbies(&args.match_stats)?;
    let layout = TabularLayout {
        own: true,
        deviation: true,
        lobby_mean: true,
        team_context: true,
        stats: (0..MATCH_STAT_COUNT).collect(),
    };
    let by_playlist: Vec<Vec<Lobby>> = TeamSize::all()
        .map(|playlist| playlist_lobbies(&lobbies, playlist.players_per_team()))
        .collect();
    for (playlist, playlist_lobbies) in TeamSize::all().zip(&by_playlist) {
        println!("{}: {} replays", playlist.name(), playlist_lobbies.len());
    }

    let mut results: Vec<PlaylistResults> = TeamSize::all()
        .map(|_| PlaylistResults::default())
        .collect();
    for &seed in &args.seeds {
        let config = GradientBoostingConfig {
            seed,
            ..GradientBoostingConfig::default()
        };
        println!("seed {seed}: joint model");
        let joint = train_model(&lobbies, &layout, &config);
        for ((playlist, playlist_lobbies), result) in
            TeamSize::all().zip(&by_playlist).zip(&mut results)
        {
            if playlist_lobbies.is_empty() {
                continue;
            }
            println!("seed {seed}: {} model", playlist.name());
            let separate = train_model(playlist_lobbies, &layout, &config);
            result
                .joint
                .runs
                .push(score(playlist_lobbies, &joint, playlist.players_per_team()));
            result.separate.runs.push(score(
                playlist_lobbies,
                &separate,
                playlist.players_per_team(),
            ));
        }
    }

    println!(
        "\nEvaluation split, mean over {} seeds (± half the seed range):",
        args.seeds.len()
    );
    for (playlist, result) in TeamSize::all().zip(&results) {
        let Some(first) = result.joint.runs.first() else {
            continue;
        };
        println!(
            "  {} ({} lobbies, {} players)",
            playlist.name(),
            first.lobby_count,
            first.player_count
        );
        print_arm("joint", &result.joint);
        print_arm("separate", &result.separate);
    }
    Ok(())
}
