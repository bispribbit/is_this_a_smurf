//! Do the named-mechanics stats help the whole-match model?
//!
//! For each seed, trains the whole-match model on cleaned labels (as `train` does) twice:
//! with every stat, and without [`NAMED_MECHANICS_STATS`]. Scores both on each playlist's
//! evaluation split and prints the metrics averaged over the seeds, with the spread between
//! seeds. Duels are scored on ~800 players, so a single run's within_r moves by several
//! hundredths from the seed alone.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin compare_stat_sets -- --seeds 1,2,3

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use feature_extractor::{MATCH_STAT_COUNT, MatchStat};
use replay_structs::TeamSize;
use skill_model::{PlaysLikeModel, TabularLayout, TabularSkillModel};
use skill_model_training::dataset::{Lobby, evaluate, read_lobbies, train_model};
use skill_model_training::evaluation::{
    LobbyMetrics, MINIMUM_PLAYERS_PER_LOBBY, PlayerPrediction, compute_lobby_metrics_with_minimum,
};
use skill_model_training::gradient_boosting::GradientBoostingConfig;
use skill_model_training::label_cleaning::{
    CLEAN_MARGIN_MMR, MATCH_MODEL_CLEANING_ROUNDS, clean_labels_in_rounds,
};

/// The mechanics added after the first tree models: speed flips to bumps.
const NAMED_MECHANICS_STATS: [MatchStat; 17] = [
    MatchStat::SpeedFlipsPerMinute,
    MatchStat::HalfFlipsPerMinute,
    MatchStat::WavedashesPerMinute,
    MatchStat::FlipResetsPerMinute,
    MatchStat::CeilingTouchesPerMinute,
    MatchStat::DribbleFraction,
    MatchStat::FlicksPerMinute,
    MatchStat::AirDribbleTouchesPerMinute,
    MatchStat::DoubleTapsPerMinute,
    MatchStat::WallTouchFraction,
    MatchStat::FlipTouchFraction,
    MatchStat::PowerShotsPerMinute,
    MatchStat::FiftyFiftiesPerMinute,
    MatchStat::FiftyFiftyWinFraction,
    MatchStat::BumpsInflictedPerMinute,
    MatchStat::BumpsReceivedPerMinute,
    MatchStat::TeammateBumpsPerMinute,
];

#[derive(Parser, Debug)]
#[command(about = "Whole-match model with and without the named-mechanics stats")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

    /// Seeds for the trees' row subsampling; each runs both arms.
    #[arg(long, value_delimiter = ',', default_value = "1,2,3")]
    seeds: Vec<u64>,
}

/// One stat set under test.
struct Arm {
    name: &'static str,
    layout: TabularLayout,
    /// Metrics per playlist, one entry per seed.
    runs_by_playlist: Vec<Vec<LobbyMetrics>>,
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

fn print_runs(label: &str, runs: &[LobbyMetrics]) {
    let within_r = summarise(runs, |metrics| metrics.within_r);
    let concordance = summarise(runs, |metrics| metrics.concordance);
    let lobby_rmse = summarise(runs, |metrics| metrics.lobby_rmse);
    let player_rmse = summarise(runs, |metrics| metrics.player_rmse);
    println!(
        "    {label:<22} within_r={:.3} ±{:.3}  concordance={:.3} ±{:.3}  lobby_rmse={:.1} ±{:.1}  player_rmse={:.1} ±{:.1}",
        within_r.mean,
        within_r.spread,
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
    let all_stats = TabularLayout {
        own: true,
        deviation: true,
        lobby_mean: true,
        team_context: true,
        stats: (0..MATCH_STAT_COUNT).collect(),
    };
    let without_named_mechanics = TabularLayout {
        own: all_stats.own,
        deviation: all_stats.deviation,
        lobby_mean: all_stats.lobby_mean,
        team_context: all_stats.team_context,
        stats: MatchStat::all()
            .filter(|stat| !NAMED_MECHANICS_STATS.contains(stat))
            .map(feature_extractor::MatchStat::index)
            .collect(),
    };
    let mut arms = [
        Arm {
            name: "without named mechanics",
            layout: without_named_mechanics,
            runs_by_playlist: TeamSize::all().map(|_| Vec::new()).collect(),
        },
        Arm {
            name: "all stats",
            layout: all_stats,
            runs_by_playlist: TeamSize::all().map(|_| Vec::new()).collect(),
        },
    ];

    println!("== label cleaning");
    let cleaned = clean_labels_in_rounds(
        &lobbies,
        &PlaysLikeModel::own_stats_layout(),
        &GradientBoostingConfig::default(),
        MATCH_MODEL_CLEANING_ROUNDS,
        CLEAN_MARGIN_MMR,
    );
    let by_playlist: Vec<Vec<Lobby>> = TeamSize::all()
        .map(|playlist| {
            cleaned
                .lobbies
                .iter()
                .filter(|lobby| lobby.players_per_team() == Some(playlist.players_per_team()))
                .cloned()
                .collect()
        })
        .collect();

    for &seed in &args.seeds {
        let config = GradientBoostingConfig {
            seed,
            ..GradientBoostingConfig::default()
        };
        for arm in &mut arms {
            println!("seed {seed}: {}", arm.name);
            let model = train_model(&cleaned.lobbies, &arm.layout, &config);
            for ((playlist, playlist_lobbies), runs) in TeamSize::all()
                .zip(&by_playlist)
                .zip(&mut arm.runs_by_playlist)
            {
                runs.push(score(playlist_lobbies, &model, playlist.players_per_team()));
            }
        }
    }

    println!(
        "\nEvaluation split, mean over {} seeds (± half the seed range):",
        args.seeds.len()
    );
    for (index, playlist) in TeamSize::all().enumerate() {
        println!("  {}", playlist.name());
        for arm in &arms {
            if let Some(runs) = arm.runs_by_playlist.get(index) {
                print_runs(arm.name, runs);
            }
        }
    }
    Ok(())
}
