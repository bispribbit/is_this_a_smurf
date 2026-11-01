//! Per-game or per-account label cleaning: which trains the better whole-match model?
//!
//! Scores every training player out-of-fold once, cleans the labels per game (what `train`
//! does) and per account ([`skill_model_training::account_cleaning`]), then trains the
//! whole-match model on each, over several seeds, and scores every playlist's evaluation
//! split. Also reports how the account rule differs (labels it drops that the per-game rule
//! kept, and the reverse), and whether evaluation players whose account also appears in
//! training are predicted better than unseen ones (leakage through the uploader's style).
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin compare_cleaning -- --seeds 1,2,3

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use feature_extractor::MATCH_STAT_COUNT;
use replay_structs::TeamSize;
use skill_model::{PlaysLikeModel, TabularLayout, TabularSkillModel};
use skill_model_training::account_cleaning::{
    AccountCleaning, AccountId, AccountIds, clean_labels_by_account,
};
use skill_model_training::dataset::{Lobby, Partition, evaluate, read_lobbies, train_model};
use skill_model_training::evaluation::{
    LobbyMetrics, MINIMUM_PLAYERS_PER_LOBBY, PlayerPrediction, compute_lobby_metrics_with_minimum,
};
use skill_model_training::gradient_boosting::GradientBoostingConfig;
use skill_model_training::label_cleaning::{
    CLEAN_MARGIN_MMR, CleanedLabels, PlayerKey, clean_labels, out_of_fold_plays_like,
};

#[derive(Parser, Debug)]
#[command(about = "Compare per-game and per-account label cleaning")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

    #[arg(long, default_value = "data/accounts.csv")]
    accounts: PathBuf,

    /// Seeds for the trees' row subsampling; each trains every variant.
    #[arg(long, value_delimiter = ',', default_value = "1,2,3")]
    seeds: Vec<u64>,
}

/// One cleaning variant and its results.
struct Variant {
    name: String,
    cleaned: CleanedLabels,
    /// Metrics per playlist, one entry per seed.
    runs_by_playlist: Vec<Vec<LobbyMetrics>>,
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

fn print_runs(label: &str, runs: &[LobbyMetrics]) {
    let within_r = summarise(runs, |metrics| metrics.within_r);
    let concordance = summarise(runs, |metrics| metrics.concordance);
    let lobby_rmse = summarise(runs, |metrics| metrics.lobby_rmse);
    let player_rmse = summarise(runs, |metrics| metrics.player_rmse);
    println!(
        "    {label:<26} within_r={:.3} ±{:.3}  concordance={:.3} ±{:.3}  lobby_rmse={:.1} ±{:.1}  player_rmse={:.1} ±{:.1}",
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

/// Root mean square of `errors`.
fn rmse(errors: &[f32]) -> f64 {
    (errors
        .iter()
        .map(|error| f64::from(*error).powi(2))
        .sum::<f64>()
        / errors.len().max(1) as f64)
        .sqrt()
}

/// Player RMSE on the evaluation split for players whose account also appears in a
/// training lobby, against those whose account does not.
fn print_leakage(lobbies: &[Lobby], model: &TabularSkillModel, accounts: &AccountIds) {
    let training_accounts: HashSet<AccountId> = lobbies
        .iter()
        .filter(|lobby| lobby.partition != Partition::Evaluation)
        .flat_map(|lobby| {
            (0..lobby.targets.len()).filter_map(|slot| {
                accounts.by_player.get(&PlayerKey {
                    replay_id: lobby.replay_id,
                    slot,
                })
            })
        })
        .copied()
        .collect();
    for playlist in TeamSize::all() {
        let mut seen = Vec::new();
        let mut unseen = Vec::new();
        for lobby in lobbies.iter().filter(|lobby| {
            lobby.partition == Partition::Evaluation
                && lobby.players_per_team() == Some(playlist.players_per_team())
        }) {
            let predictions = model.predict_stats(&lobby.stat_references());
            for slot in lobby.labelled_slots() {
                let Some(Some(prediction)) = predictions.get(slot) else {
                    continue;
                };
                let Some(target) = lobby.targets.get(slot) else {
                    continue;
                };
                let account = accounts.by_player.get(&PlayerKey {
                    replay_id: lobby.replay_id,
                    slot,
                });
                if account.is_some_and(|account| training_accounts.contains(account)) {
                    seen.push(prediction - target);
                } else {
                    unseen.push(prediction - target);
                }
            }
        }
        println!(
            "    {:<9} player RMSE: account also in training {:.0} ({} players), unseen {:.0} ({} players)",
            playlist.name(),
            rmse(&seen),
            seen.len(),
            rmse(&unseen),
            unseen.len()
        );
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let lobbies = read_lobbies(&args.match_stats)?;
    let accounts = AccountIds::read(&args.accounts)?;
    let layout = TabularLayout {
        own: true,
        deviation: true,
        lobby_mean: true,
        team_context: true,
        stats: (0..MATCH_STAT_COUNT).collect(),
    };

    println!(
        "== out-of-fold scores ({} accounts mapped)",
        accounts.by_player.len()
    );
    let out_of_fold = out_of_fold_plays_like(
        &lobbies,
        &PlaysLikeModel::own_stats_layout(),
        &GradientBoostingConfig::default(),
    );
    let per_game = clean_labels(&lobbies, &out_of_fold, CLEAN_MARGIN_MMR);
    let mut variants = vec![Variant {
        name: "per game".to_owned(),
        cleaned: per_game,
        runs_by_playlist: TeamSize::all().map(|_| Vec::new()).collect(),
    }];
    for minimum_games in [2, 3] {
        variants.push(Variant {
            name: format!("per account, {minimum_games}+ games"),
            cleaned: clean_labels_by_account(
                &lobbies,
                &out_of_fold,
                &accounts,
                AccountCleaning {
                    margin: CLEAN_MARGIN_MMR,
                    minimum_games,
                },
            ),
            runs_by_playlist: TeamSize::all().map(|_| Vec::new()).collect(),
        });
    }
    for variant in &variants {
        println!("== {}", variant.name);
        variant.cleaned.counts.print_report();
        if let Some(per_game) = variants.first().filter(|first| first.name != variant.name) {
            let extra = variant
                .cleaned
                .dropped
                .players
                .difference(&per_game.cleaned.dropped.players)
                .count();
            let rescued = per_game
                .cleaned
                .dropped
                .players
                .difference(&variant.cleaned.dropped.players)
                .count();
            println!(
                "  vs per game: {extra} more labels dropped (their account plays above it), {rescued} kept (one standout game)"
            );
        }
    }

    for &seed in &args.seeds {
        let config = GradientBoostingConfig {
            seed,
            ..GradientBoostingConfig::default()
        };
        for (index, variant) in variants.iter_mut().enumerate() {
            println!("seed {seed}: {}", variant.name);
            let model = train_model(&variant.cleaned.lobbies, &layout, &config);
            if index == 0 && Some(&seed) == args.seeds.first() {
                println!("  leakage check (per-game cleaning):");
                print_leakage(&lobbies, &model, &accounts);
            }
            for (playlist, runs) in TeamSize::all().zip(&mut variant.runs_by_playlist) {
                let playlist_lobbies = playlist_lobbies(&lobbies, playlist.players_per_team());
                runs.push(score(
                    &playlist_lobbies,
                    &model,
                    playlist.players_per_team(),
                ));
            }
        }
    }

    println!(
        "\nEvaluation split, mean over {} seeds (± half the seed range):",
        args.seeds.len()
    );
    for (index, playlist) in TeamSize::all().enumerate() {
        println!("  {}", playlist.name());
        for variant in &variants {
            if let Some(runs) = variant.runs_by_playlist.get(index) {
                print_runs(&variant.name, runs);
            }
        }
    }
    Ok(())
}
