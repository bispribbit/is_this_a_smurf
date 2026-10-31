//! End to end: the shipped bundle scores real replays the way the app does.
//!
//! Runs `SkillModelBundle::analyze` — the exact call the app makes — on the repository's
//! sample replays and checks the output is complete and plausible. Run with `--nocapture`
//! to see what the app would show.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use skill_model::SkillModelBundle;

    fn repository_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// A sample replay and its playlist size.
    struct SampleReplay {
        path: PathBuf,
        players_per_team: usize,
        /// From `data/smurf`: someone must be flagged.
        expect_smurf: bool,
    }

    fn sample_replays() -> Vec<SampleReplay> {
        let root = repository_root();
        let test_data = |name: &str, players_per_team: usize| SampleReplay {
            path: root.join("test_data").join(name),
            players_per_team,
            expect_smurf: false,
        };
        let mut replays = vec![
            test_data("2af51380-05b5-44ac-8b31-94b8b0f8da84.replay", 3),
            test_data(
                "ranked-doubles-f5650747-1fd4-40cb-a037-7dfaf166bd6d.replay",
                2,
            ),
            test_data(
                "ranked-duels-a9feb907-75e9-4514-b52f-44766e0374f1.replay",
                1,
            ),
        ];
        if let Ok(entries) = std::fs::read_dir(root.join("data/smurf")) {
            let mut smurfs: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
            smurfs.sort();
            // Smurf samples are named like the test data: `ranked-duels-…`, `ranked-doubles-…`,
            // or a bare id for standard.
            replays.extend(smurfs.into_iter().map(|path| {
                let name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let players_per_team = if name.starts_with("ranked-duels-") {
                    1
                } else if name.starts_with("ranked-doubles-") {
                    2
                } else {
                    3
                };
                SampleReplay {
                    path,
                    players_per_team,
                    expect_smurf: true,
                }
            }));
        }
        replays
    }

    #[test]
    fn bundle_scores_sample_replays() {
        let bytes = std::fs::read(repository_root().join("data/skill_model.bin"))
            .expect("data/skill_model.bin exists (run the `train` binary)");
        let bundle = SkillModelBundle::from_bytes(&bytes).expect("bundle decodes");
        assert!(
            bundle.match_model.matches_current_stats(),
            "bundle is stale: retrain"
        );

        for SampleReplay {
            path,
            players_per_team,
            expect_smurf,
        } in sample_replays()
        {
            let replay = std::fs::read(&path).expect("sample replay is readable");
            let parsed =
                replay_parser::parse_replay_from_bytes(&replay).expect("sample replay parses");
            assert_eq!(
                parsed.match_format.players_per_team,
                players_per_team,
                "{}: wrong playlist size",
                path.display()
            );
            assert!(parsed.match_format.ranked, "{}: not ranked", path.display());
            let analysis = bundle.analyze(&parsed);

            println!(
                "\n{}",
                path.file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default()
            );
            let mut scored = 0;
            for slot in 0..feature_extractor::TOTAL_SLOTS {
                let Some(name) = analysis.names.get(slot).filter(|name| !name.is_empty()) else {
                    continue;
                };
                let mmr = analysis.match_mmr.get(slot).copied().flatten();
                let form: Vec<String> = analysis
                    .timeline
                    .iter()
                    .map(|window| {
                        window
                            .player_mmr
                            .get(slot)
                            .copied()
                            .flatten()
                            .map_or_else(|| "  -".to_string(), |value| format!("{value:>5.0}"))
                    })
                    .collect();
                let roast = analysis
                    .coaching
                    .get(slot)
                    .and_then(|advice| advice.as_ref())
                    .map(|advice| advice.line(0, name))
                    .unwrap_or_default();
                let smurf = match analysis.smurf.get(slot).copied().flatten() {
                    Some(skill_model::SmurfReason::RankedAboveLobby) => {
                        "SMURF (ranked)".to_string()
                    }
                    Some(skill_model::SmurfReason::PlaysAboveLobby) => format!(
                        "SMURF (plays like {:.0})",
                        analysis
                            .plays_like_mmr
                            .get(slot)
                            .copied()
                            .flatten()
                            .unwrap_or_default()
                    ),
                    None => String::new(),
                };
                println!(
                    "  {name:<20} {:>6} [{}] {smurf} {roast}",
                    mmr.map_or_else(|| "-".to_string(), |value| format!("{value:.0}")),
                    form.join(" ")
                );
                if let Some(mmr) = mmr {
                    scored += 1;
                    assert!(
                        (0.0..=2500.0).contains(&mmr),
                        "{name}: implausible MMR {mmr}"
                    );
                    assert!(!roast.is_empty(), "{name}: no roast");
                }
            }
            // Every player of a full lobby, allowing one early leaver in standard.
            assert!(
                scored >= (players_per_team * 2).saturating_sub(1).max(2),
                "{}: only {scored} players scored",
                path.display()
            );
            assert!(
                !expect_smurf || analysis.smurf.iter().any(Option::is_some),
                "{}: smurf sample no longer flags anyone",
                path.display()
            );
            // Short clips get fewer windows; every replay gets at least one.
            assert!(
                !analysis.timeline.is_empty(),
                "{}: no timeline windows",
                path.display()
            );
        }
    }
}
