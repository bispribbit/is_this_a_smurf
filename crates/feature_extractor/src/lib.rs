#![expect(clippy::indexing_slicing)]

//! Per-player match statistics from parsed Rocket League replays.
//!
//! [`compute_player_match_stats`] summarises how each player played a whole match (movement, boost, positioning, ball control, mechanics, scoreboard…), and
//! [`compute_player_window_stats`] does the same for a slice of it. These summaries are the
//! inputs of the skill model (`skill_model`).

use std::collections::{BTreeMap, HashSet};

pub use replay_structs::{MATCH_STAT_COUNT, MatchStat};
use replay_structs::{ParsedReplay, Team};

mod match_stats;
pub use match_stats::{
    PlayerMatchStats, compute_player_match_stats, compute_player_window_stats, goal_windows,
    time_windows,
};

/// Roster slots per team. Above the largest team size (3) because casual lobbies replace
/// players who leave, so one team can field more distinct players over a match.
pub const SLOTS_PER_TEAM: usize = 5;
/// Roster slots in a match: blue slots first, then orange.
pub const TOTAL_SLOTS: usize = SLOTS_PER_TEAM * 2;

/// How a name was seen in a replay.
#[derive(Debug, Clone, Default)]
struct Sightings {
    name: String,
    blue_frames: usize,
    orange_frames: usize,
    /// Team from the header scoreboard, which outranks the per-frame attribute.
    header_team: Option<Team>,
}

impl Sightings {
    const fn frames(&self) -> usize {
        self.blue_frames + self.orange_frames
    }

    fn blue_share(&self) -> f64 {
        self.blue_frames as f64 / self.frames().max(1) as f64
    }
}

/// Names the parser gives cars it could not link to a player (`Player_<actor id>`).
fn is_placeholder_name(name: &str) -> bool {
    name.strip_prefix("Player_")
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}

/// Canonical ordered roster.
///
/// The first [`SLOTS_PER_TEAM`] slots hold the blue team sorted alphabetically by player
/// name; the next ones hold the orange team sorted the same way. Empty strings mark unused
/// slots (a duel uses one slot per team).
#[derive(Debug, Clone)]
pub struct PlayerRoster {
    /// Blue names, then orange names.
    pub names: [String; TOTAL_SLOTS],
}

impl PlayerRoster {
    /// Builds the roster from a parsed replay.
    ///
    /// Neither source is complete on its own: the header scoreboard leaves out players who
    /// quit early, and the per-frame team attribute flips back and forth in some replays.
    /// So every real name from either source is a candidate; the header decides the team
    /// when it lists the player, otherwise the team the player is seen on most (a full lobby
    /// without header teams is split evenly by how often each name is seen on blue). Each
    /// team keeps up to [`SLOTS_PER_TEAM`] players, header-listed first, then by time on the
    /// pitch.
    #[must_use]
    pub fn from_parsed(parsed: &ParsedReplay) -> Self {
        let mut by_name: BTreeMap<String, Sightings> = BTreeMap::new();
        for frame in &parsed.frames {
            for player in &frame.players {
                if is_placeholder_name(&player.name) {
                    continue;
                }
                let sightings =
                    by_name
                        .entry((*player.name).clone())
                        .or_insert_with(|| Sightings {
                            name: (*player.name).clone(),
                            ..Sightings::default()
                        });
                match player.team {
                    Team::Blue => sightings.blue_frames += 1,
                    Team::Orange => sightings.orange_frames += 1,
                }
            }
        }
        for player in &parsed.header_player_stats {
            if player.name.is_empty() {
                continue;
            }
            let sightings = by_name
                .entry(player.name.clone())
                .or_insert_with(|| Sightings {
                    name: player.name.clone(),
                    ..Sightings::default()
                });
            sightings.header_team = player.team;
        }

        let mut candidates: Vec<Sightings> = by_name.into_values().collect();
        let no_header_teams = candidates.iter().all(|s| s.header_team.is_none());
        let mut blue: Vec<Sightings> = Vec::new();
        let mut orange: Vec<Sightings> = Vec::new();
        let players_per_team = parsed.match_format.players_per_team;
        if no_header_teams && candidates.len() == players_per_team * 2 {
            candidates.sort_by(|a, b| b.blue_share().total_cmp(&a.blue_share()));
            for (index, sightings) in candidates.into_iter().enumerate() {
                if index < players_per_team {
                    blue.push(sightings);
                } else {
                    orange.push(sightings);
                }
            }
        } else {
            for sightings in candidates {
                let team = sightings.header_team.unwrap_or(
                    if sightings.blue_frames >= sightings.orange_frames {
                        Team::Blue
                    } else {
                        Team::Orange
                    },
                );
                match team {
                    Team::Blue => blue.push(sightings),
                    Team::Orange => orange.push(sightings),
                }
            }
        }
        let keep_team = |mut team: Vec<Sightings>| -> Vec<String> {
            // Header-listed players first, then by time on the pitch.
            team.sort_by(|a, b| {
                b.header_team
                    .is_some()
                    .cmp(&a.header_team.is_some())
                    .then(b.frames().cmp(&a.frames()))
            });
            let mut names: Vec<String> = team
                .into_iter()
                .take(SLOTS_PER_TEAM)
                .map(|s| s.name)
                .collect();
            names.sort();
            names
        };
        let blue = keep_team(blue);
        let orange = keep_team(orange);
        Self {
            names: core::array::from_fn(|slot| {
                if slot < SLOTS_PER_TEAM {
                    blue.get(slot).cloned().unwrap_or_default()
                } else {
                    orange
                        .get(slot - SLOTS_PER_TEAM)
                        .cloned()
                        .unwrap_or_default()
                }
            }),
        }
    }
}

/// Frame indices inside goal-replay windows: from a goal frame (exclusive) to the next
/// kickoff (exclusive), capped at 600 frames. Those frames show the celebration camera,
/// teleporting cars and a paused clock, so the stats skip them.
fn build_goal_replay_excluded_set(
    goal_frames: &[usize],
    kickoff_frames: &[usize],
) -> HashSet<usize> {
    let mut excluded = HashSet::new();
    for &goal in goal_frames {
        let next_kickoff = kickoff_frames
            .iter()
            .find(|&&kickoff| kickoff > goal)
            .copied()
            .unwrap_or(usize::MAX);
        for frame_index in (goal + 1)..next_kickoff.min(goal + 600) {
            excluded.insert(frame_index);
        }
    }
    excluded
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use replay_structs::{GameFrame, HeaderPlayerStats, MatchFormat, PlayerState};

    use super::*;

    fn player(name: &str, team: Team) -> PlayerState {
        PlayerState {
            actor_id: 0,
            name: Arc::new(name.to_string()),
            team,
            actor_state: replay_structs::ActorState::default(),
        }
    }

    /// A casual 3v3 where one blue player left and two others took the seat in turn: blue
    /// fields four distinct players over the match, and all four get a slot.
    #[test]
    fn casual_roster_keeps_replacement_players() {
        let blue = ["leaver", "stayer_one", "stayer_two", "joiner"];
        let orange = ["orange_one", "orange_two", "orange_three"];
        let frame = GameFrame {
            players: blue
                .iter()
                .map(|name| player(name, Team::Blue))
                .chain(orange.iter().map(|name| player(name, Team::Orange)))
                .collect(),
            ..GameFrame::default()
        };
        let parsed = ParsedReplay {
            frames: vec![frame; 10],
            header_player_stats: orange
                .iter()
                .map(|name| HeaderPlayerStats {
                    name: (*name).to_string(),
                    team: Some(Team::Orange),
                    ..HeaderPlayerStats::default()
                })
                .collect(),
            match_format: MatchFormat {
                players_per_team: 3,
                ranked: false,
            },
            ..ParsedReplay::default()
        };
        let roster = PlayerRoster::from_parsed(&parsed);
        let blue_slots: Vec<&str> = roster.names[..SLOTS_PER_TEAM]
            .iter()
            .filter(|name| !name.is_empty())
            .map(String::as_str)
            .collect();
        assert_eq!(blue_slots.len(), 4);
        for name in blue {
            assert!(blue_slots.contains(&name), "{name} lost its slot");
        }
        let orange_slots = roster.names[SLOTS_PER_TEAM..]
            .iter()
            .filter(|name| !name.is_empty())
            .count();
        assert_eq!(orange_slots, 3);
    }

    #[test]
    fn goal_replay_runs_until_the_next_kickoff() {
        let excluded = build_goal_replay_excluded_set(&[100], &[50, 130]);
        assert!(!excluded.contains(&100));
        assert!(excluded.contains(&101));
        assert!(excluded.contains(&129));
        assert!(!excluded.contains(&130));
    }
}
