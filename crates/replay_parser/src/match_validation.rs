//! Accepts 1v1, 2v2 and 3v3 soccar replays, ranked or casual, and reports which it is.

use std::collections::HashSet;

use boxcars::{Attribute, HeaderProp, Replay, UniqueId};
use replay_structs::{MatchFormat, UnsupportedReplayMatch};

const SOCCAR_REPLAY_GAME_TYPE: &str = "TAGame.Replay_Soccar_TA";

/// Playlist ids of the soccar queues, from the replay header's `PlaylistId`.
const CASUAL_DUELS_PLAYLIST_ID: i32 = 1;
const CASUAL_DOUBLES_PLAYLIST_ID: i32 = 2;
const CASUAL_STANDARD_PLAYLIST_ID: i32 = 3;
const RANKED_DUELS_PLAYLIST_ID: i32 = 10;
const RANKED_DOUBLES_PLAYLIST_ID: i32 = 11;
const RANKED_SOLO_STANDARD_PLAYLIST_ID: i32 = 12;
const RANKED_STANDARD_PLAYLIST_ID: i32 = 13;

/// Largest supported team size (standard).
const MAXIMUM_PLAYERS_PER_TEAM: i32 = 3;

/// Validates that a replay is a supported match and returns its format.
///
/// Ranked lobbies must have exactly two full teams of distinct players (a player who left
/// still counts). Casual lobbies only need a player on the pitch: players join and leave
/// mid-match, so the count can be above or below two full teams.
///
/// # Errors
/// Returns an error if the replay is not a supported match.
pub fn validate_supported_match(replay: &Replay) -> Result<MatchFormat, UnsupportedReplayMatch> {
    if replay.game_type != SOCCAR_REPLAY_GAME_TYPE {
        return Err(UnsupportedReplayMatch {
            detected_mode_label: game_mode_label_from_game_type(&replay.game_type),
        });
    }

    let ranked = match playlist_id_from_header(replay) {
        Some(
            RANKED_DUELS_PLAYLIST_ID
            | RANKED_DOUBLES_PLAYLIST_ID
            | RANKED_SOLO_STANDARD_PLAYLIST_ID
            | RANKED_STANDARD_PLAYLIST_ID,
        ) => true,
        Some(
            CASUAL_DUELS_PLAYLIST_ID | CASUAL_DOUBLES_PLAYLIST_ID | CASUAL_STANDARD_PLAYLIST_ID,
        ) => false,
        Some(other_id) => {
            return Err(UnsupportedReplayMatch {
                detected_mode_label: describe_playlist_id(other_id),
            });
        }
        None => match match_type_from_objects_list(replay).as_deref() {
            Some(name) if name.contains("PublicRanked") => true,
            Some(name) if name.contains("Public") => false,
            other => {
                return Err(UnsupportedReplayMatch {
                    detected_mode_label: non_ranked_or_unknown_label(other),
                });
            }
        },
    };

    let team_size =
        header_int_property(replay, "TeamSize").ok_or_else(|| UnsupportedReplayMatch {
            detected_mode_label: "missing team size in replay header".to_string(),
        })?;
    if !(1..=MAXIMUM_PLAYERS_PER_TEAM).contains(&team_size) {
        return Err(UnsupportedReplayMatch {
            detected_mode_label: team_size_label(team_size),
        });
    }
    let players_per_team = usize::try_from(team_size).unwrap_or(1);

    let player_count = effective_human_player_count(replay);
    let full_lobby = players_per_team * 2;
    if ranked && player_count != full_lobby {
        return Err(UnsupportedReplayMatch {
            detected_mode_label: format!(
                "this lobby ({player_count} players; expected {full_lobby} for ranked {players_per_team}v{players_per_team})"
            ),
        });
    }
    if player_count == 0 {
        return Err(UnsupportedReplayMatch {
            detected_mode_label: "this lobby (no players found)".to_string(),
        });
    }

    Ok(MatchFormat {
        players_per_team,
        ranked,
    })
}

fn header_int_property(replay: &Replay, key: &str) -> Option<i32> {
    replay
        .properties
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| {
            if let HeaderProp::Int(value) = v {
                Some(*value)
            } else {
                None
            }
        })
}

fn playlist_id_from_header(replay: &Replay) -> Option<i32> {
    replay
        .properties
        .iter()
        .find(|(k, _)| k == "PlaylistId")
        .and_then(|(_, v)| {
            if let HeaderProp::Int(value) = v {
                Some(*value)
            } else {
                None
            }
        })
}

/// Rows from `PlayerStats`, whether stored as a top-level array or nested under a struct.
pub fn player_stats_row_slices(replay: &Replay) -> Vec<&Vec<(String, HeaderProp)>> {
    for (key, value) in &replay.properties {
        if key == "PlayerStats" {
            return collect_player_stat_row_slices(value);
        }
    }
    Vec::new()
}

fn collect_player_stat_row_slices(property: &HeaderProp) -> Vec<&Vec<(String, HeaderProp)>> {
    match property {
        HeaderProp::Array(entries) => entries.iter().collect(),
        HeaderProp::Struct { fields, .. } => fields
            .iter()
            .flat_map(|(_, field)| collect_player_stat_row_slices(field))
            .collect(),
        _ => Vec::new(),
    }
}

/// Counts unique human players by combining three complementary signals:
///
/// - `PlayerStats` header rows: never inflated, but sometimes incomplete.
/// - Named PRI actors in network frames: never inflated (phantom PRIs have no name), but
///   collapses players that share the same display name (e.g. privacy-hidden "Player 0").
/// - `UniqueId` on PRI actors: the platform identity (Steam, PS4, Xbox, Epic, etc.) that each
///   player carries regardless of display name — this is what ballchasing uses to distinguish
///   players.
///
/// Taking the maximum of all three gives the most accurate count.
fn effective_human_player_count(replay: &Replay) -> usize {
    let header_rows = player_stats_row_slices(replay).len();
    let pri_counts = count_pri_players(replay);
    header_rows
        .max(pri_counts.named_player_count)
        .max(pri_counts.unique_id_count)
}

struct PriPlayerCounts {
    named_player_count: usize,
    unique_id_count: usize,
}

/// Scans network frames for PRI actors, collecting both unique display names and unique platform
/// identities (`UniqueId`). Display names can collide (e.g. multiple "Player 0"), but `UniqueId`
/// is always unique per player.
fn count_pri_players(replay: &Replay) -> PriPlayerCounts {
    let Some(network) = replay.network_frames.as_ref() else {
        return PriPlayerCounts {
            named_player_count: 0,
            unique_id_count: 0,
        };
    };

    let player_name_attr_id = replay
        .objects
        .iter()
        .position(|o| o == "Engine.PlayerReplicationInfo:PlayerName");

    let unique_id_attr_id = replay
        .objects
        .iter()
        .position(|o| o == "Engine.PlayerReplicationInfo:UniqueId");

    let mut pri_actor_ids: HashSet<i32> = HashSet::new();
    let mut named_players: HashSet<String> = HashSet::new();
    let mut unique_ids: HashSet<UniqueId> = HashSet::new();

    for frame in &network.frames {
        for new_actor in &frame.new_actors {
            let object_index = new_actor.object_id.0 as usize;
            if let Some(object_name) = replay.objects.get(object_index)
                && object_name.contains("PRI_TA")
                && !object_name.contains(':')
            {
                pri_actor_ids.insert(new_actor.actor_id.0);
            }
        }

        for update in &frame.updated_actors {
            let actor_id = update.actor_id.0;
            if !pri_actor_ids.contains(&actor_id) {
                continue;
            }

            if player_name_attr_id.is_some_and(|id| update.object_id.0 as usize == id)
                && let Attribute::String(name) = &update.attribute
                && !name.is_empty()
            {
                named_players.insert(name.clone());
            }

            if unique_id_attr_id.is_some_and(|id| update.object_id.0 as usize == id)
                && let Attribute::UniqueId(uid) = &update.attribute
            {
                unique_ids.insert(*uid.clone());
            }
        }
    }

    PriPlayerCounts {
        named_player_count: named_players.len(),
        unique_id_count: unique_ids.len(),
    }
}

/// Breakdown of how the replay parser estimates human player count (for diagnostics).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HumanPlayerCountBreakdown {
    /// Rows under `PlayerStats` in the replay header.
    pub player_stats_rows: usize,
    /// Unique player names observed on PRI actors in network frames.
    pub named_pri_player_count: usize,
    /// Unique platform identities (`UniqueId`) observed on PRI actors.
    pub unique_id_pri_player_count: usize,
    /// `TeamSize` from the replay header when present.
    pub team_size_header: Option<i32>,
    /// Final count used for validation.
    pub effective_count: usize,
}

/// Returns header and network-derived player counts, plus the effective count used for validation.
#[must_use]
pub fn diagnose_human_player_count(replay: &Replay) -> HumanPlayerCountBreakdown {
    let player_stats_rows = player_stats_row_slices(replay).len();
    let pri_counts = count_pri_players(replay);
    let team_size_header = header_int_property(replay, "TeamSize");
    let effective_count = player_stats_rows
        .max(pri_counts.named_player_count)
        .max(pri_counts.unique_id_count);
    HumanPlayerCountBreakdown {
        player_stats_rows,
        named_pri_player_count: pri_counts.named_player_count,
        unique_id_pri_player_count: pri_counts.unique_id_count,
        team_size_header,
        effective_count,
    }
}

/// Match type class names appear in the replay `objects` table (not necessarily as `new_actors`).
fn match_type_from_objects_list(replay: &Replay) -> Option<String> {
    let candidates: Vec<&String> = replay
        .objects
        .iter()
        .filter(|name| name.starts_with("TAGame.MatchType_"))
        .collect();
    if candidates.is_empty() {
        return None;
    }
    if let Some(ranked) = candidates.iter().find(|name| name.contains("PublicRanked")) {
        return Some((*ranked).clone());
    }
    candidates.first().map(|entry| (*entry).clone())
}

fn team_size_label(team_size: i32) -> String {
    match team_size {
        4 => "4v4 (Chaos)".to_string(),
        _ => format!("team size {team_size}"),
    }
}

fn game_mode_label_from_game_type(game_type: &str) -> String {
    if game_type.contains("Hoops") {
        return "Hoops".to_string();
    }
    if game_type.contains("Dropshot") {
        return "Dropshot".to_string();
    }
    if game_type.contains("Snow") {
        return "Snow Day".to_string();
    }
    if game_type.contains("Rumble") {
        return "Rumble".to_string();
    }
    if game_type.contains("Heatseeker") {
        return "Heatseeker".to_string();
    }
    if game_type.contains("Breakout") {
        return "Breakout".to_string();
    }
    game_type.to_string()
}

fn describe_playlist_id(playlist_id: i32) -> String {
    match playlist_id {
        10 => "ranked 1v1 (duels)".to_string(),
        11 => "ranked 2v2 (doubles)".to_string(),
        12 => "ranked solo standard 3v3".to_string(),
        1 => "casual 1v1".to_string(),
        2 => "casual 2v2".to_string(),
        3 => "casual 3v3 standard".to_string(),
        15 | 16 => "Snow Day".to_string(),
        17 | 18 => "Rocket Labs".to_string(),
        25 | 26 => "Hoops".to_string(),
        27 | 28 => "Rumble".to_string(),
        29 | 30 => "Dropshot".to_string(),
        31 | 32 => "Heatseeker".to_string(),
        _ => format!("this playlist (id {playlist_id})"),
    }
}

fn non_ranked_or_unknown_label(match_type_spawn: Option<&str>) -> String {
    match match_type_spawn {
        Some(name) if name.contains("Private") => "private matches or custom games".to_string(),
        Some(name) => format!("this match type ({name})"),
        None => "this match (its queue could not be read from the replay header)".to_string(),
    }
}
