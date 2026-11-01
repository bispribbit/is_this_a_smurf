//! Results tables and error view.

use dioxus::prelude::*;
use rand::{RngExt, rng};
use replay_structs::{Team, UnsupportedReplayMatch};

use crate::app_state::{AppState, PlayerAverage, PredictionResults, ProgressState};
use crate::branding::SMURF_SUSPECT_BADGE;
use crate::rank_icon::rank_division_icon_asset;
use crate::verdict_copy::VerdictSegment;

const PREDICTED_RANK_ROAST_LABELS: [&str; 10] = [
    "Estimated Delusion Bracket",
    "Alleged Skill Bracket",
    "Court-Ordered Rank Guessed",
    "Projected Excuse Division",
    "Certified Bad Decision Tier",
    "Approximate Main Character Level",
    "Likely Quick Chat Skill Class",
    "Estimated Blame Allocation Rank",
    "Audited Ball-Chasing Bracket",
    "Professional Opinion Nobody Asked For",
];

/// Badge tooltip for why a player is flagged.
const fn smurf_badge_text(reason: skill_model::SmurfReason) -> &'static str {
    match reason {
        skill_model::SmurfReason::RankedAboveLobby => "Predicted rank is much higher than lobby",
        skill_model::SmurfReason::DerankingAboveLobby => {
            "Forfeited while level or ahead, and plays above this lobby: deranking?"
        }
        skill_model::SmurfReason::PlaysAboveLobby => "Plays several ranks above this lobby",
        skill_model::SmurfReason::MechanicsAboveLobby => {
            "Does several mechanics far above this lobby's rank"
        }
    }
}

fn predicted_rank_roast_label() -> &'static str {
    let mut rng = rng();
    let index = rng.random_range(0..PREDICTED_RANK_ROAST_LABELS.len());
    PREDICTED_RANK_ROAST_LABELS
        .get(index)
        .copied()
        .unwrap_or("Estimated Delusion Bracket")
}

/// Blue and orange team summary cards for a completed prediction.
#[component]
pub(crate) fn PlayerSummaryGrid(results: PredictionResults) -> Element {
    let blue_players: Vec<PlayerAverage> = results
        .player_averages
        .iter()
        .filter(|player| player.team == Team::Blue)
        .cloned()
        .collect();
    let orange_players: Vec<PlayerAverage> = results
        .player_averages
        .iter()
        .filter(|player| player.team == Team::Orange)
        .cloned()
        .collect();

    let format = results.match_format;
    let size = format.players_per_team;
    let format_note = if format.ranked {
        format!("Ranked {size}v{size} · ranks are competitive {size}v{size} ranks")
    } else {
        format!(
            "Casual {size}v{size} · ranks are the competitive {size}v{size} rank each player plays like"
        )
    };

    rsx! {
        p { class: "text-center text-sm text-gray-400 mb-3", "{format_note}" }
        div { class: "grid grid-cols-1 md:grid-cols-2 gap-6 mb-10",
            TeamSummaryCard {
                team: Team::Blue,
                loading: false,
                players_complete: blue_players,
                player_names_loading: vec![],
            }
            TeamSummaryCard {
                team: Team::Orange,
                loading: false,
                players_complete: orange_players,
                player_names_loading: vec![],
            }
        }
    }
}

/// One-line verdict under the team summary: tier-flavored if no smurf, or names suspects.
#[component]
pub(crate) fn MatchVerdictBanner(results: PredictionResults) -> Element {
    let segments = crate::verdict_copy::match_verdict_segments(&results);
    rsx! {
        div { class: "rounded-xl border border-gray-800 bg-gray-900/80 px-5 py-4 mb-10",
            p { class: "text-center text-base leading-relaxed text-gray-200",
                for segment in segments {
                    match segment {
                        VerdictSegment::Plain(text) => rsx! { "{text}" },
                        VerdictSegment::BoldName(name) => rsx! {
                            span { class: "font-bold text-gray-100", "{name}" }
                        },
                    }
                }
            }
        }
    }
}

/// Same layout as [`PlayerSummaryGrid`], but team averages and rank icons stay hidden until results exist.
#[component]
pub(crate) fn PlayerSummaryGridLoading(progress: ProgressState) -> Element {
    let Some(track) = progress.timeline.as_ref() else {
        return rsx! {
            div { class: "grid grid-cols-1 md:grid-cols-2 gap-6 mb-10",
                TeamSummaryCard {
                    team: Team::Blue,
                    loading: true,
                    players_complete: vec![],
                    player_names_loading: vec![],
                }
                TeamSummaryCard {
                    team: Team::Orange,
                    loading: true,
                    players_complete: vec![],
                    player_names_loading: vec![],
                }
            }
        };
    };

    let mut blue_names: Vec<String> = Vec::new();
    let mut orange_names: Vec<String> = Vec::new();
    for player_lane_index in 0..track.player_names.len() {
        let player_name = track
            .player_names
            .get(player_lane_index)
            .cloned()
            .unwrap_or_else(|| format!("Player {}", player_lane_index + 1));
        match track
            .player_teams
            .get(player_lane_index)
            .copied()
            .unwrap_or(Team::Blue)
        {
            Team::Blue => blue_names.push(player_name),
            Team::Orange => orange_names.push(player_name),
        }
    }

    rsx! {
        div { class: "grid grid-cols-1 md:grid-cols-2 gap-6 mb-10",
            TeamSummaryCard {
                team: Team::Blue,
                loading: true,
                players_complete: vec![],
                player_names_loading: blue_names,
            }
            TeamSummaryCard {
                team: Team::Orange,
                loading: true,
                players_complete: vec![],
                player_names_loading: orange_names,
            }
        }
    }
}

/// Warning triangle used to frame the smurf suspect illustration (`alert` — car — `alert`).
#[component]
fn LobbyAlertTriangleIcon() -> Element {
    rsx! {
        svg {
            class: "w-9 h-9 sm:w-10 sm:h-10 text-red-500 shrink-0 drop-shadow-md",
            fill: "none",
            stroke: "currentColor",
            stroke_width: "1.5",
            view_box: "0 0 24 24",
            path {
                stroke_linecap: "round",
                stroke_linejoin: "round",
                d: "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z",
            }
        }
    }
}

/// Card showing team summary with player rank icons and rank names.
#[component]
fn TeamSummaryCard(
    team: Team,
    loading: bool,
    players_complete: Vec<PlayerAverage>,
    player_names_loading: Vec<String>,
) -> Element {
    let roast_label = predicted_rank_roast_label();
    let border_color = match team {
        Team::Blue => "border-blue-500/50",
        Team::Orange => "border-orange-500/50",
    };
    let header_gradient = match team {
        Team::Blue => "from-blue-500/20 to-transparent",
        Team::Orange => "from-orange-500/20 to-transparent",
    };
    let text_accent = match team {
        Team::Blue => "text-blue-400",
        Team::Orange => "text-orange-400",
    };

    rsx! {
        div { class: "bg-gray-900 rounded-xl border {border_color} overflow-hidden",
            div { class: "bg-gradient-to-r {header_gradient} px-6 py-4",
                div { class: "flex items-center justify-between gap-3",
                    h2 { class: "text-xl font-bold {text_accent}", "{team} Team" }
                    if loading {
                        span { class: "text-gray-500 text-sm font-medium", "Calculating…" }
                    } else {
                        span { "{roast_label}" }
                    }
                }
            }
            div { class: "divide-y divide-gray-800",
                if loading {
                    if player_names_loading.is_empty() {
                        p { class: "px-6 py-6 text-sm text-gray-500 text-center",
                            "Waiting for replay data…"
                        }
                    } else {
                        for name in player_names_loading {
                            div { class: "px-6 py-4 flex items-center min-h-[5.5rem]",
                                p { class: "font-semibold text-gray-100 break-words",
                                    "{name}"
                                }
                            }
                        }
                    }
                } else {
                    for player in &players_complete {
                        div { class: "px-6 py-4 flex items-center gap-3 min-h-[5.5rem]",
                            div { class: "min-w-0 flex-1 flex flex-col justify-center gap-1",
                                p { class: "font-semibold text-gray-100 break-words",
                                    "{player.name}"
                                }
                                // A smurf's lobby-anchored rank contradicts their icon and timeline,
                                // so only the rank they play like is shown.
                                if let Some(plays_like) = &player.plays_like {
                                    p { class: "text-sm font-semibold text-red-400",
                                        "Plays like {plays_like}"
                                    }
                                } else {
                                    p { class: "text-sm text-gray-500", "{player.rank}" }
                                }
                                if let Some(forfeit) = &player.forfeit {
                                    p { class: "text-sm font-semibold text-red-400", "{forfeit}" }
                                }
                                if let Some(mechanics) = &player.mechanics {
                                    p { class: "text-xs font-medium text-amber-300 leading-snug",
                                        "{mechanics}"
                                    }
                                }
                                if let Some(roast) = &player.next_rank_roast {
                                    p { class: "text-xs italic text-gray-400 leading-snug",
                                        "{roast}"
                                    }
                                }
                            }
                            if let Some(reason) = player.smurf {
                                div { class: "flex shrink-0 items-center justify-center gap-2 sm:gap-3",
                                    LobbyAlertTriangleIcon {}
                                    img {
                                        src: SMURF_SUSPECT_BADGE,
                                        alt: smurf_badge_text(reason),
                                        title: smurf_badge_text(reason),
                                        class: "h-14 w-auto max-w-[5rem] sm:h-[4.5rem] sm:max-w-[6rem] object-contain",
                                    }
                                    LobbyAlertTriangleIcon {}
                                }
                            }
                            div { class: "shrink-0 flex flex-col items-end justify-center gap-1",
                                // A smurf's icon is the rank they play like, as on their timeline.
                                img {
                                    src: rank_division_icon_asset(player.plays_like.unwrap_or(player.rank)),
                                    alt: "",
                                    title: player.plays_like.unwrap_or(player.rank).to_string(),
                                    class: "block w-12 h-11 object-contain drop-shadow-md",
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Shown when the replay is valid but not a supported match (e.g. Hoops or 4v4).
#[component]
pub(crate) fn UnsupportedReplayPage(
    details: UnsupportedReplayMatch,
    state: Signal<AppState>,
) -> Element {
    let mut state = state;
    let message = details.user_message();

    rsx! {
        div { class: "flex flex-col items-center justify-center min-h-screen px-4",
            svg {
                class: "w-20 h-20 text-amber-500 mb-6",
                fill: "none",
                stroke: "currentColor",
                stroke_width: "1.5",
                view_box: "0 0 24 24",
                path {
                    stroke_linecap: "round",
                    stroke_linejoin: "round",
                    d: "M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z",
                }
            }
            h2 { class: "text-2xl font-bold text-amber-400 mb-4", "Unsupported match type" }
            p { class: "text-gray-400 text-center max-w-md mb-8", "{message}" }
            button {
                class: "px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-colors",
                onclick: move |_| {
                    state.set(AppState::WaitingForUpload);
                },
                "Try another replay"
            }
            p { class: "text-gray-500 text-sm mt-3", "or drop a .replay anywhere on this page" }
        }
    }
}

/// Error page with a message and a retry button.
#[component]
pub(crate) fn ErrorPage(message: String, state: Signal<AppState>) -> Element {
    let mut state = state;

    rsx! {
        div { class: "flex flex-col items-center justify-center min-h-screen px-4",
            // Error icon
            svg {
                class: "w-20 h-20 text-red-500 mb-6",
                fill: "none",
                stroke: "currentColor",
                stroke_width: "1.5",
                view_box: "0 0 24 24",
                path {
                    stroke_linecap: "round",
                    stroke_linejoin: "round",
                    d: "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z",
                }
            }
            h2 { class: "text-2xl font-bold text-red-400 mb-4", "Error" }
            p { class: "text-gray-400 text-center max-w-md mb-8", "{message}" }
            button {
                class: "px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-colors",
                onclick: move |_| {
                    state.set(AppState::WaitingForUpload);
                },
                "Try again"
            }
            p { class: "text-gray-500 text-sm mt-3", "or drop a .replay anywhere on this page" }
        }
    }
}
