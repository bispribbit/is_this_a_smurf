//! Upload drop zone and full inference pipeline (spawned by [`super::app::App`]).
//!
//! ## Threading and WASM
//!
//! The default browser target `wasm32-unknown-unknown` does **not** support Rust `std::thread`.
//! True parallel threads in WASM need either:
//! - **Web Workers**: a separate JS worker loading another WASM module (or the same bundle with
//!   careful setup), with all inputs and outputs passed through `postMessage` (often large copies),
//!   or
//! - **Atomics + `wasm-bindgen-rayon`**: rebuild with `+atomics,+bulk-memory`, and serve the page
//!   with cross-origin isolation (`Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy`)
//!   so `SharedArrayBuffer` is available.
//!
//! This crate therefore runs everything on the **same thread** as the UI and calls
//! [`crate::browser_async::yield_to_ui`] between steps so the browser can repaint. Scoring a
//! match with the tree model takes a fraction of a second, so no worker is needed.

use dioxus::prelude::*;
use replay_parser::{ReplayAcceptanceError, parse_replay_from_bytes};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use super::processing::{AnalysisTimeline, EarlyWorkingPanel};
use super::results::{MatchVerdictBanner, PlayerSummaryGrid, PlayerSummaryGridLoading};
use crate::app_state::{
    AppState, LocalProcessing, PredictionResults, ProgressState, SegmentStepInfo, StepStatus,
    TimelineTrackState,
};
use crate::branding::IS_THIS_A_SMURF_HERO;
use crate::browser_async::{sleep_milliseconds, yield_for_dom_paint, yield_to_ui};
use crate::embedded_model::load_bundle;
use crate::prediction::{
    FinalScore, build_goal_markers, build_prediction_results, compute_segment_boundary_play_times,
    compute_segment_boundary_times, prepare_players_for_timeline, ranks_from_player_mmr,
    segment_step_infos,
};

/// Upload page with a centered drag-and-drop area.
///
/// The pipeline itself runs on [`super::app::App`]'s scope (see [`run_pipeline`]). On success
/// we keep `AppState::WaitingForUpload` and store the outcome in [`LocalProcessing`] so the
/// timeline and summary stay on one screen without routing. Errors use [`AppState::Error`];
/// unsupported match types use [`AppState::UnsupportedReplay`].
#[component]
pub(crate) fn UploadPage(
    local_processing: Signal<Option<LocalProcessing>>,
    on_replay_selected: Callback<web_sys::File>,
) -> Element {
    let mut local_processing = local_processing;

    if let Some(LocalProcessing {
        filename,
        progress,
        results,
    }) = local_processing()
    {
        let show_timeline = progress.timeline.is_some();
        return rsx! {
            div { class: "flex flex-col min-h-screen w-full bg-gray-950 text-gray-100",
                div { class: "max-w-7xl mx-auto px-4 py-8 w-full flex flex-col gap-8",
                    div { class: "flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 pr-10",
                        div {
                            h1 { class: "text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-orange-400",
                                "Is there a smurf in this match?"
                            }
                            p { class: "text-gray-400 mt-1", "{filename}" }
                        }
                        if results.is_some() {
                            div { class: "flex flex-col items-start sm:items-end gap-1 self-start sm:self-center shrink-0",
                            button {
                                class: "px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg transition-colors",
                                onclick: move |_| {
                                    local_processing.set(None);
                                },
                                "Analyze another replay"
                            }
                            p { class: "text-gray-500 text-xs", "or drop a .replay anywhere" }
                            }
                        }
                    }
                    if show_timeline {
                        AnalysisTimeline { progress: progress.clone() }
                    } else {
                        EarlyWorkingPanel { progress: progress.clone() }
                    }
                    if let Some(prediction_results) = results {
                        PlayerSummaryGrid { results: prediction_results.clone() }
                        MatchVerdictBanner { results: prediction_results }
                    } else {
                        PlayerSummaryGridLoading { progress }
                    }
                }
            }
        };
    }

    rsx! {
        div { class: "flex flex-col items-center justify-center min-h-screen px-4 py-10 gap-6",
            h1 { class: "w-full max-w-lg text-center text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-orange-400",
                "Is this a smurf?"
            }
            // Hero art + blurb share the same width and surface treatment as the upload zone below.
            div { class: "w-full max-w-lg rounded-2xl border border-gray-700/60 bg-gradient-to-b from-gray-900/80 to-gray-950/90 p-6 shadow-xl shadow-black/40",
                div { class: "flex justify-center rounded-xl bg-gray-950/50 p-3 ring-1 ring-inset ring-gray-800/80",
                    img {
                        src: IS_THIS_A_SMURF_HERO,
                        alt: "Cartoon Rocket League cars and rainbow banner art",
                        class: "max-h-[min(42vh,28rem)] w-auto max-w-full object-contain rounded-lg",
                    }
                }
                p { class: "mt-4 text-center text-base leading-relaxed text-gray-400",
                    "Upload a Rocket League replay. We guess everyone's rank, then expose who's committed to the smurf lifestyle."
                }
            }

            // Upload zone
            label { class: "relative flex flex-col items-center justify-center w-full max-w-lg h-64 border-2 border-dashed border-gray-600 rounded-2xl cursor-pointer hover:border-blue-500 hover:bg-gray-900/50 transition-all duration-300",
                // Icon
                svg {
                    class: "w-16 h-16 mb-4 text-gray-500",
                    fill: "none",
                    stroke: "currentColor",
                    stroke_width: "1.5",
                    view_box: "0 0 24 24",
                    path {
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        d: "M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5",
                    }
                }
                p { class: "text-gray-400 text-lg font-medium",
                    "Click or drop a "
                    span { class: "text-blue-400", ".replay" }
                    " anywhere"
                }
                p { class: "text-gray-500 text-sm mt-1", "1v1, 2v2 or 3v3 Rocket League replay, ranked or casual" }

                input {
                    id: "replay-file-input",
                    r#type: "file",
                    accept: ".replay",
                    class: "absolute inset-0 w-full h-full opacity-0 cursor-pointer",
                    onchange: move |_event: Event<FormData>| {
                        // Grab the File object from the DOM before entering async.
                        let file = web_sys::window()
                            .and_then(|window| window.document())
                            .and_then(|document| document.get_element_by_id("replay-file-input"))
                            .and_then(|element| element.dyn_into::<web_sys::HtmlInputElement>().ok())
                            .and_then(|input| input.files())
                            .and_then(|file_list| file_list.get(0));
                        tracing::info!("[replay] onchange: file selected");

                        if let Some(file) = file {
                            on_replay_selected.call(file);
                        }
                    },
                }
            }
        }
    }
}

/// Pause between revealing two timeline windows. Scoring is near-instant now; the pause
/// keeps the "scanning" animation readable.
const WINDOW_REVEAL_MILLISECONDS: u32 = 220;

/// Every step done up to and including model loading.
fn progress_after_model(
    segments: Vec<SegmentStepInfo>,
    timeline: Option<TimelineTrackState>,
) -> ProgressState {
    ProgressState {
        reading_file: StepStatus::Done("Done".to_string()),
        copying_into_memory: StepStatus::Done("Done".to_string()),
        parsing: StepStatus::Done("Done".to_string()),
        loading_model: StepStatus::Done("Done".to_string()),
        segments,
        timeline,
    }
}

/// Leaves the processing view and shows an error or unsupported-replay page.
fn fail(
    mut state: Signal<AppState>,
    mut local_processing: Signal<Option<LocalProcessing>>,
    next_state: AppState,
) {
    local_processing.set(None);
    state.set(next_state);
}

/// Reads, parses and scores the replay, updating `local_processing` as each step lands.
#[expect(clippy::future_not_send)]
pub(super) async fn run_pipeline(
    file: web_sys::File,
    state: Signal<AppState>,
    mut local_processing: Signal<Option<LocalProcessing>>,
) {
    let filename = file.name();
    let mut publish = move |progress: ProgressState, results: Option<PredictionResults>| {
        local_processing.set(Some(LocalProcessing {
            filename: filename.clone(),
            progress,
            results,
        }));
    };
    let early = |reading: StepStatus, copying: StepStatus, parsing: StepStatus| ProgressState {
        reading_file: reading,
        copying_into_memory: copying,
        parsing,
        loading_model: StepStatus::Pending,
        segments: vec![],
        timeline: None,
    };
    let done = || StepStatus::Done("Done".to_string());

    // ---- Read the file bytes via web-sys (no JS eval, no base64).
    publish(
        early(
            StepStatus::Processing,
            StepStatus::Pending,
            StepStatus::Pending,
        ),
        None,
    );
    yield_to_ui().await;
    let array_buffer = match JsFuture::from(file.array_buffer()).await {
        Ok(buffer) => buffer,
        Err(error) => {
            fail(
                state,
                local_processing,
                AppState::Error(format!("Could not read file: {error:?}")),
            );
            return;
        }
    };
    publish(
        early(done(), StepStatus::Processing, StepStatus::Pending),
        None,
    );
    yield_to_ui().await;
    yield_for_dom_paint().await;
    let data: Vec<u8> = js_sys::Uint8Array::new(&array_buffer).to_vec();

    // ---- Parse.
    publish(early(done(), done(), StepStatus::Processing), None);
    yield_to_ui().await;
    yield_for_dom_paint().await;
    let parsed = match parse_replay_from_bytes(&data) {
        Ok(parsed) => parsed,
        Err(ReplayAcceptanceError::Unsupported(details)) => {
            fail(
                state,
                local_processing,
                AppState::UnsupportedReplay(details),
            );
            return;
        }
        Err(ReplayAcceptanceError::Parse(error)) => {
            fail(
                state,
                local_processing,
                AppState::Error(format!("Replay parsing error: {error}")),
            );
            return;
        }
    };
    if parsed.frames.is_empty() {
        fail(
            state,
            local_processing,
            AppState::Error("No frames found in the replay.".to_string()),
        );
        return;
    }

    // ---- Load the model and score the match (whole match, windows, roasts).
    publish(
        ProgressState {
            loading_model: StepStatus::Processing,
            ..early(done(), done(), done())
        },
        None,
    );
    yield_to_ui().await;
    yield_for_dom_paint().await;
    let bundle = match load_bundle() {
        Ok(bundle) => bundle,
        Err(message) => {
            fail(state, local_processing, AppState::Error(message));
            return;
        }
    };
    let analysis = bundle.analyze(&parsed);

    // ---- Timeline: reveal one window at a time.
    let players = prepare_players_for_timeline(&analysis);
    let mut segment_steps = segment_step_infos(&parsed.frames, &analysis.timeline);
    let timeline = (!segment_steps.is_empty()).then(|| TimelineTrackState {
        match_duration_seconds: parsed
            .frames
            .last()
            .map_or(1.0_f32, |frame| frame.time)
            .max(0.001),
        boundary_times_seconds: compute_segment_boundary_times(&segment_steps),
        boundary_play_times_seconds: compute_segment_boundary_play_times(
            &parsed.frames,
            &analysis.timeline,
        ),
        goals: build_goal_markers(&parsed, &players.names),
        player_names: players.names.clone(),
        player_teams: players.teams.clone(),
        player_slots: players.slots.clone(),
        num_segments: segment_steps.len(),
    });
    publish(
        progress_after_model(segment_steps.clone(), timeline.clone()),
        None,
    );
    yield_to_ui().await;
    yield_for_dom_paint().await;
    for (index, window) in analysis.timeline.iter().enumerate() {
        if let Some(step) = segment_steps.get_mut(index) {
            step.player_segment_ranks = Some(ranks_from_player_mmr(&window.player_mmr));
            step.status = StepStatus::Done("Complete".to_string());
        }
        publish(
            progress_after_model(segment_steps.clone(), timeline.clone()),
            None,
        );
        yield_to_ui().await;
        sleep_milliseconds(WINDOW_REVEAL_MILLISECONDS).await;
    }

    // ---- Cards and verdict.
    publish(
        progress_after_model(segment_steps, timeline),
        Some(build_prediction_results(
            &analysis,
            parsed.match_format,
            FinalScore::from_replay(&parsed),
        )),
    );
}
