//! Root layout and route switch for [`crate::app_state::AppState`].

use dioxus::core::Task;
use dioxus::prelude::*;

use super::results::{ErrorPage, UnsupportedReplayPage};
use super::upload::{UploadPage, run_pipeline};
use crate::app_state::{AppState, LocalProcessing};
use crate::branding::{REPOSITORY_URL, SMURF_SUSPECT_BADGE};

/// Root application component.
///
/// Owns the analysis pipeline so a replay dropped anywhere, on any screen, starts a fresh
/// analysis. The pipeline future runs on **this** scope (always mounted), and starting a new
/// one cancels the previous run.
#[component]
#[expect(clippy::volatile_composites)] // from dioxus asset! macro
pub(crate) fn App() -> Element {
    let mut state = use_signal(|| AppState::WaitingForUpload);
    let mut local_processing = use_signal(|| None::<LocalProcessing>);
    let mut pipeline_task = use_signal(|| None::<Task>);
    // `dragenter` / `dragleave` fire for every child element crossed, so count the nesting
    // depth instead of toggling a boolean.
    let mut drag_depth = use_signal(|| 0_u32);

    let analyze_replay = use_callback(move |file: web_sys::File| {
        if let Some(task) = pipeline_task.take() {
            task.cancel();
        }
        local_processing.set(None);
        state.set(AppState::WaitingForUpload);
        pipeline_task.set(Some(spawn(run_pipeline(file, state, local_processing))));
    });

    rsx! {
        document::Stylesheet { href: asset!("/assets/tailwind.css") }
        document::Link {
            rel: "icon",
            r#type: Some("image/png".to_string()),
            href: Some(SMURF_SUSPECT_BADGE.into()),
        }

        div {
            class: "relative min-h-screen bg-gray-950 text-gray-100",
            ondragenter: move |event: Event<DragData>| {
                if dragged_files_present(&event) {
                    event.prevent_default();
                    drag_depth += 1;
                }
            },
            ondragover: move |event: Event<DragData>| {
                // Required for the browser to allow a drop instead of opening the file.
                if dragged_files_present(&event) {
                    event.prevent_default();
                }
            },
            ondragleave: move |_event: Event<DragData>| {
                let depth = drag_depth();
                drag_depth.set(depth.saturating_sub(1));
            },
            ondrop: move |event: Event<DragData>| {
                event.prevent_default();
                drag_depth.set(0);
                if let Some(file) = first_dropped_file(&event) {
                    tracing::info!("[replay] drop: file received");
                    analyze_replay.call(file);
                }
            },
            match state() {
                AppState::WaitingForUpload => rsx! {
                    UploadPage { local_processing, on_replay_selected: analyze_replay }
                },
                AppState::Error(message) => rsx! {
                    ErrorPage { message, state }
                },
                AppState::UnsupportedReplay(details) => rsx! {
                    UnsupportedReplayPage { details, state }
                },
            }
            RepositoryLink {}
            if drag_depth() > 0 {
                DropOverlay {}
            }
        }
    }
}

/// GitHub mark in the top-right corner, linking to the source code.
#[component]
fn RepositoryLink() -> Element {
    rsx! {
        a {
            class: "fixed top-4 right-4 z-40 text-gray-500 hover:text-gray-100 transition-colors",
            href: REPOSITORY_URL,
            target: "_blank",
            rel: "noopener noreferrer",
            title: "Source code on GitHub",
            aria_label: "Source code on GitHub",
            svg {
                class: "h-7 w-7",
                view_box: "0 0 16 16",
                fill: "currentColor",
                "aria-hidden": "true",
                path { d: "M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z" }
            }
        }
    }
}

/// Full-screen hint shown while a file is dragged over the page.
#[component]
fn DropOverlay() -> Element {
    rsx! {
        div { class: "pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-gray-950/80 backdrop-blur-sm p-4",
            div { class: "flex w-full max-w-lg flex-col items-center justify-center h-64 rounded-2xl border-2 border-dashed border-blue-500 bg-gray-900/80",
                p { class: "text-gray-200 text-lg font-medium",
                    "Drop the "
                    span { class: "text-blue-400", ".replay" }
                    " to analyze it"
                }
            }
        }
    }
}

/// The underlying browser `DragEvent`, when running on the web renderer.
fn browser_drag_event(event: &Event<DragData>) -> Option<web_sys::DragEvent> {
    // dioxus-web stores the raw `web_sys::DragEvent` behind `DragData`.
    event.data().downcast::<web_sys::DragEvent>().cloned()
}

/// Whether the drag carries files (ignores dragged text, links and images from the page).
fn dragged_files_present(event: &Event<DragData>) -> bool {
    browser_drag_event(event)
        .and_then(|drag_event| drag_event.data_transfer())
        .is_some_and(|data_transfer| {
            data_transfer
                .types()
                .iter()
                .any(|kind| kind.as_string().as_deref() == Some("Files"))
        })
}

/// First file of a drop, if any.
fn first_dropped_file(event: &Event<DragData>) -> Option<web_sys::File> {
    browser_drag_event(event)
        .and_then(|drag_event| drag_event.data_transfer())
        .and_then(|data_transfer| data_transfer.files())
        .and_then(|file_list| file_list.get(0))
}
