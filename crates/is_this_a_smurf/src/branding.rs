//! Hero and badge images for the smurf-detector branding.
//!
//! Assets are registered with [`manganis::asset!`] so the Dioxus CLI copies them into the build.

#![expect(clippy::volatile_composites)] // from manganis `asset!` macro

use manganis::{Asset, AssetOptions, ImageFormat, ImageSize, asset};

/// The project's source code, linked from every page.
pub(crate) const REPOSITORY_URL: &str = "https://github.com/bispribbit/is_this_a_smurf";

// Source asset is 1024×1536 (2:3 portrait). Encoding must keep the same aspect ratio or the
// output looks stretched. 480×720 preserves 2:3.
const HERO_WIDTH_PX: u32 = 480;
const HERO_HEIGHT_PX: u32 = 720;

/// Front-page hero image (`assets/is_this_a_smurf.png`).
pub(crate) const IS_THIS_A_SMURF_HERO: Asset = asset!(
    "/assets/is_this_a_smurf.png",
    AssetOptions::image()
        .with_size(ImageSize::Manual {
            width: HERO_WIDTH_PX,
            height: HERO_HEIGHT_PX,
        })
        .with_format(ImageFormat::Png)
);

// Source is 1024×1536 (2:3); keep ratio so the car art is not squashed.
const SMURF_BADGE_WIDTH_PX: u32 = 320;
const SMURF_BADGE_HEIGHT_PX: u32 = 480;

/// Badge when a player is far above the lobby median MMR (`assets/smurf.png`).
pub(crate) const SMURF_SUSPECT_BADGE: Asset = asset!(
    "/assets/smurf.png",
    AssetOptions::image()
        .with_size(ImageSize::Manual {
            width: SMURF_BADGE_WIDTH_PX,
            height: SMURF_BADGE_HEIGHT_PX,
        })
        .with_format(ImageFormat::Png)
);
