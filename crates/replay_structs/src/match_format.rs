//! Which kind of soccar match a replay is: team size and whether it was ranked.

use strum::IntoEnumIterator;

/// The soccar team sizes the models serve, one playlist each.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, strum::EnumIter, strum::Display, strum::IntoStaticStr,
)]
#[strum(serialize_all = "lowercase")]
pub enum TeamSize {
    Duels,
    Doubles,
    Standard,
}

impl TeamSize {
    /// Players per team: 1, 2 or 3.
    #[must_use]
    pub const fn players_per_team(self) -> usize {
        match self {
            Self::Duels => 1,
            Self::Doubles => 2,
            Self::Standard => 3,
        }
    }

    /// Lowercase playlist name: "duels", "doubles" or "standard".
    #[must_use]
    pub fn name(self) -> &'static str {
        self.into()
    }

    /// Every team size, duels first.
    pub fn all() -> impl Iterator<Item = Self> {
        Self::iter()
    }
}

/// Team size and queue of a supported replay.
///
/// The app always predicts the **competitive** rank for the team size, also for casual
/// matches; casual lobbies can have players joining and leaving mid-match, so a team can
/// field more (or fewer) players than [`Self::players_per_team`] over the whole replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatchFormat {
    /// Players per team at any moment: 1 (duels), 2 (doubles) or 3 (standard).
    pub players_per_team: usize,
    /// Competitive (ranked) queue, as opposed to casual.
    pub ranked: bool,
}

impl MatchFormat {
    /// Ranked 3v3, the format most of the training data comes from.
    pub const RANKED_STANDARD: Self = Self {
        players_per_team: 3,
        ranked: true,
    };

    /// Short label, e.g. "ranked 2v2" or "casual 1v1".
    #[must_use]
    pub fn label(self) -> String {
        let queue = if self.ranked { "ranked" } else { "casual" };
        format!("{queue} {size}v{size}", size = self.players_per_team)
    }
}

impl Default for MatchFormat {
    fn default() -> Self {
        Self::RANKED_STANDARD
    }
}
