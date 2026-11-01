//! Label cleaning by account rather than by game.
//!
//! One game is a noisy sample of a player: a real Silver's best game can play like a Gold,
//! and a smurf's quiet game can look like their label. Averaging the out-of-fold gap
//! (plays-like prediction minus label) over an account's games separates the two: an
//! account that plays far above its rank across its games loses every label, including
//! the games where it looked normal, while a real player's standout game keeps its label.
//! Accounts seen in fewer than [`AccountCleaning::minimum_games`] labelled games fall back
//! to the per-game rule.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;

use anyhow::{Context, Result};
use uuid::Uuid;

use crate::dataset::Lobby;
use crate::label_cleaning::{CleanedLabels, PlayerKey, PlaysLikeScores, clean_labels_where};

/// One account, as a hash of its "platform:id".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AccountId(u64);

/// Account per player, from `extract_stats`' accounts CSV.
#[derive(Debug, Default)]
pub struct AccountIds {
    pub by_player: HashMap<PlayerKey, AccountId>,
}

impl AccountIds {
    /// Reads `replay_id,slot,account` rows.
    ///
    /// # Errors
    ///
    /// When the file cannot be read or a row does not parse.
    pub fn read(path: &Path) -> Result<Self> {
        let mut reader =
            csv::Reader::from_path(path).with_context(|| format!("opening {}", path.display()))?;
        let mut accounts = Self::default();
        for record in reader.records() {
            let record = record?;
            let field = |index: usize| record.get(index).unwrap_or_default();
            let mut hasher = DefaultHasher::new();
            field(2).hash(&mut hasher);
            accounts.by_player.insert(
                PlayerKey {
                    replay_id: Uuid::parse_str(field(0))?,
                    slot: field(1).parse()?,
                },
                AccountId(hasher.finish()),
            );
        }
        Ok(accounts)
    }
}

/// How labels are cleaned by account.
#[derive(Debug, Clone, Copy)]
pub struct AccountCleaning {
    /// A label is dropped when its account's mean gap reaches this (or, for accounts with
    /// too few games, when its own gap does).
    pub margin: f32,
    /// Labelled games an account needs before its mean gap replaces the per-game rule.
    pub minimum_games: usize,
}

/// Mean gap and game count of one account.
#[derive(Debug, Default, Clone, Copy)]
struct AccountGap {
    total: f32,
    games: usize,
}

/// Cleans `lobbies` by account: see the module documentation.
#[must_use]
pub fn clean_labels_by_account(
    lobbies: &[Lobby],
    out_of_fold: &PlaysLikeScores,
    accounts: &AccountIds,
    cleaning: AccountCleaning,
) -> CleanedLabels {
    let mut gaps: HashMap<AccountId, AccountGap> = HashMap::new();
    for lobby in lobbies {
        let Some(scores) = out_of_fold.by_replay.get(&lobby.replay_id) else {
            continue;
        };
        for (slot, (target, score)) in lobby.targets.iter().zip(scores).enumerate() {
            let Some(score) = score else {
                continue;
            };
            let key = PlayerKey {
                replay_id: lobby.replay_id,
                slot,
            };
            if *target <= 0.0 {
                continue;
            }
            if let Some(account) = accounts.by_player.get(&key) {
                let gap = gaps.entry(*account).or_default();
                gap.total += score - target;
                gap.games += 1;
            }
        }
    }
    clean_labels_where(lobbies, out_of_fold, |player, own_gap| {
        let account_gap = accounts
            .by_player
            .get(&player)
            .and_then(|account| gaps.get(account))
            .filter(|gap| gap.games >= cleaning.minimum_games);
        match account_gap {
            Some(gap) => gap.total / gap.games as f32 >= cleaning.margin,
            None => own_gap >= cleaning.margin,
        }
    })
}
