//! Match-level verdict copy: one line picked from tier-specific pools (no tech wording).

use rand::{Rng, RngExt, rng};
use replay_structs::RankDivision;

use crate::app_state::PredictionResults;
use crate::prediction::lobby_median_mmr;

mod no_smurf_lines;

const SMURF_NAME_PLACEHOLDER: &str = "{name}";
const SMURF_NAMES_PLACEHOLDER: &str = "{names}";
const SMURF_SINGLE_LINE_FALLBACK: &str =
    "One player looked wildly overqualified for the nonsense happening in this lobby.";
const NO_SMURF_LINE_FALLBACK: &str =
    "This lobby stayed honest. Loud, messy, and blessedly free of smurfs.";

/// Returns the static pool of "no smurf" lines for the lobby median named rank tier.
#[must_use]
pub(crate) const fn no_smurf_lines_for_median_rank(rank: RankDivision) -> &'static [&'static str] {
    match rank {
        RankDivision::BronzeIDivision1
        | RankDivision::BronzeIDivision2
        | RankDivision::BronzeIDivision3
        | RankDivision::BronzeIDivision4 => no_smurf_lines::BRONZE_I,
        RankDivision::BronzeIIDivision1
        | RankDivision::BronzeIIDivision2
        | RankDivision::BronzeIIDivision3
        | RankDivision::BronzeIIDivision4 => no_smurf_lines::BRONZE_II,
        RankDivision::BronzeIIIDivision1
        | RankDivision::BronzeIIIDivision2
        | RankDivision::BronzeIIIDivision3
        | RankDivision::BronzeIIIDivision4 => no_smurf_lines::BRONZE_III,
        RankDivision::SilverIDivision1
        | RankDivision::SilverIDivision2
        | RankDivision::SilverIDivision3
        | RankDivision::SilverIDivision4 => no_smurf_lines::SILVER_I,
        RankDivision::SilverIIDivision1
        | RankDivision::SilverIIDivision2
        | RankDivision::SilverIIDivision3
        | RankDivision::SilverIIDivision4 => no_smurf_lines::SILVER_II,
        RankDivision::SilverIIIDivision1
        | RankDivision::SilverIIIDivision2
        | RankDivision::SilverIIIDivision3
        | RankDivision::SilverIIIDivision4 => no_smurf_lines::SILVER_III,
        RankDivision::GoldIDivision1
        | RankDivision::GoldIDivision2
        | RankDivision::GoldIDivision3
        | RankDivision::GoldIDivision4 => no_smurf_lines::GOLD_I,
        RankDivision::GoldIIDivision1
        | RankDivision::GoldIIDivision2
        | RankDivision::GoldIIDivision3
        | RankDivision::GoldIIDivision4 => no_smurf_lines::GOLD_II,
        RankDivision::GoldIIIDivision1
        | RankDivision::GoldIIIDivision2
        | RankDivision::GoldIIIDivision3
        | RankDivision::GoldIIIDivision4 => no_smurf_lines::GOLD_III,
        RankDivision::PlatinumIDivision1
        | RankDivision::PlatinumIDivision2
        | RankDivision::PlatinumIDivision3
        | RankDivision::PlatinumIDivision4 => no_smurf_lines::PLATINUM_I,
        RankDivision::PlatinumIIDivision1
        | RankDivision::PlatinumIIDivision2
        | RankDivision::PlatinumIIDivision3
        | RankDivision::PlatinumIIDivision4 => no_smurf_lines::PLATINUM_II,
        RankDivision::PlatinumIIIDivision1
        | RankDivision::PlatinumIIIDivision2
        | RankDivision::PlatinumIIIDivision3
        | RankDivision::PlatinumIIIDivision4 => no_smurf_lines::PLATINUM_III,
        RankDivision::DiamondIDivision1
        | RankDivision::DiamondIDivision2
        | RankDivision::DiamondIDivision3
        | RankDivision::DiamondIDivision4 => no_smurf_lines::DIAMOND_I,
        RankDivision::DiamondIIDivision1
        | RankDivision::DiamondIIDivision2
        | RankDivision::DiamondIIDivision3
        | RankDivision::DiamondIIDivision4 => no_smurf_lines::DIAMOND_II,
        RankDivision::DiamondIIIDivision1
        | RankDivision::DiamondIIIDivision2
        | RankDivision::DiamondIIIDivision3
        | RankDivision::DiamondIIIDivision4 => no_smurf_lines::DIAMOND_III,
        RankDivision::ChampionIDivision1
        | RankDivision::ChampionIDivision2
        | RankDivision::ChampionIDivision3
        | RankDivision::ChampionIDivision4 => no_smurf_lines::CHAMPION_I,
        RankDivision::ChampionIIDivision1
        | RankDivision::ChampionIIDivision2
        | RankDivision::ChampionIIDivision3
        | RankDivision::ChampionIIDivision4 => no_smurf_lines::CHAMPION_II,
        RankDivision::ChampionIIIDivision1
        | RankDivision::ChampionIIIDivision2
        | RankDivision::ChampionIIIDivision3
        | RankDivision::ChampionIIIDivision4 => no_smurf_lines::CHAMPION_III,
        RankDivision::GrandChampionIDivision1
        | RankDivision::GrandChampionIDivision2
        | RankDivision::GrandChampionIDivision3
        | RankDivision::GrandChampionIDivision4 => no_smurf_lines::GRAND_CHAMPION_I,
        RankDivision::GrandChampionIIDivision1
        | RankDivision::GrandChampionIIDivision2
        | RankDivision::GrandChampionIIDivision3
        | RankDivision::GrandChampionIIDivision4 => no_smurf_lines::GRAND_CHAMPION_II,
        RankDivision::GrandChampionIIIDivision1
        | RankDivision::GrandChampionIIIDivision2
        | RankDivision::GrandChampionIIIDivision3
        | RankDivision::GrandChampionIIIDivision4 => no_smurf_lines::GRAND_CHAMPION_III,
        RankDivision::SupersonicLegend => no_smurf_lines::SUPERSONIC_LEGEND,
    }
}

fn random_pool_index(rng: &mut impl Rng, pool_len: usize) -> usize {
    if pool_len == 0 {
        return 0;
    }
    rng.random_range(0..pool_len)
}

/// One fragment of the verdict line: normal copy or a smurf player name (rendered bold in the UI).
#[derive(Debug, Clone)]
pub(crate) enum VerdictSegment {
    Plain(String),
    BoldName(String),
}

fn join_suspect_name_segments(names: &[String]) -> Vec<VerdictSegment> {
    match names.len() {
        0 => Vec::new(),
        1 => vec![VerdictSegment::BoldName(
            names.first().cloned().unwrap_or_default(),
        )],
        2 => vec![
            VerdictSegment::BoldName(names.first().cloned().unwrap_or_default()),
            VerdictSegment::Plain(" and ".to_string()),
            VerdictSegment::BoldName(names.get(1).cloned().unwrap_or_default()),
        ],
        _ => {
            let mut segments: Vec<VerdictSegment> = Vec::new();
            for (index, name) in names.iter().enumerate() {
                if index > 0 {
                    if index == names.len() - 1 {
                        segments.push(VerdictSegment::Plain(", and ".to_string()));
                    } else {
                        segments.push(VerdictSegment::Plain(", ".to_string()));
                    }
                }
                segments.push(VerdictSegment::BoldName(name.clone()));
            }
            segments
        }
    }
}

fn split_single_smurf_line(template: &str, name: &str) -> Vec<VerdictSegment> {
    if !template.contains(SMURF_NAME_PLACEHOLDER) {
        return vec![VerdictSegment::Plain(template.to_string())];
    }
    let parts: Vec<&str> = template.split(SMURF_NAME_PLACEHOLDER).collect();
    let before = parts.first().copied().unwrap_or("");
    let after = parts.get(1).copied().unwrap_or("");
    vec![
        VerdictSegment::Plain(before.to_string()),
        VerdictSegment::BoldName(name.to_string()),
        VerdictSegment::Plain(after.to_string()),
    ]
}

fn split_multi_smurf_line(
    template: &str,
    name_segments: Vec<VerdictSegment>,
) -> Vec<VerdictSegment> {
    if !template.contains(SMURF_NAMES_PLACEHOLDER) {
        return vec![VerdictSegment::Plain(template.to_string())];
    }
    let parts: Vec<&str> = template.split(SMURF_NAMES_PLACEHOLDER).collect();
    let before = parts.first().copied().unwrap_or("");
    let after = parts.get(1).copied().unwrap_or("");
    let mut segments: Vec<VerdictSegment> = Vec::new();
    segments.push(VerdictSegment::Plain(before.to_string()));
    segments.extend(name_segments);
    segments.push(VerdictSegment::Plain(after.to_string()));
    segments
}

/// Single-smurf lines: use `{name}` placeholder.
const SINGLE_SMURF_LINES: &[&str] = &[
    "{name} is moving like they got lost on the way to their actual rank.",
    "{name} did not queue this lobby, they audited it.",
    "The rest of the lobby is playing ranked; {name} is filming a before-and-after ad.",
    "{name} brought main-account hands to a side-account alibi.",
    "Every time the ball finds {name}, the replay starts sounding like tax fraud.",
    "{name} has that unmistakable 'this account was made for crimes' energy.",
    "{name} is playing chess in a food court.",
    "{name} is the reason this match feels like a tutorial boss wandered into public matchmaking.",
    "{name} is handing out life lessons in a lobby that still eats crayons for boost paths.",
    "{name} looks like they installed themselves into the wrong tax bracket on purpose.",
    "{name} is what happens when matchmaking forgets to check ID at the door.",
    "{name} is farming this lobby like rent is due tonight.",
];

/// Multi-smurf lines: use `{names}` placeholder (Oxford-style list).
const MULTI_SMURF_LINES: &[&str] = &[
    "{names} did not queue for a fair fight; they queued for content.",
    "{names} showed up like substitute teachers for pain.",
    "This lobby ordered regular matchmaking and got {names} with extra fraud.",
    "{names} have that 'we swear this is just our warm-up account' chemistry.",
    "If alt accounts could carpool, they would look a lot like {names}.",
    "{names} are treating this lobby like a clip farm with public funding.",
    "The rest of the server brought cars; {names} brought an alibi.",
    "{names} arrived with matching fake mustaches and somehow less believable stories.",
    "{names} are running a group project called ruining the MMR curve.",
    "{names} turned this lobby into a scholarship program for suffering.",
    "{names} look like they share one Wi-Fi network and one terrible excuse.",
    "{names} are the reason the rest of the lobby suddenly needs a union rep.",
];

/// Segments for the verdict banner under the summary grid (smurf names are separate for bold styling).
#[must_use]
pub(crate) fn match_verdict_segments(results: &PredictionResults) -> Vec<VerdictSegment> {
    let mut rng = rng();
    let lobby_median_mmr = lobby_median_mmr(&results.player_averages);
    let median_rank = RankDivision::from(lobby_median_mmr);
    let no_smurf_pool = no_smurf_lines_for_median_rank(median_rank);

    let suspects: Vec<String> = results
        .player_averages
        .iter()
        .filter(|player| player.smurf.is_some())
        .map(|player| player.name.clone())
        .collect();

    if suspects.is_empty() {
        let variant = random_pool_index(&mut rng, no_smurf_pool.len());
        let line = no_smurf_pool
            .get(variant)
            .copied()
            .unwrap_or(NO_SMURF_LINE_FALLBACK);
        vec![VerdictSegment::Plain(line.to_string())]
    } else if suspects.len() == 1 {
        let variant = random_pool_index(&mut rng, SINGLE_SMURF_LINES.len());
        let line = SINGLE_SMURF_LINES
            .get(variant)
            .copied()
            .unwrap_or(SMURF_SINGLE_LINE_FALLBACK);
        let name = suspects.first().map_or("", String::as_str);
        split_single_smurf_line(line, name)
    } else {
        let variant = random_pool_index(&mut rng, MULTI_SMURF_LINES.len());
        let line = MULTI_SMURF_LINES
            .get(variant)
            .copied()
            .unwrap_or("{names} tilted this lobby off its usual axis.");
        let name_segments = join_suspect_name_segments(&suspects);
        split_multi_smurf_line(line, name_segments)
    }
}
