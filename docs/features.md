# What each stat is worth

[`features.csv`](features.csv) measures every per-player match stat
([`MatchStat`](../crates/replay_structs/src/match_stat.rs)) against rank on the full dataset:
123,943 labelled players with over two minutes of play (standard 92,319, doubles 20,899,
duels 10,725). This page reads it. Regenerate both after a new `extract_stats` run:

```
cargo run --release -p skill_model_training --bin feature_stats
```

## The columns

One row per stat and playlist.

| Column | Meaning |
|---|---|
| `spearman_rho` | Rank correlation with the player's MMR label: how well the stat alone orders players. Negative means lower is better. |
| `eta_squared` | Share of the stat's variance explained by the rank tier (22 tiers). The rest is spread *inside* a tier: player style and match-to-match noise. |
| `separation` | (SSL median − Bronze I median) ÷ the average within-tier IQR: how many "one rank's spread" apart the extremes are. `nan` or `0` when both medians are 0 (rare events). |
| `within_tier_iqr` | Average inter-quartile range inside one tier. |
| `p10`, `p50`, `p90` | Over all players of the playlist. |
| `nonzero_share_b1`, `nonzero_share_ssl` | Share of Bronze I and SSL players with a non-zero value: the useful view for rare mechanics, whose medians are 0. |
| `median_B1` … `median_SSL` | Median per tier. |

`spearman_rho` and `eta_squared` disagree for skewed stats: `rho` rewards a stat that orders
players consistently even if most are 0; `eta_squared` rewards big jumps between tier means.

## The short version

- **Car control is the rank signal.** Air time, aerial touches, jumping, speed, boosting,
  speed flips and landing speed kept each order players at ρ 0.6–0.8 on their own. They do
  not depend on the opponents, which is why the smurf model (`plays_like`) is built on them.
- **Rotation, positioning and outcome stats barely move with rank.** Overcommitting,
  last-back time, field thirds, kickoff and 50/50 win rates sit near ρ 0. Matchmaking puts
  equals against each other, so anything measured *against the opponents* averages out at
  every rank. They still describe a player inside one lobby.
- **Rare mechanics are zero for most players at every rank.** Flip resets, double taps,
  ceiling shots, air dribbles and redirects separate ranks by *whether* they happen
  (`nonzero_share`), not by their median.
- **Per-match noise is large.** Even the best stat leaves most of the variance inside a tier
  (η² ≤ 0.63). Ratio stats built on a handful of events per match (shot accuracy, kickoff,
  50/50 win rate) are the noisiest.

## By strength (standard; duels / doubles where they differ)

### Strong: ρ ≥ 0.5

| Stat | ρ | η² | Bronze I → Gold I → Diamond I → Champion I → SSL | Note |
|---|---|---|---|---|
| `high_air_fraction` | 0.81 | 0.63 | 2% → 3% → 6% → 8% → 12% | Best single stat in standard; 0.59 in duels. |
| `mean_height` | 0.76 | 0.55 | | The same signal as air time. |
| `aerial_touch_fraction` | 0.75 | 0.55 | 6% → 17% → 32% → 40% → 51% | |
| `speed_flips_per_minute` | 0.70 | 0.42 | 0 → 0 → 0 → 0.56 → 1.50 | 10% of Bronze I vs 99.7% of SSL. The kickoff skill shows up here. |
| `landing_speed_kept_fraction` | 0.69 | 0.47 | 0.74 → 0.86 → 0.91 → 0.93 → 0.96 | Recovery and momentum. |
| `jumps_pressed_per_minute`, `double_jumps_per_minute` | 0.68 | 0.39 | | |
| `ground_fraction` | −0.67 | 0.45 | 78% → 72% → 68% → 64% → 59% | |
| `mean_speed`, `boosting_fraction`, `supersonic_fraction` | 0.64–0.66 | 0.38–0.42 | supersonic 4% → 16% | `mean_speed` is weaker in duels (0.41). |
| `aerial_dodges_per_minute` | 0.64 | 0.15 | | |
| `air_roll_fraction` | 0.61 | 0.27 | | |
| `tornado_spins_per_minute` | 0.60 | 0.30 | 0 until Champion, 0.45 at SSL | 8% of Bronze I vs 87% of SSL. |
| `boost_while_supersonic_fraction` | 0.60 | 0.25 | | |
| `wavedash_like_dodges_per_minute` | 0.56 | 0.23 | | |
| `dodges_per_minute` | 0.55 | 0.14 | | 0.70 in duels. |
| `double_commit_fraction` | −0.55 | 0.32 | 2.2% → 0.8% | Standard and doubles only; meaningless in duels (ρ 0.01). |
| `slow_fraction` | −0.53 | 0.26 | 21% → 12% | Not in duels (ρ −0.04). |
| `power_shots_per_minute` | 0.50 | 0.23 | 0.17 → 0.38 → 0.68 → 0.75 → 0.88 | |

### Moderate: ρ 0.3–0.5

`clean_landing_fraction` (0.49; 0.83 → 0.95), `diagonal_dodge_fraction` (0.49),
`mean_distance_to_nearest_teammate` (0.48, standard only), `wavedashes_per_minute` (0.48),
`powerslide_fraction` (0.48), `touches_per_minute` (0.47), `jumps_per_minute` (0.47),
`air_dribble_touches_per_minute` (0.46; 5% of Bronze I vs 77% of SSL),
`small_pads_per_minute` (0.46; 6.0 → 11.7), `mean_absolute_steer` (0.46),
`teammate_bumps_per_minute` (−0.45), boost collected / used (0.42–0.43, but η² 0.03: huge
spread inside a rank), `mean_air_angular_speed` (0.42), `low_air_fraction` (0.37; 0.70 in
duels), `airborne_upright_fraction` (−0.37: better players spend *more* air time rotated,
so it is not a recovery measure), `pitch_positive_dodge_fraction` (−0.36: front flips give
way to diagonals), `big_pads_per_minute` (0.36), `half_flips_per_minute` (0.35; 12% of
Bronze I vs 64% of SSL), `ceiling_touches_per_minute` (0.34), `wall_reads_per_minute`
(0.32; 0.17 → 0.36 → 0.52 → 0.55 → 0.70), `mean_ball_speed_after_touch` (0.31),
`flip_resets_per_minute` (0.31; 1% of Bronze I vs 33% of SSL).

### Weak: ρ 0.1–0.3

Useful as context, for coaching a specific habit, or inside a lobby, not as a rank signal on
their own.

- **Shooting:** `scoreboard_saves`, `scoreboard_shots`, `scoreboard_score_per_minute`,
  `fast_aerial_fraction` (0.22: noisy, a fraction of few double jumps),
  `forward_touch_fraction` (0.16), `shot_accuracy` (0.08 in standard, 0.21 in duels; see
  below).
- **Defending:** `goal_side_after_touch_fraction` (0.20), `shadow_defense_fraction` (0.19,
  0.25–0.27 in duels and doubles), `goal_side_when_defending_fraction` (0.17).
- **Team play:** `passes_per_minute` (0.17 in standard, nothing in doubles),
  `redirects_per_minute` (0.12; 4% of Bronze I vs 14% of SSL).
- **Ball control:** `fifty_fifties_per_minute` (0.28), `double_taps_per_minute` (0.22),
  `dribble_fraction` (0.12 in standard, 0.37 in duels).
- **Other:** `kickoff_approach_speed` (0.16), `demos_inflicted` / `demos_received`,
  `wall_touch_fraction`, `possession_fraction`, `supersonic_per_boost`, `reverse_fraction`
  (−0.19), `slow_with_boost_fraction` (−0.21), `fast_dodge_fraction` (−0.27),
  `mean_touch_speed_gain` (−0.28), `wall_fraction`.

### No rank signal: |ρ| < 0.1

`overcommit_fraction`, `kickoff_first_touch_fraction`, `kickoff_win_fraction`,
`fifty_fifty_win_fraction`, `boost_full_fraction`, `own_goals`, `offensive_third_fraction`,
`forfeited_level_or_ahead`, `scoreboard_goals_per_shot`, `flip_touch_fraction`, `goals`
(0.37 in duels), `last_back_goal_distance_when_defending`, `last_back_on_team_fraction`,
`most_forward_on_team_fraction`, `behind_ball_fraction` (0.38 in duels),
`last_back_when_defending_fraction`, `lost_by_forfeit`, `roll_dodge_fraction`,
`scoreboard_assists`, `bumps_inflicted_per_minute`, `bumps_received_per_minute`,
`mean_canonical_y`, `mean_distance_to_ball`, `flicks_per_minute` (0.31 in duels),
`mean_boost`, `closest_on_team_fraction`, `pitch_negative_dodge_fraction`,
`mean_distance_to_own_goal`.

These are kept anyway: the models see them relative to the lobby (own − lobby mean), where
a smurf stands out, and some carry a flag on their own (forfeits).

## The skills players are told to learn

| Skill | Best stat | ρ (standard) | Verdict |
|---|---|---|---|
| Powershots | `power_shots_per_minute` | 0.50 | Strong. "Where you want it" is `shot_accuracy`: Bronze I 25% on target, Gold I 40%, SSL 46%, but too few shots per match to order players (ρ 0.08). Better summed over several matches, or for a roast after a clearly bad game. |
| Fast recoveries | `landing_speed_kept_fraction` | 0.69 | Strong; `clean_landing_fraction` 0.49. |
| Half flip | `half_flips_per_minute` | 0.35 | Moderate; weaker in duels (0.10). |
| Fast aerial | `aerial_touch_fraction` / `fast_aerial_fraction` | 0.75 / 0.22 | Reaching high balls is strong; the take-off measure is noisy. |
| Kickoffs (speed flip) | `speed_flips_per_minute` | 0.70 | Strong through speed flips. `kickoff_approach_speed` is weak (0.16) and win rates are 0. |
| Shadow defense | `shadow_defense_fraction` | 0.19 | Weak; 0.25–0.27 in duels and doubles. |
| Aerial car control | `air_roll_fraction`, `tornado_spins_per_minute` | 0.61, 0.60 | Strong. |
| Powerslides and boost pathing | `powerslide_fraction`, `small_pads_per_minute`, `supersonic_fraction` | 0.48, 0.46, 0.64 | Strong. |
| Backboard and wall reads | `wall_reads_per_minute` | 0.32 | Moderate. |
| 50/50 challenges | `fifty_fifties_per_minute` | 0.28 | Weak; the win rate is always about 50%. |
| Ground dribbling | `dribble_fraction` | 0.12 | Weak: carries are rare in ranked standard (median 0.04% of play at SSL). 0.37 in duels. |
| Basic flicks | `flicks_per_minute` | 0.09 | Weak in standard (5% of SSL flick), 0.31 in duels. |
| Wavedash | `wavedash_like_dodges_per_minute` | 0.56 | Strong. |
| Bumps and demos | `demos_inflicted` | 0.19 | Weak; bumps have no rank signal. |
| Passing and redirects | `passes_per_minute`, `redirects_per_minute` | 0.17, 0.12 | Weak. |
| Air dribbles | `air_dribble_touches_per_minute` | 0.46 | Moderate; rare below Grand Champion. |
| Double taps | `double_taps_per_minute` | 0.22 | Weak; 24% of SSL. |
| Ceiling shots | `ceiling_touches_per_minute` | 0.34 | Moderate; counts ceiling touches, not the shot. |
| Flip resets | `flip_resets_per_minute` | 0.31 | Moderate; 33% of SSL. |

## Caveats

- **The label is the player's rank, not their play.** Smurfs and boosted accounts sit in the
  wrong tier and pull every correlation down a little. Training cleans the worst of them;
  this table does not.
- **One match per row.** A player's average over many matches would show stronger
  correlations, especially for event-count stats.
- **Medians of 0.** For zero-inflated stats, `separation` is `nan` or `0` and the nonzero
  shares are the useful columns.
- **Playlists differ.** Team-play stats (double commits, passes, teammate spacing) mean
  nothing in duels. Air time separates ranks less in duels, where control on the ground
  matters more.
