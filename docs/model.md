# The skill model

How "Is this a smurf?" turns a replay into ranks, a timeline and a roast, how to rebuild it,
and what we learned getting here.

## What ships

One file, `data/skill_model.bin` (~1.4 MB, see [Bundle format](#bundle-format)), embedded in the app. It holds:

| Part | Input | Output | Used for |
|---|---|---|---|
| **Match model** | each player's whole-match stats | one MMR per player | rank cards, smurf badge, verdict |
| **Window model** | the same stats over 60 s of live play | one MMR per player per minute | the timeline |
| **Coaching table** | per-rank medians of 19 actionable stats | one roast per player | the line under each card |

Everything runs in the browser in a fraction of a second; there is no neural network.

### 1. Per-player stats (`crates/feature_extractor`)

`compute_player_match_stats` reads every live frame (goal replays skipped) and summarises each
player in **71 numbers**, all measured from that player's own side of the field, plus the
playlist's team size (`players_per_team`, see [Playlists](#playlists)):

- **movement** — speed, supersonic / slow time, ground / air / wall time, jumps
- **boost** — average, time empty or full, pads per minute, boosting while supersonic
- **positioning** — field thirds, last back, behind the ball, distance to ball / teammates / goal
- **ball** — possession, touches per minute, ball speed after a touch, aerial and forward touches
- **mechanics** (from controller inputs) — flips, wavedash-like and aerial flips, flip
  direction, fast aerials, powerslide, air roll, air control
- **scoreboard** — in-game score, assists, saves, shots
- **plays** — demos inflicted, kickoff first touches and wins, double commits
- **situational positioning** — last back / goal-side *while defending*, overcommitting,
  shadow defence, support spacing, getting back after a touch

### 2. Gradient-boosted trees (`crates/skill_model`, trained by `crates/skill_model_training`)

A tree asks yes/no questions about the stats ("did this lobby spend more than 6 % of the game
high in the air?") and ends in a number. Training adds ~2,000 small trees one at a time, each
correcting what the previous ones still get wrong, and stops when a held-out 10 % of training
replays stops improving.

Each model is really two ensembles:

- **absolute** — predicts every player's MMR from their stats, the lobby's average stats and
  the difference; its lobby average is the **lobby level**;
- **within** — predicts only *how far each player is from their lobby*, from differences and
  team context (vs teammates, team vs opponents, goal difference).

**Prediction = lobby level + calibrated gap to the lobby.** Splitting the two matters: account
rank barely varies inside a lobby (≈ 60 MMR), and a single model spends almost all its effort
on the lobby level.

### 3. Timeline

Per-minute *lobby levels* are noisy (whole lobbies jump ±100 MMR from one minute to the next),
so the timeline never shows them. Each window shows the player's **whole-match rank, moved by
how much better or worse than their own average they did that minute**, scaled by
`timeline_emphasis` (≈ ×5) so a typical swing reads as about one tier. It is a form readout,
not a second rank estimate; the cards always come from the match model.

### 4. Roast (`crates/skill_model/src/coaching.rs`)

Each player is compared with the **median player two divisions up** (Diamond I → Diamond III,
Champion III → Grand Champion II): from Diamond up a whole named rank is several hundred MMR,
too far to be the next step, while the next division's median is only a few wins away from a
player at the top of their own division. The roast picks the stat where they fall furthest
short of that division, measured in its own spread, among stats that (a) actually improve
from their tier to three tiers above (neighbouring divisions differ too little to tell what
separates ranks from noise; judged on one division, a third to half of players got no tip),
and (b) show a gap a reader would notice ("0 % vs 1 %" never qualifies). A tip is disabled
automatically if the data says its "better" direction is wrong.
Lines and tips live in code; editing them needs no retraining.

## Playlists

One model scores **duels, doubles and standard**, ranked or casual.

- **Team size is a stat.** Every player carries `players_per_team` (1, 2 or 3), so the trees
  can treat a duel's 30 touches a minute differently from a standard game's 10, while
  mechanics and positioning learned from the larger 3v3 set carry over.
- **Labels share one scale.** A label is the division's middle on the 3v3 MMR scale, so the
  model predicts a *rank*, and "Platinum II" means Platinum II in the replay's own playlist.
- **Casual** replays are scored as the competitive playlist of the same size. Players can
  join and leave mid-match, so the roster has [`SLOTS_PER_TEAM`] = 5 slots per team; players
  with under 30 s of live play (`MINIMUM_MATCH_SECONDS`) are not scored and do not count
  towards the lobby, in training and in the app alike.
- **Roasts** use one coaching table per playlist size: the next rank up in duels does
  different things than in standard.

### How much data each playlist needs

The 3v3 download target (1,200 per rank, ~27k replays) was sized for the LSTM. The
`learning_curve` binary trains the match model on growing prefixes of the 3v3 training
replays and scores each on the full evaluation split:

| Training replays | player RMSE | lobby RMSE | within_r |
|---|---|---|---|
| 1,000 | 129 | 118 | 0.39 |
| 2,000 | 122 | 111 | 0.44 |
| 4,000 | 117 | 105 | 0.48 |
| 8,000 | 112 | 100 | 0.50 |
| 16,000 | 110 | 98 | 0.52 |
| 24,450 | 107 | 95 | 0.52 |

Past ~8,000 replays each doubling buys 2–3 MMR, and duels and doubles also borrow from the
3v3 data, so they target **350 replays per rank** (~7,700 per playlist). Simulating doubles by
keeping two players per team gives the same shape. Ballchasing allows ~200 replay downloads
an hour, so the two playlists take about three days to download.

[`SLOTS_PER_TEAM`]: ../crates/feature_extractor/src/lib.rs

## Bundle format

`crates/skill_model/src/encoding.rs`. ~500k tree nodes dominate the size, so each ensemble is
stored as the least information that rebuilds it, in separate streams of like data:

- **topology** — 1 bit per node in preorder (split or leaf); child indexes are implied;
- **split columns** — `u16` per split;
- **thresholds** — `u8` per split, indexing a per-column table of the exact `f32` cut values
  (the trainer only ever splits on ≤ 256 bin boundaries per column), so splits are bit-exact;
- **leaf values** — `i16` with one `f32` scale per tree.

The whole payload is then brotli-compressed (quality 11). Result: 1.38 MB, against 4.1 MB for
fixed 8-byte node records and ~20 MB as JSON. Quantising leaves moves predictions by at most
~0.01 MMR; `train` checks the reloaded file and fails above 0.5 MMR. The app only links the
brotli *decoder*; the encoder is behind `skill_model`'s `encode` feature, used by training.

## Rebuilding

```bash
# Parse every downloaded replay into data/match_stats.csv and data/window_stats.csv (~5 min)
cargo run --release -p skill_model_training --bin extract_stats

# Train both models, build the roast tables, write and verify data/skill_model.bin (~15 min)
cargo run --release -p skill_model_training --bin train

# How accuracy grows with data (optionally --players-per-team 1|2, --keep-players-per-team)
cargo run --release -p skill_model_training --bin learning_curve

# Only iterate on roast lines (no retraining)
cargo run --release -p skill_model_training --bin train -- --reuse-bundle

# End-to-end check on the sample replays (prints what the app would show)
cargo test --release -p skill_model_training --test analyze_replays -- --nocapture
```

Any change to the stats in `feature_extractor` means re-extracting and retraining; the app
refuses a bundle trained on a different stat list.

## How good is it

Measured on the database's **evaluation split** (3,013 replays, ~7,400 labelled players),
which no part of training sees. Labels are account ranks at match time.

| Metric | Meaning | Value |
|---|---|---|
| player RMSE | whole-match error per player | **~106 MMR** |
| lobby RMSE | error of the lobby's average | **~95 MMR** |
| within_r | correlation of predicted vs true *gaps to the lobby* (0 = no signal) | **~0.56** |
| concordance | share of lobbymate pairs (>25 MMR apart) ordered correctly (0.5 = chance) | **~0.65** |

The smurf badge fires when a player is **100 MMR above the lobby's median prediction**
(`skill_model::SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR`). Against players ranked ≥150 above their
lobby (mostly parties with a stronger friend: smurf-like lobbies where the truth is known) it
flags ~0.7 % of players with **~60 % precision**. Raising the margin trades recall for
precision (+150: ~80 %, +200: ~90 %); `train` prints the whole table.

Expect **±0.005 within_r** from the trees' random seed alone, so compare changes over a few
seeds before believing a small gain.

## What we learned

The project spent most of 2026 on an LSTM over 20-second windows of raw car positions. The
tree model replaced it in October 2026. What is worth remembering:

- **Whole-match stats beat raw sequences by a wide margin.** On identical players the best
  LSTM reached within_r 0.27 / player RMSE 169 after 24 GPU-hours; 38 hand-written stats and
  20 seconds of tree training reached 0.49 / 113. One minute of stats (0.40) already beat the
  LSTM's whole match. Adding the LSTM's output to the trees added nothing.
- **Account rank is a noisy label for "how well you played".** Within a lobby it varies by
  ~60 MMR, and a smurf's label is wrong by definition. Gains in ordering lobbymates come slowly
  and need large evaluation sets; single runs are noisy (the LSTM's seed noise was 5× its
  measurement error).
- **Data bugs hid behind the model.** The parser ignored the old boost attribute (75 % of
  bronze-1 replays had boost stuck at 0, making "no boost" a bronze marker), ignored
  `DemolishExtended` (most demolitions invisible), and some replays flip a player's team
  attribute mid-match. All fixed; when a result looks too good, check the data first.
- **Positioning matters and the labels agree.** Among mechanically strong players, those who
  overcommit sit ~12 MMR lower relative to their lobby than disciplined ones, and the model
  reproduces it.
- **Score the whole match, show the windows.** Averaging per-window predictions loses ~0.08
  within_r against scoring the whole match directly.

## Ideas not tried yet

- More data: the learning curve was still rising at 24k training lobbies.
- Make the parser's car/component linking deterministic (it iterates `HashMap`s, so repeated
  extractions differ by ~1 % on boost/jump counts).
- Store Ballchasing platform ids, which would allow real smurf labels from account history.
