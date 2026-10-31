# The skill model

How "Is this a smurf?" turns a replay into ranks, a timeline and a roast, how to rebuild it,
and what we learned getting here.

## What ships

One file, `data/skill_model.bin` (~2.1 MB, see [Bundle format](#bundle-format)), embedded in the app. It holds:

| Part | Input | Output | Used for |
|---|---|---|---|
| **Match model** | each player's whole-match stats | one MMR per player | rank cards, smurf badge, verdict |
| **Plays-like model** | each player's own driving, boost and mechanics stats | the rank they play like | smurf badge, verdict |
| **Window model** | the same stats over 60 s of live play | one MMR per player per minute | the timeline |
| **Coaching table** | per-rank medians of 19 actionable stats | one roast per player | the line under each card |

Everything runs in the browser in a fraction of a second; there is no neural network.

### 1. Per-player stats (`crates/feature_extractor`)

`compute_player_match_stats` reads every live frame (goal replays skipped) and summarises each
player in **88 numbers**, all measured from that player's own side of the field, plus the
playlist's team size (`players_per_team`, see [Playlists](#playlists)):

- **movement** — speed, supersonic / slow time, ground / air / wall time, jumps
- **boost** — average, time empty or full, pads per minute, boosting while supersonic
- **positioning** — field thirds, last back, behind the ball, distance to ball / teammates / goal
- **ball** — possession, touches per minute, ball speed after a touch, aerial and forward touches
- **mechanics** (from controller inputs) — flips, wavedash-like and aerial flips, flip
  direction, fast aerials, powerslide, air roll, air control
- **scoreboard** — in-game score, assists, saves, shots
- **named mechanics** — speed flips, half flips, wavedashes, flip resets, ceiling touches,
  dribbles and flicks, air dribbles, double taps, wall and flip touches, power shots,
  50/50s and their win rate, bumps inflicted / received, teammate bumps
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

### 5. Smurf flag (`crates/skill_model/src/smurf.rs`)

`smurf_suspicions` is the one rule the badge, the verdict and the evaluation tools share. A
player is flagged when either:

- **ranked above the lobby**: their match-model prediction is more than 100 MMR over the
  lobby's median (`SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR`). It finds parties carrying a weaker
  friend in standard, and almost never fires in duels or doubles: the match model learns
  gaps from rank labels, and labels hardly ever differ inside those lobbies;
- **plays above the others**: the rank they play like is more than five divisions (doubles)
  or four (duels, standard) over the median of the *other* players, each of them scored
  half by the rank they play like and half by the match model
  (`smurf_margin_over_others_tiers`). The card then says "Plays like Champion II". The
  margins favour recall: a smurf detector and coach would rather flag a strong player than
  miss a smurf.

The plays-like model is one ensemble on the player's own values of 46 stats that do not
depend on the opponents (`OPPONENT_INDEPENDENT_STATS`: speed, boost, pads, jumps, flips,
aerials, air roll, shot speed, speed flips, half flips, wavedashes, flip resets, ceiling
touches, ...). Possession, touches, field position, scoreboard,
kickoffs and the situational positioning stats are left out: a smurf keeps the ball in the
other half all game, and in a duel those stats are a split with one opponent. It trains on
labels cleaned twice ([section 6](#6-label-cleaning-cratesskill_model_trainingsrclabel_cleaningrs)).
The choices, each measured with `smurf_experiment`:

- **opponent-independent stats** catch more smurfs at the same flag rate than all stats,
  including smurfs picked out only by stats the model never sees (scoreline, possession);
- **divisions, not MMR**: divisions are several times wider at the top, and with an MMR
  margin a third of the flags landed on Diamond-and-up players;
- **against the others, not the lobby**: the lobby median includes the player, so in a duel
  it sits half way to the smurf and halved their margin;
- **the others half by the match model**: one opponent's off game made a normal Diamond
  duelist look five divisions better. The match model reads the whole lobby, so blending it
  into the others' score halved the duels flags on Diamond-and-up players, which let the
  duels margin drop from five to four. Doubles keeps five: at four it flags ~8 % of players
  for two more smurfs.

### 6. Label cleaning (`crates/skill_model_training/src/label_cleaning.rs`)

A smurf's label is the smurf account's rank, so every Champion on a Bronze account teaches
the models that Champion stats happen in Bronze (about a tenth of Bronze I duelists play
like Champions). `train` first scores every training player with a model on all their own
stats that never saw their replay (two folds; a better rank estimate than the
opponent-independent plays-like model, which cleaned too much), drops labels at least 300 MMR below that score
(`CLEAN_MARGIN_MMR`; 3.5 % of standard labels, 5 % of doubles, 6 % of duels, Bronze I most
of all), and trains everything else (match, window and plays-like models, coaching tables)
on what is left. Evaluation labels are never touched.

The plays-like model gets a second round (`PLAYS_LIKE_CLEANING_ROUNDS`): the first round's
models learned from the smurf labels they hunt, so they see some smurfing as normal Bronze
play. Re-scoring with models trained on the cleaned labels drops another 3 % of duels
labels, 2 % of doubles and 1.4 % of standard, and the flag catches up to twice as many
mechanical smurfs. The match model keeps one round (`MATCH_MODEL_CLEANING_ROUNDS`): with two,
duels within_r fell from 0.27 to 0.20 and concordance from 0.64 to 0.60 (doubles and
standard barely moved), probably because the second round also drops real low-rank players
on their best games.

## Playlists

One model scores **duels, doubles and standard**, ranked or casual.

- **Team size is a stat.** Every player carries `players_per_team` (1, 2 or 3), so the trees
  can treat a duel's 30 touches a minute differently from a standard game's 10, while
  mechanics and positioning learned from the larger 3v3 set carry over.
- **Labels share one scale.** A label is the division's middle on the 3v3 MMR scale, so the
  model predicts a *rank*, and "Platinum II" means Platinum II in the replay's own playlist.
  The same player is ranked lower in duels than in standard; the trees learn that from
  `players_per_team` (no playlist is biased as a whole: mean error +3, -5 and 0 MMR).
- **One model, not three.** `compare_playlist_models` trains a joint model and one model per
  playlist over three seeds: they land within ~4 MMR of each other everywhere (joint better
  in duels and standard, separate in doubles), so one model ships.
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

# One model for every playlist vs one per playlist
cargo run --release -p skill_model_training --bin compare_playlist_models

# Flagged smurfs in a playlist's evaluation replays (1 duels, 2 doubles, 3 standard)
cargo run --release -p skill_model_training --bin find_smurf_examples -- --players-per-team 1

# How the smurf flag and label cleaning were chosen
cargo run --release -p skill_model_training --bin smurf_experiment

# Smurf flag from reference-rule label drops, per-stat-group models and their weighted mix
cargo run --release -p skill_model_training --bin smurf_signals_experiment

# Whole-match model with and without the named-mechanics stats, over seeds
cargo run --release -p skill_model_training --bin compare_stat_sets

# Only iterate on roast lines (no retraining)
cargo run --release -p skill_model_training --bin train -- --reuse-bundle

# End-to-end check on the sample replays (prints what the app would show)
cargo test --release -p skill_model_training --test analyze_replays -- --nocapture
```

Any change to the stats in `feature_extractor` means re-extracting and retraining; the app
refuses a bundle trained on a different stat list.

## How good is it

Measured on the database's **evaluation split** (763 duels, 793 doubles and 3,002 standard
replays), which no part of training sees. Labels are account ranks at match time.

| Metric | Meaning | Duels | Doubles | Standard |
|---|---|---|---|---|
| player RMSE | whole-match error per player | 178 | 113 | **102** |
| lobby RMSE | error of the lobby's average | 177 | 104 | **89** |
| within_r | correlation of predicted vs true *gaps to the lobby* (0 = no signal) | 0.21 | 0.37 | **0.53** |
| concordance | share of lobbymate pairs (>25 MMR apart) ordered correctly (0.5 = chance) | 0.61 | 0.71 | **0.66** |

Duels and doubles are scored on 400 to 800 players, so their within_r carries ±0.05; over
three seeds duels within_r is 0.23 ± 0.05 (`compare_stat_sets`).

The smurf flag ([section 5](#5-smurf-flag-cratesskill_modelsrcsmurfrs)) fires on 5.8 % of
duels players, 6.5 % of doubles and 4.9 % of standard. Labels cannot confirm a real smurf
(their label is the smurf account's), so it was chosen with `smurf_experiment` against two
references:

| Reference (labelled Gold II or lower) | Duels | Doubles | Standard |
|---|---|---|---|
| **mechanical**: Champion I mechanics on most aerial stats, team won by 2+ | 5/14 | 11/16 | 10/14 |
| **dominant**: team won by 3+, most possession and touches in the lobby | 16/61 | 17/40 | 14/46 |
| flags on players labelled Diamond or higher | 14 % | 5 % | 8 % |

With one cleaning round and no blend, the same margins caught 2/14, 7/16 and 10/14
mechanical smurfs, 9/61, 13/40 and 11/46 dominant ones, and put 17 %, 8 % and 9 % of the
flags on Diamond-and-up players.

The mechanical set favours the plays-like model (it sees those stats); the dominant set is
built only from stats it never sees. Before the plays-like rule the flag caught none of
either in any playlist. Parties (players ranked ≥150 over their lobby) are the ranked-above
rule's job: it flags 0.4 % of players with ~57 % precision (`train` prints its margin table).

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
- **A "gap to the lobby" model cannot see smurfs where labels never differ.** 90 % of ranked
  duels have both players within 60 MMR, so the match model learned that duels have no
  gaps: a Champion-level player 5-0 up on a Bronze I scored 141 against their opponent's 117.
  Scoring each player from their own stats alone fixed that in doubles and standard.
- **Margins belong in divisions, not MMR.** Bronze divisions are ~60 MMR wide, Champion
  ones ~120: a fixed MMR margin is several ranks at the bottom and barely one at the top.
- **Labels the model cannot reach distort everything.** SSL used to be labelled 2,200,
  above its own listed range (1,883 to 2,000); the trees predicted SSLs 400 to 600 MMR lower
  and those few players dominated the duels error. Labelling SSL at 1,941 cut duels lobby
  RMSE from 196 to 170.

## Ideas not tried yet

- More data: the learning curve was still rising at 24k training lobbies.
- Make the parser's car/component linking deterministic (it iterates `HashMap`s, so repeated
  extractions differ by ~1 % on boost/jump counts).
- Store Ballchasing platform ids, which would allow real smurf labels from account history.
- **Ship the weighted stat groups as the smurf flag.** `smurf_signals_experiment` trains one
  plays-like model per stat group (aerial, ground and flips, speed and boost, named
  mechanics) and averages their margins, each weighted by 1 / RMSE². At the shipped flag
  rate it catches more dominant smurfs in every playlist (duels 16 → 18 of 61, doubles
  17 → 25 of 40, standard 14 → 17 of 46) and puts far fewer flags on Diamond-and-up
  players (duels 14 % → 4 %). Needs the group models in the bundle and margins re-chosen
  for the averaged score.
- Dropping the mechanical reference's labels from training (on top of cleaning) helps the
  flag on the dominant yardstick, but moved duels rank metrics both ways between runs.
- The dribble detector finds the ball on the roof 0.3 % of the time at every rank, so it
  (and flicks, built on it) carries no signal yet. Every other named mechanic rises with
  rank except bumps, which every rank does about as often; teammate bumps fall 5× from
  Bronze to SSL.
