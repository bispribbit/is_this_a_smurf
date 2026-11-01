//! Per-player, whole-match summary statistics.
//!
//! These are the inputs of the skill model (see `docs/model.md`): 94 hand-written stats per
//! player, covering movement, boost, positioning, ball control, mechanics (speed flips,
//! half flips, wavedashes, tornado spins, flip resets, dribbles, double taps, 50/50s, bumps, ...), forfeits and own goals, the scoreboard
//! and situational positioning, plus the playlist's team size.
//!
//! Everything positional is measured in the **team-canonical frame** (own goal at `−y`),
//! so a blue and an orange player doing the same thing get the same numbers. Time-based
//! stats are weighted by frame duration and exclude goal-replay frames, exactly as the
//! sequence features do.

use std::collections::HashMap;
use std::ops::Range;

use replay_structs::{
    BallState, MATCH_STAT_COUNT, MatchStat, ParsedReplay, Quaternion, Team, Vector3,
};

use crate::{PlayerRoster, SLOTS_PER_TEAM, TOTAL_SLOTS, build_goal_replay_excluded_set};

/// One player's stats over one match, in [`MatchStat`] order.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerMatchStats {
    /// Stat values.
    pub values: [f32; MATCH_STAT_COUNT],
    /// Seconds of live play this player was present for.
    pub live_seconds: f32,
}

/// Field thirds boundary in Unreal units (field half-width / 3).
const THIRD_BOUNDARY_Y: f32 = 5120.0 / 3.0;
/// Car height below which a car counts as grounded.
const GROUND_HEIGHT: f32 = 40.0;
/// Car height above which an airborne car counts as high in the air.
const HIGH_AIR_HEIGHT: f32 = 300.0;
/// A car beyond these coordinates and above [`WALL_MINIMUM_HEIGHT`] is on (or next to) a wall.
const WALL_X: f32 = 3900.0;
const WALL_Y: f32 = 4900.0;
const WALL_MINIMUM_HEIGHT: f32 = 100.0;
const SUPERSONIC_SPEED: f32 = 2200.0;
const SLOW_SPEED: f32 = 500.0;
/// Boost increments above this are big pads (100), below are small pads (12).
const BIG_PAD_INCREMENT: f32 = 0.25;
const SMALL_PAD_MINIMUM_INCREMENT: f32 = 0.04;
/// Ball velocity change that counts as a touch when a car is close enough.
const TOUCH_MINIMUM_VELOCITY_CHANGE: f32 = 250.0;
/// Maximum centre-to-centre car/ball distance for a touch (ball radius 93 + car reach).
const TOUCH_MAXIMUM_DISTANCE: f32 = 260.0;
/// Frame durations are clamped so a stalled network frame cannot dominate a time average.
const MAXIMUM_FRAME_SECONDS: f32 = 0.1;
/// Car height above which a car is on the ceiling (the ceiling is at 2044).
const CEILING_HEIGHT: f32 = 1980.0;
/// Highest car a speed flip or a half flip starts from.
const LOW_DODGE_MAXIMUM_HEIGHT: f32 = 120.0;
/// Jump-to-dodge delay below which a boosting diagonal front flip is a speed flip.
const SPEED_FLIP_FOLLOW_UP_SECONDS: f32 = 0.2;
const SPEED_FLIP_MINIMUM_SPEED: f32 = 1000.0;
/// Seconds after a back flip within which facing the other way makes it a half flip.
const HALF_FLIP_SECONDS: f32 = 1.2;
/// Forward-vector dot product below which the car has turned around.
const TURNED_AROUND_DOT: f32 = -0.7;
/// Airborne time before a landing dodge counts as a wavedash rather than a jump and flip.
const WAVEDASH_MINIMUM_AIRBORNE_SECONDS: f32 = 0.3;
/// Airborne time after which coming back down counts as a landing (a hop is not one).
const LANDING_MINIMUM_AIRBORNE_SECONDS: f32 = 0.5;
/// Height at which a falling car's landing is judged: a frame or two before contact, and
/// above a car lying on its side (~40), which never counts as grounded.
const LANDING_HEIGHT: f32 = 60.0;
/// Up-vector z above which the car comes down on its four wheels (about 45° of tilt).
const LANDING_UPRIGHT_UP_Z: f32 = 0.7;
/// Seconds after a landing at which the speed kept is measured.
const LANDING_SPEED_SECONDS: f32 = 0.4;
/// Horizontal speed at landing below which there is no momentum to keep (a car dropping
/// straight down).
const LANDING_MINIMUM_SPEED: f32 = 500.0;
/// Cap on the speed kept: boosting away from a landing should not hide a bad one.
const LANDING_MAXIMUM_SPEED_KEPT: f32 = 1.2;
/// Replays store angular velocity in hundredths of rad/s (full air roll, 5.5 rad/s, is 550).
const ANGULAR_VELOCITY_SCALE: f32 = 100.0;
/// Roll rate (rad/s) of a tornado spin: over half of full air roll.
const TORNADO_MINIMUM_ROLL_RATE: f32 = 3.0;
/// Share of the rotation that must be roll, about the car's nose.
const TORNADO_MINIMUM_ROLL_SHARE: f32 = 0.8;
/// Spin time that makes a tornado spin. On 30 ranked replays per rank, at least one such
/// spin in a match: 18 % of Bronze and Gold players, 43 % of Diamond, 64 % of Champion,
/// 90 % of SSL.
const TORNADO_MINIMUM_SECONDS: f32 = 0.5;
/// Forward-vector dot product with the spin's first frame below which the nose has drifted
/// (~35°): the spin is out of control, not a tornado spin.
const TORNADO_MINIMUM_NOSE_STEADINESS: f32 = 0.82;
/// Car-to-ball distance at which an airborne car can put its wheels on the ball.
const FLIP_RESET_BALL_DISTANCE: f32 = 200.0;
/// The ball rides on the car: its height over the car, its horizontal offset (the hood
/// counts), how high the car itself may be, and how closely the ball's horizontal velocity
/// must follow a car that is actually moving. Measured on high-rank replays, where carried
/// balls sit 50 to 200 over the car and up to ~200 off its centre.
const DRIBBLE_MINIMUM_RELATIVE_HEIGHT: f32 = 50.0;
const DRIBBLE_MAXIMUM_RELATIVE_HEIGHT: f32 = 200.0;
const DRIBBLE_MAXIMUM_HORIZONTAL_DISTANCE: f32 = 180.0;
const DRIBBLE_MAXIMUM_CAR_HEIGHT: f32 = 150.0;
const DRIBBLE_MAXIMUM_SPEED_DIFFERENCE: f32 = 300.0;
const DRIBBLE_MINIMUM_CAR_SPEED: f32 = 300.0;
/// Carry time before a dodge counts as a flick.
const FLICK_MINIMUM_CARRY_SECONDS: f32 = 0.4;
/// Seconds between the end of a carry and the dodge for it to still be a flick: the jump
/// before the flip lifts the ball off the roof.
const FLICK_MAXIMUM_SECONDS_AFTER_CARRY: f32 = 0.6;
/// Speed (along the car) of a car driving backwards into a half flip.
const HALF_FLIP_MINIMUM_REVERSE_SPEED: f32 = 300.0;
/// Speed (along the nose) of a half flip that ended facing the way the car travels.
const HALF_FLIP_MINIMUM_FORWARD_SPEED: f32 = 300.0;
/// Two aerial touches by the same player this close together are an air dribble.
const AIR_DRIBBLE_SECONDS: f32 = 1.0;
/// Ball past this canonical y and above this height is on the opponents' backboard (the
/// goal is 642 high).
const BACKBOARD_Y: f32 = 4900.0;
const BACKBOARD_MINIMUM_HEIGHT: f32 = 700.0;
/// Seconds from the first touch within which a touch off the backboard is a double tap.
const DOUBLE_TAP_SECONDS: f32 = 3.0;
/// A touch this soon after the car's dodge started is a flip touch.
const FLIP_TOUCH_SECONDS: f32 = 0.3;
/// Ball speed after a touch that makes it a power shot (~100 km/h).
const POWER_SHOT_SPEED: f32 = 2800.0;
/// An opponent this close to the ball when it is touched makes the touch a 50/50.
const FIFTY_FIFTY_DISTANCE: f32 = 300.0;
/// Seconds after a 50/50 at which the ball's side of the contact point decides who won it.
const FIFTY_FIFTY_OUTCOME_SECONDS: f32 = 1.0;

/// Centre-to-centre distance at which two cars are touching (a car is ~120 long).
const BUMP_DISTANCE: f32 = 180.0;
/// Velocity change in one frame that makes a car-to-car contact a bump.
const BUMP_MINIMUM_VELOCITY_CHANGE: f32 = 400.0;
/// Contacts between the same two cars closer together than this are one bump.
const BUMP_REPEAT_SECONDS: f32 = 1.0;

/// A horizontal unit direction.
#[derive(Debug, Clone, Copy)]
struct HorizontalDirection {
    x: f32,
    y: f32,
}

/// A landing waiting to see how much of its speed the car keeps.
#[derive(Debug, Clone, Copy)]
struct PendingLandingSpeed {
    seconds_left: f32,
    /// Horizontal speed when the landing was judged.
    speed: f32,
}

/// An air roll in progress that may become a tornado spin.
#[derive(Debug, Clone, Copy)]
struct TornadoSpin {
    seconds: f32,
    /// The nose's direction when the spin started: a controlled spin keeps it.
    start_forward: Vector3,
    counted: bool,
}

/// A back flip waiting to see whether the car turns around (a half flip).
#[derive(Debug, Clone, Copy)]
struct PendingHalfFlip {
    seconds_left: f32,
    forward: HorizontalDirection,
    /// Horizontal direction the car was travelling (backwards) when it flipped.
    travel: HorizontalDirection,
}

/// Per-car state of the mechanics that span several frames.
#[derive(Debug, Clone, Copy, Default)]
struct MechanicsState {
    seconds_airborne: f32,
    /// Double jumps and dodges since the car last touched a surface.
    flips_since_surface: u32,
    /// Whether the car came within wheel reach of the ball since it left a surface.
    near_ball_since_surface: bool,
    half_flip: Option<PendingHalfFlip>,
    carrying_seconds: f32,
    /// Length of the last finished carry and the seconds since it ended, for flicks.
    last_carry_seconds: f32,
    seconds_since_carry: Option<f32>,
    on_ceiling: bool,
    seconds_since_dodge: Option<f32>,
    /// This airborne stretch's landing has been judged.
    landing_judged: bool,
    landing_speed: Option<PendingLandingSpeed>,
    tornado: Option<TornadoSpin>,
}

/// The last touch of the ball, for air dribbles and double taps.
#[derive(Debug, Clone, Copy)]
struct LastTouch {
    slot: usize,
    time: f32,
    /// Car and ball both in the air.
    aerial: bool,
    /// The ball reached the opponents' backboard since.
    off_backboard: bool,
    /// Ball y when touched, for passes.
    ball_y: f32,
}

/// A 50/50 waiting for its outcome.
#[derive(Debug, Clone, Copy)]
struct PendingFiftyFifty {
    time: f32,
    ball_y: f32,
    toucher: usize,
    opponent: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct PreviousCarState {
    boost: f32,
    height: f32,
    was_demolished: bool,
    present: bool,
    jumping: bool,
    double_jumping: bool,
    dodging: bool,
    /// Seconds since the last jump press, for flip timing.
    seconds_since_jump: f32,
    mechanics: MechanicsState,
    velocity: Vector3,
}

/// Time-weighted and count accumulators for one player.
#[derive(Debug, Clone, Default)]
struct Accumulator {
    seconds: f64,
    speed: f64,
    supersonic: f64,
    slow: f64,
    ground: f64,
    low_air: f64,
    high_air: f64,
    wall: f64,
    height: f64,
    jumps: u32,
    boost: f64,
    boost_empty: f64,
    boost_full: f64,
    boost_collected: f64,
    boost_used: f64,
    big_pads: u32,
    small_pads: u32,
    boost_while_supersonic: f64,
    defensive_third: f64,
    offensive_third: f64,
    canonical_y: f64,
    behind_ball: f64,
    distance_to_ball: f64,
    closest_on_team: f64,
    last_back_on_team: f64,
    most_forward_on_team: f64,
    nearest_teammate_distance: f64,
    nearest_teammate_seconds: f64,
    distance_to_own_goal: f64,
    possession: f64,
    touches: u32,
    ball_speed_after_touch: f64,
    aerial_touches: u32,
    forward_touches: u32,
    touch_speed_gain: f64,
    goals: u32,
    demos_received: u32,
    airborne_seconds: f64,
    airborne_upright: f64,
    slow_with_boost: f64,
    jump_presses: u32,
    double_jumps: u32,
    fast_aerials: u32,
    dodges: u32,
    wavedash_like_dodges: u32,
    aerial_dodges: u32,
    forward_dodges: u32,
    backward_dodges: u32,
    sideways_dodges: u32,
    diagonal_dodges: u32,
    fast_dodges: u32,
    boosting: f64,
    powerslide: f64,
    air_roll: f64,
    reverse: f64,
    absolute_steer: f64,
    air_angular_speed: f64,
    demos_inflicted: u32,
    kickoff_first_touches: u32,
    kickoff_wins: u32,
    double_commit: f64,
    defending_seconds: f64,
    last_back_defending: f64,
    goal_side_defending: f64,
    last_back_goal_distance_defending: f64,
    overcommit: f64,
    opponent_possession_own_half: f64,
    /// Time as the team's closest player to an opponent on the ball in the own half.
    engaged_defending: f64,
    shadow_defense: f64,
    kickoffs_taken: u32,
    /// Sum over kickoffs taken of the approach speed, in units of supersonic speed.
    kickoff_approach_speed: f64,
    passes: u32,
    redirects: u32,
    wall_reads: u32,
    shot_attempts: u32,
    shots_on_target: u32,
    teammate_possession: f64,
    support_spacing: f64,
    recovery_touches: u32,
    recovered_goal_side: u32,
    speed_flips: u32,
    half_flips: u32,
    wavedashes: u32,
    tornado_spins: u32,
    landings: u32,
    clean_landings: u32,
    /// Sum of the speed kept after landings with speed to keep, and how many there were.
    landing_speed_kept: f64,
    landings_with_speed: u32,
    flip_resets: u32,
    ceiling_touches: u32,
    dribble: f64,
    flicks: u32,
    air_dribble_touches: u32,
    double_taps: u32,
    wall_touches: u32,
    flip_touches: u32,
    power_shots: u32,
    fifty_fifties: u32,
    fifty_fifty_wins: u32,
    bumps_inflicted: u32,
    bumps_received: u32,
    teammate_bumps: u32,
    own_goals: u32,
    /// The player's team lost by forfeit (whole matches only).
    lost_by_forfeit: bool,
    /// ... while level or ahead on goals: a deliberate throw.
    forfeited_level_or_ahead: bool,
}

fn length(vector: &Vector3) -> f32 {
    vector
        .z
        .mul_add(vector.z, vector.x.mul_add(vector.x, vector.y * vector.y))
        .sqrt()
}

fn distance(a: &Vector3, b: &Vector3) -> f32 {
    let difference = Vector3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    };
    length(&difference)
}

/// Z component of the car's up vector (1 = wheels down, −1 = upside down).
fn up_z(rotation: &Quaternion) -> f32 {
    2.0f32.mul_add(
        -rotation.x.mul_add(rotation.x, rotation.y * rotation.y),
        1.0,
    )
}

/// Direction the car's nose points.
fn nose_direction(rotation: &Quaternion) -> Vector3 {
    Vector3 {
        x: 2.0f32.mul_add(
            -rotation.y.mul_add(rotation.y, rotation.z * rotation.z),
            1.0,
        ),
        y: 2.0 * rotation.x.mul_add(rotation.y, rotation.w * rotation.z),
        z: 2.0 * rotation.x.mul_add(rotation.z, -rotation.w * rotation.y),
    }
}

/// Whether the ball is against a side wall, or a back wall outside the goal mouth, off the
/// floor.
fn ball_on_wall(ball: &Vector3) -> bool {
    let in_goal_mouth = ball.x.abs() < GOAL_MOUTH_HALF_WIDTH && ball.z < GOAL_MOUTH_HEIGHT;
    ball.z > BALL_WALL_MINIMUM_HEIGHT
        && (ball.x.abs() > BALL_ON_SIDE_WALL_X
            || (ball.y.abs() > BALL_ON_BACK_WALL_Y && !in_goal_mouth))
}

/// Where a ball's projected flight meets the back wall of the goal team `sign` attacks
/// (canonical +y).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShotTarget {
    /// Into the goal mouth.
    OnTarget,
    /// Aimed at goal (within the goal area) but wide or over: post, crossbar, side net.
    Missed,
}

/// Follows the ball with gravity and bounces off the floor, side walls and ceiling until it
/// reaches the attacked back wall. `None` when it does not get there within
/// [`PROJECTION_MAXIMUM_STEPS`] or arrives outside the goal area (a clear, a pass).
fn shot_target(position: &Vector3, velocity: &Vector3, sign: f32) -> Option<ShotTarget> {
    let mut x = position.x;
    let mut y = sign * position.y;
    let mut z = position.z;
    let mut velocity_x = velocity.x;
    let velocity_y = sign * velocity.y;
    let mut velocity_z = velocity.z;
    if velocity_y <= 0.0 {
        return None;
    }
    for _ in 0..PROJECTION_MAXIMUM_STEPS {
        velocity_z = GRAVITY.mul_add(-PROJECTION_STEP_SECONDS, velocity_z);
        x = velocity_x.mul_add(PROJECTION_STEP_SECONDS, x);
        y = velocity_y.mul_add(PROJECTION_STEP_SECONDS, y);
        z = velocity_z.mul_add(PROJECTION_STEP_SECONDS, z);
        if z < BALL_RADIUS && velocity_z < 0.0 {
            z = BALL_RADIUS;
            velocity_z *= -BOUNCE_RESTITUTION;
        }
        if z > CEILING_Z - BALL_RADIUS && velocity_z > 0.0 {
            velocity_z *= -BOUNCE_RESTITUTION;
        }
        if x.abs() > SIDE_WALL_X - BALL_RADIUS && x * velocity_x > 0.0 {
            velocity_x *= -BOUNCE_RESTITUTION;
        }
        if y >= BACK_WALL_Y - BALL_RADIUS {
            return if x.abs() < ON_TARGET_HALF_WIDTH && z < ON_TARGET_HEIGHT {
                Some(ShotTarget::OnTarget)
            } else if x.abs() < SHOT_AREA_HALF_WIDTH && z < SHOT_AREA_HEIGHT {
                Some(ShotTarget::Missed)
            } else {
                None
            };
        }
    }
    None
}

/// Horizontal direction the car's nose points, or `None` when it points (nearly) straight
/// up or down.
fn horizontal_forward(rotation: &Quaternion) -> Option<HorizontalDirection> {
    let x = 2.0f32.mul_add(
        -rotation.y.mul_add(rotation.y, rotation.z * rotation.z),
        1.0,
    );
    let y = 2.0 * rotation.x.mul_add(rotation.y, rotation.w * rotation.z);
    let length = x.hypot(y);
    (length > 0.3).then(|| HorizontalDirection {
        x: x / length,
        y: y / length,
    })
}

/// One present, live car in the current frame.
#[derive(Debug, Clone, Copy)]
struct LiveCar {
    slot: usize,
    position: Vector3,
    velocity: Vector3,
    /// Size of the velocity change since the previous frame (0 on the car's first frame).
    velocity_change: f32,
    on_wall: bool,
    canonical_y: f32,
    distance_to_ball: f32,
}

/// Computes [`PlayerMatchStats`] for every roster slot. Slots whose player never appears
/// are `None`.
#[must_use]
pub fn compute_player_match_stats(
    parsed: &ParsedReplay,
    roster: &PlayerRoster,
) -> [Option<PlayerMatchStats>; TOTAL_SLOTS] {
    compute_player_window_stats(parsed, roster, 0..parsed.frames.len())
}

/// Same stats as [`compute_player_match_stats`], restricted to the frames in `frames`.
///
/// Goals and kickoffs count only when they happen inside the window. The scoreboard
/// family comes from the end-of-match header, so it is filled only when the window covers
/// the whole replay and is zero otherwise.
#[must_use]
pub fn compute_player_window_stats(
    parsed: &ParsedReplay,
    roster: &PlayerRoster,
    frames: Range<usize>,
) -> [Option<PlayerMatchStats>; TOTAL_SLOTS] {
    let whole_match = frames.start == 0 && frames.end >= parsed.frames.len();
    let slot_by_name: HashMap<&str, usize> = roster
        .names
        .iter()
        .enumerate()
        .filter(|(_, name)| !name.is_empty())
        .map(|(slot, name)| (name.as_str(), slot))
        .collect();
    let excluded = build_goal_replay_excluded_set(&parsed.goal_frames, &parsed.kickoff_frames);

    let mut accumulators: [Accumulator; TOTAL_SLOTS] =
        core::array::from_fn(|_| Accumulator::default());
    let mut previous: [PreviousCarState; TOTAL_SLOTS] = [PreviousCarState::default(); TOTAL_SLOTS];
    let mut previous_ball_velocity: Option<Vector3> = None;
    let mut last_demolition_time: HashMap<i32, f32> = HashMap::new();
    // Time of each player's last touch while waiting to see them get back goal-side.
    let mut pending_recovery: [Option<f32>; TOTAL_SLOTS] = [None; TOTAL_SLOTS];
    let mut last_touch: Option<LastTouch> = None;
    // Per team (blue, orange): when the ball last came off a wall or backboard in its half.
    let mut ball_off_own_wall: [Option<f32>; 2] = [None; 2];
    let mut pending_fifty_fifty: Option<PendingFiftyFifty> = None;
    // Time of the last bump between each pair of slots (bumper, bumped).
    let mut last_bump_time = [[f32::NEG_INFINITY; TOTAL_SLOTS]; TOTAL_SLOTS];

    // Goals per team (blue attacks +y), counted from where the ball went in so own goals
    // land on the right side.
    let mut team_goals = [0u32; 2];
    for goal in &parsed.goals {
        let Some(ball_y) = parsed
            .frames
            .get(goal.frame)
            .map(|frame| frame.ball.position.y)
        else {
            continue;
        };
        if let Some(points) = team_goals.get_mut(usize::from(ball_y < 0.0)) {
            *points += 1;
        }
        if !frames.contains(&goal.frame) {
            continue;
        }
        if let Some(&slot) = slot_by_name.get(goal.player_name.as_str())
            && let Some(accumulator) = accumulators.get_mut(slot)
        {
            accumulator.goals += 1;
            accumulator.own_goals += u32::from(team_sign(slot) * ball_y < 0.0);
        }
    }
    if whole_match && parsed.outcome.forfeit {
        for (slot, accumulator) in accumulators.iter_mut().enumerate() {
            let team = usize::from(slot >= SLOTS_PER_TEAM);
            let won = parsed
                .outcome
                .winning_team
                .map(|winner| usize::from(u8::from(winner)) == team);
            accumulator.lost_by_forfeit = won == Some(false);
            accumulator.forfeited_level_or_ahead =
                accumulator.lost_by_forfeit && team_goals.get(team) >= team_goals.get(1 - team);
        }
    }

    for (frame_index, frame) in parsed
        .frames
        .iter()
        .enumerate()
        .skip(frames.start)
        .take(frames.end.saturating_sub(frames.start))
    {
        if excluded.contains(&frame_index) {
            previous_ball_velocity = None;
            previous = [PreviousCarState::default(); TOTAL_SLOTS];
            last_touch = None;
            ball_off_own_wall = [None; 2];
            pending_fifty_fifty = None;
            continue;
        }
        let seconds = f64::from(frame.delta.clamp(0.0, MAXIMUM_FRAME_SECONDS));
        let ball = &frame.ball;

        let mut live_cars: Vec<LiveCar> = Vec::with_capacity(TOTAL_SLOTS);
        for player in &frame.players {
            let Some(&slot) = slot_by_name.get(player.name.as_str()) else {
                continue;
            };
            let state = &player.actor_state;
            let Some(accumulator) = accumulators.get_mut(slot) else {
                continue;
            };
            let Some(prior) = previous.get_mut(slot) else {
                continue;
            };

            if state.is_demolished {
                if !prior.was_demolished {
                    accumulator.demos_received += 1;
                }
                *prior = PreviousCarState {
                    was_demolished: true,
                    present: true,
                    ..*prior
                };
                continue;
            }

            // The roster slot decides the team, not the frame: some replays flip a player's
            // team attribute back and forth, which would mirror their positioning mid-match.
            let sign = team_sign(slot);
            let speed = length(&state.velocity);
            let height = state.position.z;
            let on_wall = (state.position.x.abs() > WALL_X || state.position.y.abs() > WALL_Y)
                && height > WALL_MINIMUM_HEIGHT;
            let grounded = height < GROUND_HEIGHT;
            let canonical_y = sign * state.position.y;
            let ball_canonical_y = sign * ball.position.y;
            let distance_to_ball = distance(&state.position, &ball.position);
            let own_goal = Vector3 {
                x: 0.0,
                y: -sign * 5120.0,
                z: 0.0,
            };

            accumulator.seconds += seconds;
            accumulator.speed = f64::mul_add(seconds, f64::from(speed), accumulator.speed);
            accumulator.supersonic = f64::mul_add(
                seconds,
                f64::from(u8::from(speed >= SUPERSONIC_SPEED)),
                accumulator.supersonic,
            );
            accumulator.slow = f64::mul_add(
                seconds,
                f64::from(u8::from(speed < SLOW_SPEED)),
                accumulator.slow,
            );
            accumulator.ground =
                f64::mul_add(seconds, f64::from(u8::from(grounded)), accumulator.ground);
            accumulator.wall =
                f64::mul_add(seconds, f64::from(u8::from(on_wall)), accumulator.wall);
            accumulator.high_air = f64::mul_add(
                seconds,
                f64::from(u8::from(!on_wall && height >= HIGH_AIR_HEIGHT)),
                accumulator.high_air,
            );
            accumulator.low_air = f64::mul_add(
                seconds,
                f64::from(u8::from(!on_wall && !grounded && height < HIGH_AIR_HEIGHT)),
                accumulator.low_air,
            );
            accumulator.height = f64::mul_add(seconds, f64::from(height), accumulator.height);
            if !grounded && !on_wall {
                accumulator.airborne_seconds += seconds;
                accumulator.airborne_upright = f64::mul_add(
                    seconds,
                    f64::from(u8::from(up_z(&state.rotation) > 0.7)),
                    accumulator.airborne_upright,
                );
            }

            accumulator.boost = f64::mul_add(seconds, f64::from(state.boost), accumulator.boost);
            accumulator.boost_empty = f64::mul_add(
                seconds,
                f64::from(u8::from(state.boost < 0.03)),
                accumulator.boost_empty,
            );
            accumulator.boost_full = f64::mul_add(
                seconds,
                f64::from(u8::from(state.boost > 0.95)),
                accumulator.boost_full,
            );
            accumulator.slow_with_boost = f64::mul_add(
                seconds,
                f64::from(u8::from(speed < SLOW_SPEED && state.boost > 0.5)),
                accumulator.slow_with_boost,
            );

            if prior.present && !prior.was_demolished {
                let boost_change = state.boost - prior.boost;
                if boost_change > 0.0 {
                    accumulator.boost_collected += f64::from(boost_change);
                    if boost_change > BIG_PAD_INCREMENT {
                        accumulator.big_pads += 1;
                    } else if boost_change > SMALL_PAD_MINIMUM_INCREMENT {
                        accumulator.small_pads += 1;
                    }
                } else if boost_change < 0.0 {
                    accumulator.boost_used += f64::from(-boost_change);
                    if speed >= SUPERSONIC_SPEED {
                        accumulator.boost_while_supersonic += seconds;
                    }
                }
                if prior.height < GROUND_HEIGHT && !grounded && !on_wall {
                    accumulator.jumps += 1;
                }
            }

            accumulator.defensive_third = f64::mul_add(
                seconds,
                f64::from(u8::from(canonical_y < -THIRD_BOUNDARY_Y)),
                accumulator.defensive_third,
            );
            accumulator.offensive_third = f64::mul_add(
                seconds,
                f64::from(u8::from(canonical_y > THIRD_BOUNDARY_Y)),
                accumulator.offensive_third,
            );
            accumulator.canonical_y =
                f64::mul_add(seconds, f64::from(canonical_y), accumulator.canonical_y);
            accumulator.behind_ball = f64::mul_add(
                seconds,
                f64::from(u8::from(canonical_y < ball_canonical_y)),
                accumulator.behind_ball,
            );
            accumulator.distance_to_ball = f64::mul_add(
                seconds,
                f64::from(distance_to_ball),
                accumulator.distance_to_ball,
            );
            accumulator.distance_to_own_goal = f64::mul_add(
                seconds,
                f64::from(distance(&state.position, &own_goal)),
                accumulator.distance_to_own_goal,
            );

            let velocity_change = if prior.present {
                distance(&state.velocity, &prior.velocity)
            } else {
                0.0
            };
            let seconds_since_jump =
                accumulate_mechanics(accumulator, prior, state, grounded, on_wall, seconds);
            let mechanics = accumulate_multi_frame_mechanics(
                accumulator,
                prior,
                state,
                ball,
                CarContact { grounded, on_wall },
                seconds_since_jump,
                seconds as f32,
            );

            *prior = PreviousCarState {
                boost: state.boost,
                height,
                was_demolished: false,
                present: true,
                jumping: state.controls.jumping,
                double_jumping: state.controls.double_jumping,
                dodging: state.controls.dodging,
                seconds_since_jump,
                mechanics,
                velocity: state.velocity,
            };
            live_cars.push(LiveCar {
                slot,
                position: state.position,
                velocity: state.velocity,
                velocity_change,
                on_wall,
                canonical_y,
                distance_to_ball,
            });
        }

        accumulate_team_roles(&mut accumulators, &live_cars, seconds);
        accumulate_double_commits(&mut accumulators, &live_cars, seconds);
        accumulate_bumps(
            &mut accumulators,
            &live_cars,
            &mut last_bump_time,
            frame.time,
        );
        let possessor = live_cars
            .iter()
            .min_by(|a, b| a.distance_to_ball.total_cmp(&b.distance_to_ball))
            .copied();
        accumulate_situational(
            &mut accumulators,
            &live_cars,
            ball.position.y,
            possessor.as_ref(),
            seconds,
        );
        for car in &live_cars {
            let Some(touched_at) = pending_recovery.get(car.slot).copied().flatten() else {
                continue;
            };
            let ball_canonical_y = team_sign(car.slot) * ball.position.y;
            if car.canonical_y < ball_canonical_y {
                if let Some(accumulator) = accumulators.get_mut(car.slot) {
                    accumulator.recovered_goal_side += 1;
                }
                pending_recovery[car.slot] = None;
            } else if frame.time - touched_at > RECOVERY_SECONDS {
                pending_recovery[car.slot] = None;
            }
        }
        for demolition in &frame.demolitions {
            let recent = last_demolition_time
                .get(&demolition.victim_actor_id)
                .is_some_and(|&time| frame.time - time < DEMOLITION_REPEAT_SECONDS);
            last_demolition_time.insert(demolition.victim_actor_id, frame.time);
            if recent {
                continue;
            }
            let attacker_slot = frame
                .players
                .iter()
                .find(|player| player.actor_id == demolition.attacker_actor_id)
                .and_then(|player| slot_by_name.get(player.name.as_str()));
            if let Some(&slot) = attacker_slot
                && let Some(accumulator) = accumulators.get_mut(slot)
            {
                accumulator.demos_inflicted += 1;
            }
        }

        if ball_on_wall(&ball.position) {
            // The half the ball is in: that team's own wall or backboard.
            ball_off_own_wall[usize::from(ball.position.y > 0.0)] = Some(frame.time);
        }
        if let Some(touch) = last_touch.as_mut()
            && team_sign(touch.slot) * ball.position.y > BACKBOARD_Y
            && ball.position.z > BACKBOARD_MINIMUM_HEIGHT
        {
            touch.off_backboard = true;
        }
        if let Some(contest) = pending_fifty_fifty
            && frame.time - contest.time > FIFTY_FIFTY_OUTCOME_SECONDS
        {
            for slot in [contest.toucher, contest.opponent] {
                if team_sign(slot) * (ball.position.y - contest.ball_y) > 0.0
                    && let Some(accumulator) = accumulators.get_mut(slot)
                {
                    accumulator.fifty_fifty_wins += 1;
                }
            }
            pending_fifty_fifty = None;
        }

        if let Some(closest) = live_cars
            .iter()
            .min_by(|a, b| a.distance_to_ball.total_cmp(&b.distance_to_ball))
        {
            let mut touched = false;
            if let Some(accumulator) = accumulators.get_mut(closest.slot) {
                accumulator.possession += seconds;
            }
            if let Some(before) = previous_ball_velocity {
                let change = Vector3 {
                    x: ball.velocity.x - before.x,
                    y: ball.velocity.y - before.y,
                    z: ball.velocity.z - before.z,
                };
                if length(&change) > TOUCH_MINIMUM_VELOCITY_CHANGE
                    && closest.distance_to_ball < TOUCH_MAXIMUM_DISTANCE
                    && let Some(accumulator) = accumulators.get_mut(closest.slot)
                {
                    let sign = if closest.slot < SLOTS_PER_TEAM {
                        1.0
                    } else {
                        -1.0
                    };
                    let speed_after = length(&ball.velocity);
                    accumulator.touches += 1;
                    accumulator.recovery_touches += 1;
                    if let Some(pending) = pending_recovery.get_mut(closest.slot) {
                        *pending = Some(frame.time);
                    }
                    accumulator.ball_speed_after_touch += f64::from(speed_after);
                    accumulator.touch_speed_gain += f64::from(speed_after - length(&before));
                    accumulator.aerial_touches += u32::from(ball.position.z > HIGH_AIR_HEIGHT);
                    accumulator.forward_touches += u32::from(sign * ball.velocity.y > 0.0);
                    accumulator.wall_touches += u32::from(closest.on_wall);
                    accumulator.power_shots += u32::from(speed_after >= POWER_SHOT_SPEED);
                    accumulator.flip_touches += u32::from(
                        previous
                            .get(closest.slot)
                            .and_then(|car| car.mechanics.seconds_since_dodge)
                            .is_some_and(|since| since < FLIP_TOUCH_SECONDS),
                    );
                    let aerial = ball.position.z > HIGH_AIR_HEIGHT
                        && closest.position.z > GROUND_HEIGHT
                        && !closest.on_wall;
                    let team = usize::from(closest.slot >= SLOTS_PER_TEAM);
                    if let Some(last) = last_touch
                        && last.slot != closest.slot
                        && (last.slot >= SLOTS_PER_TEAM) == (closest.slot >= SLOTS_PER_TEAM)
                    {
                        let elapsed = frame.time - last.time;
                        if elapsed < PASS_SECONDS
                            && sign * (ball.position.y - last.ball_y) > 0.0
                            && let Some(passer) = accumulators.get_mut(last.slot)
                        {
                            passer.passes += 1;
                        }
                        if elapsed < REDIRECT_SECONDS
                            && speed_after >= REDIRECT_MINIMUM_BALL_SPEED
                            && shot_target(&ball.position, &ball.velocity, sign)
                                == Some(ShotTarget::OnTarget)
                            && let Some(receiver) = accumulators.get_mut(closest.slot)
                        {
                            receiver.redirects += 1;
                        }
                    }
                    if ball_off_own_wall[team]
                        .is_some_and(|time| frame.time - time < WALL_READ_SECONDS)
                        && sign * ball.velocity.y > 0.0
                        && let Some(accumulator) = accumulators.get_mut(closest.slot)
                    {
                        accumulator.wall_reads += 1;
                        ball_off_own_wall[team] = None;
                    }
                    let Some(accumulator) = accumulators.get_mut(closest.slot) else {
                        continue;
                    };
                    if speed_after >= SHOT_MINIMUM_BALL_SPEED
                        && let Some(target) = shot_target(&ball.position, &ball.velocity, sign)
                    {
                        accumulator.shot_attempts += 1;
                        accumulator.shots_on_target += u32::from(target == ShotTarget::OnTarget);
                    }
                    if let Some(last) = last_touch
                        && last.slot == closest.slot
                        && ball.position.z > HIGH_AIR_HEIGHT
                    {
                        let elapsed = frame.time - last.time;
                        if last.off_backboard && elapsed < DOUBLE_TAP_SECONDS {
                            accumulator.double_taps += 1;
                        } else if last.aerial && aerial && elapsed < AIR_DRIBBLE_SECONDS {
                            accumulator.air_dribble_touches += 1;
                        }
                    }
                    last_touch = Some(LastTouch {
                        slot: closest.slot,
                        time: frame.time,
                        aerial,
                        off_backboard: false,
                        ball_y: ball.position.y,
                    });
                    touched = true;
                }
            }
            if touched
                && pending_fifty_fifty.is_none()
                && let Some(opponent) = live_cars.iter().find(|car| {
                    (car.slot < SLOTS_PER_TEAM) != (closest.slot < SLOTS_PER_TEAM)
                        && car.distance_to_ball < FIFTY_FIFTY_DISTANCE
                })
            {
                for slot in [closest.slot, opponent.slot] {
                    if let Some(accumulator) = accumulators.get_mut(slot) {
                        accumulator.fifty_fifties += 1;
                    }
                }
                pending_fifty_fifty = Some(PendingFiftyFifty {
                    time: frame.time,
                    ball_y: ball.position.y,
                    toucher: closest.slot,
                    opponent: opponent.slot,
                });
            }
        }
        previous_ball_velocity = Some(ball.velocity);
    }

    let kickoff_count = accumulate_kickoffs(&mut accumulators, parsed, &slot_by_name, &frames);
    let scoreboard_by_name: HashMap<&str, &replay_structs::HeaderPlayerStats> = parsed
        .header_player_stats
        .iter()
        .filter(|_| whole_match)
        .map(|stats| (stats.name.as_str(), stats))
        .collect();
    core::array::from_fn(|slot| {
        let scoreboard = roster
            .names
            .get(slot)
            .and_then(|name| scoreboard_by_name.get(name.as_str()))
            .copied();
        accumulators.get(slot).and_then(|accumulator| {
            finish(
                accumulator,
                scoreboard,
                kickoff_count,
                parsed.match_format.players_per_team,
            )
        })
    })
}

/// Height below which a dodge on a descending car counts as a wavedash-like landing dodge.
const WAVEDASH_MAXIMUM_HEIGHT: f32 = 50.0;
/// Jump-to-dodge or jump-to-double-jump delay below which the move counts as "fast".
const FAST_FOLLOW_UP_SECONDS: f32 = 0.25;
/// Dodge torque component ratio above which the flip counts as pure pitch or pure roll.
const DOMINANT_TORQUE_RATIO: f32 = 2.0;

/// Input-derived mechanics for one car in one frame. Returns the updated time since the
/// last jump press.
fn accumulate_mechanics(
    accumulator: &mut Accumulator,
    prior: &PreviousCarState,
    state: &replay_structs::ActorState,
    grounded: bool,
    on_wall: bool,
    seconds: f64,
) -> f32 {
    let controls = state.controls;
    let airborne = !grounded && !on_wall;
    accumulator.boosting =
        seconds.mul_add(f64::from(u8::from(controls.boosting)), accumulator.boosting);
    accumulator.powerslide = seconds.mul_add(
        f64::from(u8::from(controls.handbrake && grounded)),
        accumulator.powerslide,
    );
    accumulator.air_roll = seconds.mul_add(
        f64::from(u8::from(controls.handbrake && airborne)),
        accumulator.air_roll,
    );
    accumulator.reverse = seconds.mul_add(
        f64::from(u8::from(controls.throttle < -0.1 && grounded)),
        accumulator.reverse,
    );
    accumulator.absolute_steer =
        seconds.mul_add(f64::from(controls.steer.abs()), accumulator.absolute_steer);
    if airborne {
        accumulator.air_angular_speed = seconds.mul_add(
            f64::from(length(&state.angular_velocity)),
            accumulator.air_angular_speed,
        );
    }

    let jumped = controls.jumping && !prior.jumping;
    let seconds_since_jump = if jumped {
        0.0
    } else if prior.present {
        prior.seconds_since_jump + seconds as f32
    } else {
        f32::MAX
    };
    if jumped {
        accumulator.jump_presses += 1;
    }
    if controls.double_jumping && !prior.double_jumping {
        accumulator.double_jumps += 1;
        accumulator.fast_aerials += u32::from(seconds_since_jump < FAST_FOLLOW_UP_SECONDS);
    }
    if controls.dodging && !prior.dodging {
        accumulator.dodges += 1;
        let height = state.position.z;
        accumulator.wavedash_like_dodges +=
            u32::from(height < WAVEDASH_MAXIMUM_HEIGHT && state.velocity.z < 0.0);
        accumulator.aerial_dodges += u32::from(height > HIGH_AIR_HEIGHT);
        accumulator.fast_dodges += u32::from(seconds_since_jump < FAST_FOLLOW_UP_SECONDS);
        let pitch = controls.dodge_torque.y.abs();
        let roll = controls.dodge_torque.x.abs();
        if pitch + roll > f32::EPSILON {
            if pitch > DOMINANT_TORQUE_RATIO * roll {
                if controls.dodge_torque.y > 0.0 {
                    accumulator.forward_dodges += 1;
                } else {
                    accumulator.backward_dodges += 1;
                }
            } else if roll > DOMINANT_TORQUE_RATIO * pitch {
                accumulator.sideways_dodges += 1;
            } else {
                accumulator.diagonal_dodges += 1;
            }
        }
    }
    seconds_since_jump
}

/// Bumps: two touching cars, one of them knocked off its course. The bumper is the car
/// driving harder into the other; the bumped car is the one whose velocity jumped.
fn accumulate_bumps(
    accumulators: &mut [Accumulator; TOTAL_SLOTS],
    live_cars: &[LiveCar],
    last_bump_time: &mut [[f32; TOTAL_SLOTS]; TOTAL_SLOTS],
    time: f32,
) {
    for bumper in live_cars {
        for bumped in live_cars {
            if bumper.slot == bumped.slot
                || bumped.velocity_change < BUMP_MINIMUM_VELOCITY_CHANGE
                || distance(&bumper.position, &bumped.position) > BUMP_DISTANCE
            {
                continue;
            }
            let toward = Vector3 {
                x: bumped.position.x - bumper.position.x,
                y: bumped.position.y - bumper.position.y,
                z: bumped.position.z - bumper.position.z,
            };
            let closing = |velocity: &Vector3| {
                toward.z.mul_add(
                    velocity.z,
                    toward.x.mul_add(velocity.x, toward.y * velocity.y),
                )
            };
            // Closing speed of each car toward the other, along the line between them.
            if closing(&bumper.velocity) <= -closing(&bumped.velocity) {
                continue;
            }
            let Some(last) = last_bump_time
                .get_mut(bumper.slot)
                .and_then(|row| row.get_mut(bumped.slot))
            else {
                continue;
            };
            if time - *last < BUMP_REPEAT_SECONDS {
                continue;
            }
            *last = time;
            if (bumper.slot < SLOTS_PER_TEAM) == (bumped.slot < SLOTS_PER_TEAM) {
                if let Some(accumulator) = accumulators.get_mut(bumper.slot) {
                    accumulator.teammate_bumps += 1;
                }
            } else {
                if let Some(accumulator) = accumulators.get_mut(bumper.slot) {
                    accumulator.bumps_inflicted += 1;
                }
                if let Some(accumulator) = accumulators.get_mut(bumped.slot) {
                    accumulator.bumps_received += 1;
                }
            }
        }
    }
}

/// Where the car is relative to the ground and walls this frame.
#[derive(Debug, Clone, Copy)]
struct CarContact {
    grounded: bool,
    on_wall: bool,
}

/// Mechanics that span several frames: speed flips, half flips, wavedashes, flip resets,
/// ceiling touches, dribbles and flicks. Returns the car's updated mechanics state.
fn accumulate_multi_frame_mechanics(
    accumulator: &mut Accumulator,
    prior: &PreviousCarState,
    state: &replay_structs::ActorState,
    ball: &BallState,
    contact: CarContact,
    seconds_since_jump: f32,
    seconds: f32,
) -> MechanicsState {
    let previous = if prior.present {
        prior.mechanics
    } else {
        MechanicsState::default()
    };
    let controls = state.controls;
    let height = state.position.z;
    let dodge_started = controls.dodging && !prior.dodging;
    let double_jump_started = controls.double_jumping && !prior.double_jumping;
    let on_ceiling = height > CEILING_HEIGHT && !contact.on_wall;
    let on_surface = contact.grounded || contact.on_wall || on_ceiling;
    let forward = horizontal_forward(&state.rotation);
    let pitch = controls.dodge_torque.y;
    let roll = controls.dodge_torque.x;

    if on_ceiling && !previous.on_ceiling {
        accumulator.ceiling_touches += 1;
    }

    // Flip resets: a second double jump or dodge in one airborne stretch, high up, after
    // the car was within wheel reach of the ball.
    let mut flips_since_surface = if on_surface {
        0
    } else {
        previous.flips_since_surface
    };
    let near_ball_since_surface = !on_surface
        && (previous.near_ball_since_surface
            || distance(&state.position, &ball.position) < FLIP_RESET_BALL_DISTANCE);
    if (dodge_started || double_jump_started) && !on_surface {
        flips_since_surface += 1;
        if flips_since_surface >= 2 && near_ball_since_surface && height > HIGH_AIR_HEIGHT {
            accumulator.flip_resets += 1;
        }
    }

    // Wavedashes: a dodge just before landing, after a real stretch in the air.
    let seconds_airborne = if contact.grounded || contact.on_wall {
        0.0
    } else {
        previous.seconds_airborne + seconds
    };
    if dodge_started
        && height < WAVEDASH_MAXIMUM_HEIGHT
        && state.velocity.z < 0.0
        && previous.seconds_airborne > WAVEDASH_MINIMUM_AIRBORNE_SECONDS
    {
        accumulator.wavedashes += 1;
    }

    // Landings: coming down after a real stretch in the air, the car should meet the
    // ground on its four wheels, not its roof or side. Judged once per stretch, just above
    // the ground.
    let judging_landing = !previous.landing_judged
        && previous.seconds_airborne > LANDING_MINIMUM_AIRBORNE_SECONDS
        && height < LANDING_HEIGHT
        && state.velocity.z <= 0.0;
    if judging_landing {
        accumulator.landings += 1;
        accumulator.clean_landings += u32::from(up_z(&state.rotation) > LANDING_UPRIGHT_UP_Z);
    }
    let landing_judged = seconds_airborne > 0.0 && (previous.landing_judged || judging_landing);

    // Momentum: the share of the landing's horizontal speed still there a moment later.
    // A wavedash or a clean landing keeps it; landing on the roof or side throws it away.
    let horizontal_speed = state.velocity.x.hypot(state.velocity.y);
    let mut landing_speed = previous.landing_speed.map(|pending| PendingLandingSpeed {
        seconds_left: pending.seconds_left - seconds,
        ..pending
    });
    if let Some(pending) = landing_speed
        && pending.seconds_left <= 0.0
    {
        accumulator.landing_speed_kept +=
            f64::from((horizontal_speed / pending.speed).min(LANDING_MAXIMUM_SPEED_KEPT));
        accumulator.landings_with_speed += 1;
        landing_speed = None;
    }
    if judging_landing && horizontal_speed > LANDING_MINIMUM_SPEED {
        landing_speed = Some(PendingLandingSpeed {
            seconds_left: LANDING_SPEED_SECONDS,
            speed: horizontal_speed,
        });
    }

    // Tornado spins: a fast, sustained air roll while boosting (not a side flip), the nose
    // held steady. The replay's own angular velocity is used: rotations update only every
    // other frame or so and their differences are noisy.
    let nose = nose_direction(&state.rotation);
    let angular_velocity = &state.angular_velocity;
    let roll_rate = angular_velocity
        .z
        .mul_add(
            nose.z,
            angular_velocity
                .x
                .mul_add(nose.x, angular_velocity.y * nose.y),
        )
        .abs()
        / ANGULAR_VELOCITY_SCALE;
    let spinning = !on_surface
        && controls.boosting
        && !controls.dodging
        && roll_rate > TORNADO_MINIMUM_ROLL_RATE
        && roll_rate
            > TORNADO_MINIMUM_ROLL_SHARE * length(angular_velocity) / ANGULAR_VELOCITY_SCALE;
    let tornado = match previous.tornado {
        _ if !spinning => None,
        Some(spin) => {
            let steadiness = nose.z.mul_add(
                spin.start_forward.z,
                nose.x
                    .mul_add(spin.start_forward.x, nose.y * spin.start_forward.y),
            );
            (steadiness > TORNADO_MINIMUM_NOSE_STEADINESS).then_some(TornadoSpin {
                seconds: spin.seconds + seconds,
                ..spin
            })
        }
        None => Some(TornadoSpin {
            seconds,
            start_forward: nose,
            counted: false,
        }),
    };
    let tornado = tornado.map(|spin| {
        if !spin.counted && spin.seconds >= TORNADO_MINIMUM_SECONDS {
            accumulator.tornado_spins += 1;
            TornadoSpin {
                counted: true,
                ..spin
            }
        } else {
            spin
        }
    });

    // Speed flips: a fast, boosting, diagonal front flip off the ground.
    if dodge_started
        && height < LOW_DODGE_MAXIMUM_HEIGHT
        && controls.boosting
        && seconds_since_jump < SPEED_FLIP_FOLLOW_UP_SECONDS
        && length(&state.velocity) > SPEED_FLIP_MINIMUM_SPEED
        && pitch > 0.0
        && pitch.abs() <= DOMINANT_TORQUE_RATIO * roll.abs()
        && roll.abs() <= DOMINANT_TORQUE_RATIO * pitch.abs()
    {
        accumulator.speed_flips += 1;
    }

    // Half flips: driving backwards, a back flip, and the car comes out facing the way it
    // travels. A back flip and a slow turn afterwards (common at low rank) is not one.
    let horizontal_speed = state.velocity.x.hypot(state.velocity.y);
    let travel = (horizontal_speed > 1.0).then(|| HorizontalDirection {
        x: state.velocity.x / horizontal_speed,
        y: state.velocity.y / horizontal_speed,
    });
    let speed_along = |direction: HorizontalDirection| {
        state
            .velocity
            .x
            .mul_add(direction.x, state.velocity.y * direction.y)
    };
    let mut half_flip = previous
        .half_flip
        .map(|pending| PendingHalfFlip {
            seconds_left: pending.seconds_left - seconds,
            ..pending
        })
        .filter(|pending| pending.seconds_left > 0.0);
    if let Some(pending) = half_flip
        && height < LOW_DODGE_MAXIMUM_HEIGHT
        && let Some(forward) = forward
        && forward
            .x
            .mul_add(pending.forward.x, forward.y * pending.forward.y)
            < TURNED_AROUND_DOT
        && speed_along(forward) > HALF_FLIP_MINIMUM_FORWARD_SPEED
        && speed_along(pending.travel) > HALF_FLIP_MINIMUM_FORWARD_SPEED
    {
        accumulator.half_flips += 1;
        half_flip = None;
    }
    if dodge_started
        && height < LOW_DODGE_MAXIMUM_HEIGHT
        && pitch < 0.0
        && pitch.abs() > roll.abs()
        && let Some(forward) = forward
        && let Some(travel) = travel
        && speed_along(forward) < -HALF_FLIP_MINIMUM_REVERSE_SPEED
    {
        half_flip = Some(PendingHalfFlip {
            seconds_left: HALF_FLIP_SECONDS,
            forward,
            travel,
        });
    }

    // Dribbles and flicks: the ball on the car's roof, then a dodge.
    let relative_height = ball.position.z - height;
    let carrying = height < DRIBBLE_MAXIMUM_CAR_HEIGHT
        && (DRIBBLE_MINIMUM_RELATIVE_HEIGHT..=DRIBBLE_MAXIMUM_RELATIVE_HEIGHT)
            .contains(&relative_height)
        && (ball.position.x - state.position.x).hypot(ball.position.y - state.position.y)
            < DRIBBLE_MAXIMUM_HORIZONTAL_DISTANCE
        && state.velocity.x.hypot(state.velocity.y) > DRIBBLE_MINIMUM_CAR_SPEED
        && (ball.velocity.x - state.velocity.x).hypot(ball.velocity.y - state.velocity.y)
            < DRIBBLE_MAXIMUM_SPEED_DIFFERENCE;
    let carrying_seconds = if carrying {
        accumulator.dribble += f64::from(seconds);
        previous.carrying_seconds + seconds
    } else {
        0.0
    };
    // The jump before a flick lifts the ball off the roof, so the carry has usually just
    // ended when the dodge starts.
    let carry_ended = !carrying && previous.carrying_seconds > 0.0;
    let last_carry_seconds = if carry_ended {
        previous.carrying_seconds
    } else {
        previous.last_carry_seconds
    };
    let seconds_since_carry = if carrying {
        None
    } else if carry_ended {
        Some(0.0)
    } else {
        previous.seconds_since_carry.map(|since| since + seconds)
    };
    let recently_carried = seconds_since_carry
        .is_some_and(|since| since < FLICK_MAXIMUM_SECONDS_AFTER_CARRY)
        && last_carry_seconds > FLICK_MINIMUM_CARRY_SECONDS;
    if dodge_started && (carrying_seconds > FLICK_MINIMUM_CARRY_SECONDS || recently_carried) {
        accumulator.flicks += 1;
    }

    MechanicsState {
        seconds_airborne,
        flips_since_surface,
        near_ball_since_surface,
        half_flip,
        carrying_seconds,
        last_carry_seconds,
        seconds_since_carry,
        on_ceiling,
        seconds_since_dodge: if dodge_started {
            Some(0.0)
        } else {
            previous.seconds_since_dodge.map(|since| since + seconds)
        },
        landing_judged,
        landing_speed,
        tornado,
    }
}

/// Demolition events of the same victim closer together than this are one demolition.
const DEMOLITION_REPEAT_SECONDS: f32 = 1.0;
/// Ball speed that marks the first touch after a kickoff.
const KICKOFF_TOUCH_BALL_SPEED: f32 = 100.0;
/// Seconds after the first kickoff touch at which the ball side decides who won it.
const KICKOFF_OUTCOME_SECONDS: f32 = 3.0;
/// Two teammates closer than this to the ball at once are double-committing.
const DOUBLE_COMMIT_DISTANCE: f32 = 600.0;

/// Seconds after a touch within which getting back goal-side counts as a recovery.
const RECOVERY_SECONDS: f32 = 3.0;
/// Maximum distance to the ball for goal-side defending to count as shadowing.
const SHADOW_DISTANCE: f32 = 2000.0;
/// Speed toward the own goal of a defender backing off an attacker rather than diving in.
const SHADOW_MINIMUM_RETREAT_SPEED: f32 = 200.0;
/// A teammate's touch this soon after a player's touch, with the ball moved upfield,
/// makes the first touch a pass.
const PASS_SECONDS: f32 = 2.0;
/// A touch on goal this soon after a teammate's touch is a redirect.
const REDIRECT_SECONDS: f32 = 1.5;
/// Ball speed of a redirect: a real hit, not a deflection.
const REDIRECT_MINIMUM_BALL_SPEED: f32 = 1500.0;
/// Ball speed of a touch that can be a shot.
const SHOT_MINIMUM_BALL_SPEED: f32 = 1000.0;
/// Where a shot's projected flight meets the back wall: inside the goal mouth (ball centre
/// a margin inside the posts and crossbar) it is on target; inside the wider goal area
/// (about twice the mouth) it was aimed at goal and missed; farther out it was not a shot.
const ON_TARGET_HALF_WIDTH: f32 = 840.0;
const ON_TARGET_HEIGHT: f32 = 590.0;
const SHOT_AREA_HALF_WIDTH: f32 = 1800.0;
const SHOT_AREA_HEIGHT: f32 = 1300.0;
/// Flight projection: gravity, the ball's radius, the bounce kept off the floor, walls and
/// ceiling, the step and the longest flight followed.
const GRAVITY: f32 = 650.0;
const BALL_RADIUS: f32 = 93.0;
const BOUNCE_RESTITUTION: f32 = 0.6;
const SIDE_WALL_X: f32 = 4096.0;
const CEILING_Z: f32 = 2044.0;
const PROJECTION_STEP_SECONDS: f32 = 1.0 / 60.0;
/// Four seconds of flight.
const PROJECTION_MAXIMUM_STEPS: usize = 240;
/// The ball within this of a side wall, or of a back wall outside the goal mouth, is off
/// a wall (ball radius 93 + margin; the side walls are at ±4096, the back walls ±5120).
const BALL_ON_SIDE_WALL_X: f32 = 3950.0;
const BALL_ON_BACK_WALL_Y: f32 = 4975.0;
/// Goal mouth half-width and height (with the ball's radius).
const GOAL_MOUTH_HALF_WIDTH: f32 = 893.0;
const GOAL_MOUTH_HEIGHT: f32 = 642.0;
/// Ball height above which wall contact is a wall or backboard ball, not a roll.
const BALL_WALL_MINIMUM_HEIGHT: f32 = 150.0;
/// A clear this soon after the ball came off a wall or backboard in the player's own half
/// is a wall read.
const WALL_READ_SECONDS: f32 = 1.5;
/// Car speed at which a kickoff taker has started moving.
const KICKOFF_MOVING_SPEED: f32 = 100.0;
/// The back walls, where the goal lines are.
const BACK_WALL_Y: f32 = 5120.0;
/// Distance band to the ball that counts as useful support while a teammate has it.
const SUPPORT_MINIMUM_DISTANCE: f32 = 800.0;
const SUPPORT_MAXIMUM_DISTANCE: f32 = 3000.0;

/// `+1` for blue slots, `−1` for orange: multiplies y into the team-canonical frame.
const fn team_sign(slot: usize) -> f32 {
    if slot < SLOTS_PER_TEAM { 1.0 } else { -1.0 }
}

/// Positioning judged against the situation rather than as raw occupancy.
///
/// "Defending" means the ball is in the player's own half. The possessor is the car
/// closest to the ball over all six players.
fn accumulate_situational(
    accumulators: &mut [Accumulator; TOTAL_SLOTS],
    live_cars: &[LiveCar],
    ball_y: f32,
    possessor: Option<&LiveCar>,
    seconds: f64,
) {
    for team_start in [0, SLOTS_PER_TEAM] {
        let team: Vec<&LiveCar> = live_cars
            .iter()
            .filter(|car| (team_start..team_start + SLOTS_PER_TEAM).contains(&car.slot))
            .collect();
        let Some(first) = team.first() else {
            continue;
        };
        let sign = team_sign(first.slot);
        let ball_canonical_y = sign * ball_y;
        let defending = ball_canonical_y < 0.0;
        let last_back = team
            .iter()
            .min_by(|a, b| a.canonical_y.total_cmp(&b.canonical_y))
            .map(|car| car.slot);
        let closest_on_team = team
            .iter()
            .min_by(|a, b| a.distance_to_ball.total_cmp(&b.distance_to_ball))
            .map(|car| car.slot);
        let possessor_slot = possessor.map(|car| car.slot);
        let opponents_have_ball = possessor_slot
            .is_some_and(|slot| !(team_start..team_start + SLOTS_PER_TEAM).contains(&slot));
        let own_goal = Vector3 {
            x: 0.0,
            y: -sign * 5120.0,
            z: 0.0,
        };

        for car in &team {
            let Some(accumulator) = accumulators.get_mut(car.slot) else {
                continue;
            };
            let goal_side = car.canonical_y < ball_canonical_y;
            if defending {
                accumulator.defending_seconds += seconds;
                if goal_side {
                    accumulator.goal_side_defending += seconds;
                }
                if last_back == Some(car.slot) {
                    accumulator.last_back_defending += seconds;
                    accumulator.last_back_goal_distance_defending = seconds.mul_add(
                        f64::from(distance(&car.position, &own_goal)),
                        accumulator.last_back_goal_distance_defending,
                    );
                }
                if opponents_have_ball {
                    accumulator.opponent_possession_own_half += seconds;
                    // Shadowing: the defender facing the attacker stays goal-side and backs
                    // off toward the own goal instead of diving in.
                    if closest_on_team == Some(car.slot) {
                        accumulator.engaged_defending += seconds;
                        let retreating = sign * car.velocity.y < -SHADOW_MINIMUM_RETREAT_SPEED;
                        if goal_side && car.distance_to_ball < SHADOW_DISTANCE && retreating {
                            accumulator.shadow_defense += seconds;
                        }
                    }
                }
            }
            if !goal_side && closest_on_team != Some(car.slot) {
                accumulator.overcommit += seconds;
            }
            let teammate_has_ball = possessor_slot.is_some_and(|slot| {
                slot != car.slot && (team_start..team_start + SLOTS_PER_TEAM).contains(&slot)
            });
            if teammate_has_ball {
                accumulator.teammate_possession += seconds;
                if (SUPPORT_MINIMUM_DISTANCE..=SUPPORT_MAXIMUM_DISTANCE)
                    .contains(&car.distance_to_ball)
                {
                    accumulator.support_spacing += seconds;
                }
            }
        }
    }
}

/// Credits time spent double-committing: two or more teammates near the ball at once.
fn accumulate_double_commits(
    accumulators: &mut [Accumulator; TOTAL_SLOTS],
    live_cars: &[LiveCar],
    seconds: f64,
) {
    for team_start in [0, SLOTS_PER_TEAM] {
        let committed: Vec<usize> = live_cars
            .iter()
            .filter(|car| (team_start..team_start + SLOTS_PER_TEAM).contains(&car.slot))
            .filter(|car| car.distance_to_ball < DOUBLE_COMMIT_DISTANCE)
            .map(|car| car.slot)
            .collect();
        if committed.len() >= 2 {
            for slot in committed {
                if let Some(accumulator) = accumulators.get_mut(slot) {
                    accumulator.double_commit += seconds;
                }
            }
        }
    }
}

/// For every kickoff: who touched the ball first, and whether the ball was on the
/// opponents' side [`KICKOFF_OUTCOME_SECONDS`] later. Returns the number of kickoffs seen.
fn accumulate_kickoffs(
    accumulators: &mut [Accumulator; TOTAL_SLOTS],
    parsed: &ParsedReplay,
    slot_by_name: &HashMap<&str, usize>,
    frames: &Range<usize>,
) -> u32 {
    let mut kickoffs = 0;
    for &kickoff in parsed
        .kickoff_frames
        .iter()
        .filter(|kickoff| frames.contains(kickoff))
    {
        let Some(touch_index) = parsed
            .frames
            .iter()
            .enumerate()
            .skip(kickoff)
            .find(|(_, frame)| length(&frame.ball.velocity) > KICKOFF_TOUCH_BALL_SPEED)
            .map(|(index, _)| index)
        else {
            continue;
        };
        let Some(touch_frame) = parsed.frames.get(touch_index) else {
            continue;
        };
        let toucher = touch_frame
            .players
            .iter()
            .filter(|player| !player.actor_state.is_demolished)
            .min_by(|a, b| {
                distance(&a.actor_state.position, &touch_frame.ball.position).total_cmp(&distance(
                    &b.actor_state.position,
                    &touch_frame.ball.position,
                ))
            });
        let Some(toucher) = toucher else {
            continue;
        };
        kickoffs += 1;
        let Some(&slot) = slot_by_name.get(toucher.name.as_str()) else {
            continue;
        };
        let outcome_frame = parsed
            .frames
            .iter()
            .skip(touch_index)
            .find(|frame| frame.time - touch_frame.time >= KICKOFF_OUTCOME_SECONDS);
        accumulate_kickoff_approaches(accumulators, parsed, slot_by_name, kickoff, touch_index);
        if let Some(accumulator) = accumulators.get_mut(slot) {
            accumulator.kickoff_first_touches += 1;
            if let Some(outcome) = outcome_frame {
                let won = team_sign(slot) * outcome.ball.position.y > 0.0;
                accumulator.kickoff_wins += u32::from(won);
            }
        }
    }
    kickoffs
}

/// Each team's kickoff taker (its car closest to the ball at the first touch): distance
/// covered toward the ball from when they started moving to the first touch, over that
/// time, in units of supersonic speed. A speed flip or a fast flip gets there faster from the same spawn.
fn accumulate_kickoff_approaches(
    accumulators: &mut [Accumulator; TOTAL_SLOTS],
    parsed: &ParsedReplay,
    slot_by_name: &HashMap<&str, usize>,
    kickoff: usize,
    touch_index: usize,
) {
    let Some(touch_frame) = parsed.frames.get(touch_index) else {
        return;
    };
    for team in [Team::Blue, Team::Orange] {
        let taker = touch_frame
            .players
            .iter()
            .filter(|player| player.team == team && !player.actor_state.is_demolished)
            .min_by(|a, b| {
                distance(&a.actor_state.position, &touch_frame.ball.position).total_cmp(&distance(
                    &b.actor_state.position,
                    &touch_frame.ball.position,
                ))
            });
        let Some(taker) = taker else {
            continue;
        };
        let Some(&slot) = slot_by_name.get(taker.name.as_str()) else {
            continue;
        };
        let start = parsed
            .frames
            .iter()
            .take(touch_index)
            .skip(kickoff)
            .find_map(|frame| {
                frame
                    .players
                    .iter()
                    .find(|player| player.name == taker.name)
                    .filter(|player| length(&player.actor_state.velocity) > KICKOFF_MOVING_SPEED)
                    .map(|player| {
                        (
                            frame.time,
                            distance(&player.actor_state.position, &frame.ball.position),
                        )
                    })
            });
        let Some((start_time, start_distance)) = start else {
            continue;
        };
        let seconds = touch_frame.time - start_time;
        if seconds <= 0.0 {
            continue;
        }
        // Distance covered toward the ball: the taker who did not touch first is not there yet.
        let covered =
            start_distance - distance(&taker.actor_state.position, &touch_frame.ball.position);
        if let Some(accumulator) = accumulators.get_mut(slot) {
            accumulator.kickoffs_taken += 1;
            accumulator.kickoff_approach_speed += f64::from(covered / seconds / SUPERSONIC_SPEED);
        }
    }
}

/// Per-frame team roles: closest to ball, last back, most forward, nearest-teammate spacing.
fn accumulate_team_roles(
    accumulators: &mut [Accumulator; TOTAL_SLOTS],
    live_cars: &[LiveCar],
    seconds: f64,
) {
    for team_start in [0, SLOTS_PER_TEAM] {
        let team: Vec<&LiveCar> = live_cars
            .iter()
            .filter(|car| (team_start..team_start + SLOTS_PER_TEAM).contains(&car.slot))
            .collect();
        if team.len() < 2 {
            continue;
        }
        let closest = team
            .iter()
            .min_by(|a, b| a.distance_to_ball.total_cmp(&b.distance_to_ball));
        let last_back = team
            .iter()
            .min_by(|a, b| a.canonical_y.total_cmp(&b.canonical_y));
        let most_forward = team
            .iter()
            .max_by(|a, b| a.canonical_y.total_cmp(&b.canonical_y));
        if let Some(car) = closest
            && let Some(accumulator) = accumulators.get_mut(car.slot)
        {
            accumulator.closest_on_team += seconds;
        }
        if let Some(car) = last_back
            && let Some(accumulator) = accumulators.get_mut(car.slot)
        {
            accumulator.last_back_on_team += seconds;
        }
        if let Some(car) = most_forward
            && let Some(accumulator) = accumulators.get_mut(car.slot)
        {
            accumulator.most_forward_on_team += seconds;
        }
        for car in &team {
            let nearest = team
                .iter()
                .filter(|other| other.slot != car.slot)
                .map(|other| distance(&car.position, &other.position))
                .fold(f32::MAX, f32::min);
            if let Some(accumulator) = accumulators.get_mut(car.slot) {
                accumulator.nearest_teammate_distance =
                    seconds.mul_add(f64::from(nearest), accumulator.nearest_teammate_distance);
                accumulator.nearest_teammate_seconds += seconds;
            }
        }
    }
}

fn finish(
    accumulator: &Accumulator,
    scoreboard: Option<&replay_structs::HeaderPlayerStats>,
    kickoff_count: u32,
    players_per_team: usize,
) -> Option<PlayerMatchStats> {
    if accumulator.seconds <= 0.0 {
        return None;
    }
    let seconds = accumulator.seconds;
    let minutes = seconds / 60.0;
    let fraction = |value: f64| (value / seconds) as f32;
    let per_minute = |value: f64| (value / minutes) as f32;
    let ratio = |numerator: f64, denominator: f64| {
        if denominator > 0.0 {
            (numerator / denominator) as f32
        } else {
            0.0
        }
    };
    let per_dodge = |count: u32| {
        if accumulator.dodges == 0 {
            0.0
        } else {
            count as f32 / accumulator.dodges as f32
        }
    };
    let per_touch = |value: f64| {
        if accumulator.touches == 0 {
            0.0
        } else {
            (value / f64::from(accumulator.touches)) as f32
        }
    };
    let value = |stat: MatchStat| match stat {
        MatchStat::MeanSpeed => fraction(accumulator.speed) / 2300.0,
        MatchStat::SupersonicFraction => fraction(accumulator.supersonic),
        MatchStat::SlowFraction => fraction(accumulator.slow),
        MatchStat::GroundFraction => fraction(accumulator.ground),
        MatchStat::LowAirFraction => fraction(accumulator.low_air),
        MatchStat::HighAirFraction => fraction(accumulator.high_air),
        MatchStat::WallFraction => fraction(accumulator.wall),
        MatchStat::MeanHeight => fraction(accumulator.height) / 2044.0,
        MatchStat::JumpsPerMinute => per_minute(f64::from(accumulator.jumps)),
        MatchStat::MeanBoost => fraction(accumulator.boost),
        MatchStat::BoostEmptyFraction => fraction(accumulator.boost_empty),
        MatchStat::BoostFullFraction => fraction(accumulator.boost_full),
        MatchStat::BoostCollectedPerMinute => per_minute(accumulator.boost_collected * 100.0),
        MatchStat::BoostUsedPerMinute => per_minute(accumulator.boost_used * 100.0),
        MatchStat::BigPadsPerMinute => per_minute(f64::from(accumulator.big_pads)),
        MatchStat::SmallPadsPerMinute => per_minute(f64::from(accumulator.small_pads)),
        MatchStat::BoostWhileSupersonicFraction => fraction(accumulator.boost_while_supersonic),
        MatchStat::DefensiveThirdFraction => fraction(accumulator.defensive_third),
        MatchStat::OffensiveThirdFraction => fraction(accumulator.offensive_third),
        MatchStat::MeanCanonicalY => fraction(accumulator.canonical_y) / 5120.0,
        MatchStat::BehindBallFraction => fraction(accumulator.behind_ball),
        MatchStat::MeanDistanceToBall => fraction(accumulator.distance_to_ball) / 7245.0,
        MatchStat::ClosestOnTeamFraction => fraction(accumulator.closest_on_team),
        MatchStat::LastBackOnTeamFraction => fraction(accumulator.last_back_on_team),
        MatchStat::MostForwardOnTeamFraction => fraction(accumulator.most_forward_on_team),
        MatchStat::MeanDistanceToNearestTeammate => {
            if accumulator.nearest_teammate_seconds > 0.0 {
                (accumulator.nearest_teammate_distance / accumulator.nearest_teammate_seconds)
                    as f32
                    / 7245.0
            } else {
                0.0
            }
        }
        MatchStat::MeanDistanceToOwnGoal => fraction(accumulator.distance_to_own_goal) / 7245.0,
        MatchStat::PossessionFraction => fraction(accumulator.possession),
        MatchStat::TouchesPerMinute => per_minute(f64::from(accumulator.touches)),
        MatchStat::MeanBallSpeedAfterTouch => {
            per_touch(accumulator.ball_speed_after_touch) / 6000.0
        }
        MatchStat::AerialTouchFraction => per_touch(f64::from(accumulator.aerial_touches)),
        MatchStat::ForwardTouchFraction => per_touch(f64::from(accumulator.forward_touches)),
        MatchStat::MeanTouchSpeedGain => per_touch(accumulator.touch_speed_gain) / 6000.0,
        MatchStat::Goals => accumulator.goals as f32,
        MatchStat::DemosReceived => accumulator.demos_received as f32,
        MatchStat::AirborneUprightFraction => {
            if accumulator.airborne_seconds > 0.0 {
                (accumulator.airborne_upright / accumulator.airborne_seconds) as f32
            } else {
                0.0
            }
        }
        MatchStat::SlowWithBoostFraction => fraction(accumulator.slow_with_boost),
        MatchStat::SupersonicPerBoost => {
            if accumulator.boost_used > 0.0 {
                (accumulator.supersonic / (accumulator.boost_used * 100.0)) as f32
            } else {
                0.0
            }
        }
        MatchStat::JumpsPressedPerMinute => per_minute(f64::from(accumulator.jump_presses)),
        MatchStat::DoubleJumpsPerMinute => per_minute(f64::from(accumulator.double_jumps)),
        MatchStat::DodgesPerMinute => per_minute(f64::from(accumulator.dodges)),
        MatchStat::WavedashLikeDodgesPerMinute => {
            per_minute(f64::from(accumulator.wavedash_like_dodges))
        }
        MatchStat::AerialDodgesPerMinute => per_minute(f64::from(accumulator.aerial_dodges)),
        MatchStat::PitchPositiveDodgeFraction => per_dodge(accumulator.forward_dodges),
        MatchStat::PitchNegativeDodgeFraction => per_dodge(accumulator.backward_dodges),
        MatchStat::RollDodgeFraction => per_dodge(accumulator.sideways_dodges),
        MatchStat::DiagonalDodgeFraction => per_dodge(accumulator.diagonal_dodges),
        MatchStat::FastDodgeFraction => per_dodge(accumulator.fast_dodges),
        MatchStat::BoostingFraction => fraction(accumulator.boosting),
        MatchStat::PowerslideFraction => fraction(accumulator.powerslide),
        MatchStat::AirRollFraction => fraction(accumulator.air_roll),
        MatchStat::ReverseFraction => fraction(accumulator.reverse),
        MatchStat::MeanAbsoluteSteer => fraction(accumulator.absolute_steer),
        MatchStat::MeanAirAngularSpeed => {
            if accumulator.airborne_seconds > 0.0 {
                (accumulator.air_angular_speed / accumulator.airborne_seconds) as f32
            } else {
                0.0
            }
        }
        MatchStat::FastAerialFraction => {
            if accumulator.double_jumps > 0 {
                accumulator.fast_aerials as f32 / accumulator.double_jumps as f32
            } else {
                0.0
            }
        }
        MatchStat::ScoreboardScorePerMinute => {
            scoreboard.map_or(0.0, |s| s.score as f32 / minutes as f32)
        }
        MatchStat::ScoreboardAssists => scoreboard.map_or(0.0, |s| s.assists as f32),
        MatchStat::ScoreboardSaves => scoreboard.map_or(0.0, |s| s.saves as f32),
        MatchStat::ScoreboardShots => scoreboard.map_or(0.0, |s| s.shots as f32),
        MatchStat::ScoreboardGoalsPerShot => scoreboard.map_or(0.0, |s| {
            if s.shots > 0 {
                s.goals as f32 / s.shots as f32
            } else {
                0.0
            }
        }),
        MatchStat::DemosInflicted => accumulator.demos_inflicted as f32,
        MatchStat::KickoffFirstTouchFraction => {
            if kickoff_count > 0 {
                accumulator.kickoff_first_touches as f32 / kickoff_count as f32
            } else {
                0.0
            }
        }
        MatchStat::KickoffWinFraction => {
            if accumulator.kickoff_first_touches > 0 {
                accumulator.kickoff_wins as f32 / accumulator.kickoff_first_touches as f32
            } else {
                0.0
            }
        }
        MatchStat::DoubleCommitFraction => fraction(accumulator.double_commit),
        MatchStat::LastBackWhenDefendingFraction => ratio(
            accumulator.last_back_defending,
            accumulator.defending_seconds,
        ),
        MatchStat::GoalSideWhenDefendingFraction => ratio(
            accumulator.goal_side_defending,
            accumulator.defending_seconds,
        ),
        MatchStat::LastBackGoalDistanceWhenDefending => {
            ratio(
                accumulator.last_back_goal_distance_defending,
                accumulator.last_back_defending,
            ) / 7245.0
        }
        MatchStat::OvercommitFraction => fraction(accumulator.overcommit),
        MatchStat::ShadowDefenseFraction => {
            ratio(accumulator.shadow_defense, accumulator.engaged_defending)
        }
        MatchStat::SupportSpacingFraction => {
            ratio(accumulator.support_spacing, accumulator.teammate_possession)
        }
        MatchStat::GoalSideAfterTouchFraction => ratio(
            f64::from(accumulator.recovered_goal_side),
            f64::from(accumulator.recovery_touches),
        ),
        MatchStat::SpeedFlipsPerMinute => per_minute(f64::from(accumulator.speed_flips)),
        MatchStat::HalfFlipsPerMinute => per_minute(f64::from(accumulator.half_flips)),
        MatchStat::WavedashesPerMinute => per_minute(f64::from(accumulator.wavedashes)),
        MatchStat::TornadoSpinsPerMinute => per_minute(f64::from(accumulator.tornado_spins)),
        MatchStat::FlipResetsPerMinute => per_minute(f64::from(accumulator.flip_resets)),
        MatchStat::CeilingTouchesPerMinute => per_minute(f64::from(accumulator.ceiling_touches)),
        MatchStat::DribbleFraction => fraction(accumulator.dribble),
        MatchStat::FlicksPerMinute => per_minute(f64::from(accumulator.flicks)),
        MatchStat::AirDribbleTouchesPerMinute => {
            per_minute(f64::from(accumulator.air_dribble_touches))
        }
        MatchStat::DoubleTapsPerMinute => per_minute(f64::from(accumulator.double_taps)),
        MatchStat::WallTouchFraction => per_touch(f64::from(accumulator.wall_touches)),
        MatchStat::FlipTouchFraction => per_touch(f64::from(accumulator.flip_touches)),
        MatchStat::PowerShotsPerMinute => per_minute(f64::from(accumulator.power_shots)),
        MatchStat::FiftyFiftiesPerMinute => per_minute(f64::from(accumulator.fifty_fifties)),
        MatchStat::FiftyFiftyWinFraction => ratio(
            f64::from(accumulator.fifty_fifty_wins),
            f64::from(accumulator.fifty_fifties),
        ),
        MatchStat::BumpsInflictedPerMinute => per_minute(f64::from(accumulator.bumps_inflicted)),
        MatchStat::BumpsReceivedPerMinute => per_minute(f64::from(accumulator.bumps_received)),
        MatchStat::TeammateBumpsPerMinute => per_minute(f64::from(accumulator.teammate_bumps)),
        MatchStat::OwnGoals => accumulator.own_goals as f32,
        MatchStat::LostByForfeit => f32::from(u8::from(accumulator.lost_by_forfeit)),
        MatchStat::ForfeitedLevelOrAhead => {
            f32::from(u8::from(accumulator.forfeited_level_or_ahead))
        }
        MatchStat::CleanLandingFraction => ratio(
            f64::from(accumulator.clean_landings),
            f64::from(accumulator.landings),
        ),
        MatchStat::LandingSpeedKeptFraction => ratio(
            accumulator.landing_speed_kept,
            f64::from(accumulator.landings_with_speed),
        ),
        MatchStat::KickoffApproachSpeed => ratio(
            accumulator.kickoff_approach_speed,
            f64::from(accumulator.kickoffs_taken),
        ),
        MatchStat::PassesPerMinute => per_minute(f64::from(accumulator.passes)),
        MatchStat::RedirectsPerMinute => per_minute(f64::from(accumulator.redirects)),
        MatchStat::WallReadsPerMinute => per_minute(f64::from(accumulator.wall_reads)),
        MatchStat::ShotAccuracy => ratio(
            f64::from(accumulator.shots_on_target),
            f64::from(accumulator.shot_attempts),
        ),
        MatchStat::PlayersPerTeam => players_per_team as f32,
    };
    let mut values = [0.0; MATCH_STAT_COUNT];
    for stat in MatchStat::all() {
        values[stat.index()] = value(stat);
    }
    Some(PlayerMatchStats {
        values,
        live_seconds: seconds as f32,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use replay_structs::{ActorState, BallState, GameFrame, PlayerState, Team};

    use super::*;

    fn car(name: &str, team: Team, x: f32, y: f32, boost: f32) -> PlayerState {
        PlayerState {
            actor_id: 0,
            name: Arc::new(name.to_string()),
            team,
            actor_state: ActorState {
                position: Vector3 { x, y, z: 17.0 },
                velocity: Vector3 {
                    x: 0.0,
                    y: 1000.0,
                    z: 0.0,
                },
                rotation: Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                boost,
                is_demolished: false,
                ..ActorState::default()
            },
        }
    }

    fn roster() -> PlayerRoster {
        PlayerRoster {
            names: core::array::from_fn(|slot| match slot {
                0..3 => format!("b{slot}"),
                _ if (SLOTS_PER_TEAM..SLOTS_PER_TEAM + 3).contains(&slot) => {
                    format!("o{}", slot - SLOTS_PER_TEAM)
                }
                _ => String::new(),
            }),
        }
    }

    fn frame(blue_y: f32, orange_y: f32) -> GameFrame {
        GameFrame {
            time: 0.0,
            delta: 1.0 / 30.0,
            seconds_remaining: 200,
            ball: BallState::default(),
            demolitions: Vec::new(),
            players: vec![
                car("b0", Team::Blue, 0.0, blue_y, 0.5),
                car("b1", Team::Blue, 500.0, blue_y + 100.0, 0.5),
                car("b2", Team::Blue, -500.0, blue_y + 200.0, 0.5),
                car("o0", Team::Orange, 0.0, orange_y, 0.5),
                car("o1", Team::Orange, 500.0, orange_y - 100.0, 0.5),
                car("o2", Team::Orange, -500.0, orange_y - 200.0, 0.5),
            ],
        }
    }

    /// Mirrored positions produce identical positional stats for the two teams.
    #[test]
    fn positions_are_team_canonical() {
        let parsed = ParsedReplay {
            frames: (0..60).map(|_| frame(-4000.0, 4000.0)).collect(),
            ..ParsedReplay::default()
        };
        let stats = compute_player_match_stats(&parsed, &roster());
        let blue = stats[0]
            .as_ref()
            .map_or([0.0; MATCH_STAT_COUNT], |s| s.values);
        let orange = stats[SLOTS_PER_TEAM]
            .as_ref()
            .map_or([0.0; MATCH_STAT_COUNT], |s| s.values);
        let defensive = MatchStat::DefensiveThirdFraction.index();
        assert!((blue[defensive] - 1.0).abs() < 1e-6);
        assert!((orange[defensive] - 1.0).abs() < 1e-6);
        let last_back = MatchStat::LastBackOnTeamFraction.index();
        assert!((blue[last_back] - 1.0).abs() < 1e-6);
        assert!((orange[last_back] - 1.0).abs() < 1e-6);
        assert!(
            (blue[MatchStat::MeanCanonicalY.index()] - orange[MatchStat::MeanCanonicalY.index()])
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn every_rostered_player_gets_stats() {
        let parsed = ParsedReplay {
            frames: vec![frame(0.0, 0.0), frame(0.0, 0.0)],
            ..ParsedReplay::default()
        };
        let stats = compute_player_match_stats(&parsed, &roster());
        let roster = roster();
        for (slot, name) in roster.names.iter().enumerate() {
            assert_eq!(stats[slot].is_some(), !name.is_empty(), "slot {slot}");
        }
        let players_per_team = stats[0]
            .as_ref()
            .map_or(0.0, |s| s.values[MatchStat::PlayersPerTeam.index()]);
        assert!((players_per_team - 3.0).abs() < f32::EPSILON);
    }

    /// A big pad pickup is counted once and the boost gain is recorded.
    #[test]
    fn big_pad_pickup_is_counted() {
        let mut frames: Vec<GameFrame> = (0..30).map(|_| frame(0.0, 0.0)).collect();
        for later in frames.iter_mut().skip(15) {
            later.players[0].actor_state.boost = 1.0;
        }
        let parsed = ParsedReplay {
            frames,
            ..ParsedReplay::default()
        };
        let stats = compute_player_match_stats(&parsed, &roster());
        let values = stats[0]
            .as_ref()
            .map_or([0.0; MATCH_STAT_COUNT], |s| s.values);
        let seconds = 30.0 / 30.0;
        assert!((values[MatchStat::BigPadsPerMinute.index()] - 60.0 / seconds).abs() < 1e-3);
    }

    fn stats_of_first_car(frames: Vec<GameFrame>) -> [f32; MATCH_STAT_COUNT] {
        let parsed = ParsedReplay {
            frames,
            ..ParsedReplay::default()
        };
        compute_player_match_stats(&parsed, &roster())[0]
            .as_ref()
            .map_or([0.0; MATCH_STAT_COUNT], |s| s.values)
    }

    /// A second flip in one airborne stretch, after the wheels reached the ball, is a flip
    /// reset; one flip is not.
    #[test]
    fn a_second_flip_near_the_ball_is_a_flip_reset() {
        let airborne = |double_jump: bool, dodge: bool| {
            let mut frames: Vec<GameFrame> = (0..40).map(|_| frame(0.0, 0.0)).collect();
            for (index, later) in frames.iter_mut().enumerate().skip(1) {
                later.ball.position = Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 600.0,
                };
                let car = &mut later.players[0].actor_state;
                car.position.z = 500.0;
                car.controls.double_jumping = double_jump && index >= 10;
                car.controls.dodging = dodge && index >= 20;
            }
            stats_of_first_car(frames)[MatchStat::FlipResetsPerMinute.index()]
        };
        assert!(airborne(true, true) > 0.0);
        assert!(airborne(true, false).abs() < f32::EPSILON);
    }

    /// Coming down from the air on the wheels is a clean landing; on the roof it is not.
    #[test]
    fn landing_on_the_wheels_is_clean() {
        let landing = |rotation: Quaternion| {
            let mut frames: Vec<GameFrame> = (0..60).map(|_| frame(0.0, 0.0)).collect();
            for (index, later) in frames.iter_mut().enumerate() {
                let car = &mut later.players[0].actor_state;
                // Up for a second, then down at ~25 per frame.
                if (5..30).contains(&index) {
                    car.position.z = 300.0;
                } else if (30..42).contains(&index) {
                    car.position.z = ((index - 29) as f32).mul_add(-25.0, 300.0);
                    car.velocity.z = -750.0;
                    car.rotation = rotation;
                } else if index >= 42 {
                    car.rotation = rotation;
                }
            }
            stats_of_first_car(frames)[MatchStat::CleanLandingFraction.index()]
        };
        let wheels_down = Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        };
        // Half a turn about the forward axis: on the roof.
        let on_the_roof = Quaternion {
            x: 1.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        };
        assert!((landing(wheels_down) - 1.0).abs() < f32::EPSILON);
        assert!(landing(on_the_roof).abs() < f32::EPSILON);
    }

    /// Driving on at the landing's speed keeps all of it; stopping dead keeps none.
    #[test]
    fn landing_speed_kept_is_measured_after_touchdown() {
        let landing = |speed_after: f32| {
            let mut frames: Vec<GameFrame> = (0..70).map(|_| frame(0.0, 0.0)).collect();
            for (index, later) in frames.iter_mut().enumerate() {
                let car = &mut later.players[0].actor_state;
                car.rotation.w = 1.0;
                car.velocity.y = 1000.0;
                if (5..30).contains(&index) {
                    car.position.z = 300.0;
                } else if (30..42).contains(&index) {
                    car.position.z = ((index - 29) as f32).mul_add(-25.0, 300.0);
                    car.velocity.z = -750.0;
                } else if index >= 42 {
                    car.velocity.y = speed_after;
                }
            }
            stats_of_first_car(frames)[MatchStat::LandingSpeedKeptFraction.index()]
        };
        assert!((landing(1000.0) - 1.0).abs() < 1e-3);
        assert!(landing(0.0).abs() < 1e-3);
    }

    /// A second of fast, boosting air roll with the nose held steady is a tornado spin; the
    /// same roll with the nose swinging around is not.
    #[test]
    fn a_steady_boosting_air_roll_is_a_tornado_spin() {
        let spin = |nose_drift_per_frame: f32| {
            let mut frames: Vec<GameFrame> = (0..60).map(|_| frame(0.0, 0.0)).collect();
            for (index, later) in frames.iter_mut().enumerate() {
                let car = &mut later.players[0].actor_state;
                car.position.z = 800.0;
                car.controls.boosting = true;
                let nose = |rotation: &Quaternion| nose_direction(rotation);
                // Nose 30° up, yaw drifting by `nose_drift_per_frame`, rolling 5 rad/s about
                // the nose (yaw, then pitch, then roll).
                let yaw = index as f32 * nose_drift_per_frame;
                let pitch = -30.0f32.to_radians();
                let roll = 5.0 * index as f32 / 30.0;
                let (yaw_sine, yaw_cosine) = (yaw / 2.0).sin_cos();
                let (pitch_sine, pitch_cosine) = (pitch / 2.0).sin_cos();
                let (roll_sine, roll_cosine) = (roll / 2.0).sin_cos();
                car.rotation = Quaternion {
                    w: (yaw_cosine * pitch_cosine)
                        .mul_add(roll_cosine, yaw_sine * pitch_sine * roll_sine),
                    x: (yaw_cosine * pitch_cosine)
                        .mul_add(roll_sine, -(yaw_sine * pitch_sine * roll_cosine)),
                    y: (yaw_cosine * pitch_sine)
                        .mul_add(roll_cosine, yaw_sine * pitch_cosine * roll_sine),
                    z: (yaw_sine * pitch_cosine)
                        .mul_add(roll_cosine, -(yaw_cosine * pitch_sine * roll_sine)),
                };
                // 5 rad/s about the nose, in the replay's hundredths.
                let direction = nose(&car.rotation);
                car.angular_velocity = Vector3 {
                    x: direction.x * 500.0,
                    y: direction.y * 500.0,
                    z: direction.z * 500.0,
                };
            }
            stats_of_first_car(frames)[MatchStat::TornadoSpinsPerMinute.index()]
        };
        assert!(spin(0.0) > 0.0);
        assert!(spin(0.1).abs() < f32::EPSILON);
    }

    /// Driving into an opponent who gets knocked back is a bump for the driver.
    #[test]
    fn knocking_an_opponent_back_is_a_bump() {
        let mut frames: Vec<GameFrame> = (0..20).map(|_| frame(0.0, 150.0)).collect();
        for (index, later) in frames.iter_mut().enumerate() {
            later.players[0].actor_state.velocity.y = 1500.0;
            later.players[3].actor_state.velocity.y = if index < 10 { -500.0 } else { 1500.0 };
        }
        let parsed = ParsedReplay {
            frames,
            ..ParsedReplay::default()
        };
        let stats = compute_player_match_stats(&parsed, &roster());
        let value = |slot: usize, stat: MatchStat| {
            stats[slot].as_ref().map_or(0.0, |s| s.values[stat.index()])
        };
        assert!(value(0, MatchStat::BumpsInflictedPerMinute) > 0.0);
        assert!(value(SLOTS_PER_TEAM, MatchStat::BumpsReceivedPerMinute) > 0.0);
        assert!(value(SLOTS_PER_TEAM, MatchStat::BumpsInflictedPerMinute).abs() < f32::EPSILON);
    }

    /// A fast, boosting, diagonal front flip off the ground is a speed flip.
    #[test]
    fn a_fast_boosting_diagonal_flip_is_a_speed_flip() {
        let mut frames: Vec<GameFrame> = (0..30).map(|_| frame(0.0, 0.0)).collect();
        for (index, later) in frames.iter_mut().enumerate() {
            let car = &mut later.players[0].actor_state;
            car.velocity.y = 1500.0;
            car.controls.boosting = true;
            car.controls.jumping = index >= 5;
            if index >= 7 {
                car.position.z = 60.0;
                car.controls.dodging = true;
                car.controls.dodge_torque = Vector3 {
                    x: 0.7,
                    y: 0.7,
                    z: 0.0,
                };
            }
        }
        assert!(stats_of_first_car(frames)[MatchStat::SpeedFlipsPerMinute.index()] > 0.0);
    }
}

/// Splits a replay into windows that each end just after a goal.
///
/// Kickoff → goal, kickoff → goal, …, last kickoff → end. Goal-replay frames fall inside
/// the window after the goal and are skipped by the stats themselves.
#[must_use]
pub fn goal_windows(parsed: &ParsedReplay) -> Vec<Range<usize>> {
    let mut boundaries: Vec<usize> = parsed
        .goal_frames
        .iter()
        .map(|&frame| frame + 1)
        .filter(|&frame| frame < parsed.frames.len())
        .collect();
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut windows = Vec::with_capacity(boundaries.len() + 1);
    let mut start = 0;
    for boundary in boundaries {
        if boundary > start {
            windows.push(start..boundary);
            start = boundary;
        }
    }
    if start < parsed.frames.len() {
        windows.push(start..parsed.frames.len());
    }
    windows
}

/// Splits a replay into consecutive windows of about `seconds` of **live** play (goal
/// replays excluded). A trailing remainder shorter than half a window joins the previous
/// window.
#[must_use]
pub fn time_windows(parsed: &ParsedReplay, seconds: f32) -> Vec<Range<usize>> {
    let excluded = build_goal_replay_excluded_set(&parsed.goal_frames, &parsed.kickoff_frames);
    let mut windows = Vec::new();
    let mut start = 0;
    let mut live = 0.0;
    for (index, frame) in parsed.frames.iter().enumerate() {
        if !excluded.contains(&index) {
            live += frame.delta.clamp(0.0, MAXIMUM_FRAME_SECONDS);
        }
        if live >= seconds {
            windows.push(start..index + 1);
            start = index + 1;
            live = 0.0;
        }
    }
    if start < parsed.frames.len() {
        if live < seconds / 2.0
            && let Some(last) = windows.last_mut()
        {
            last.end = parsed.frames.len();
        } else {
            windows.push(start..parsed.frames.len());
        }
    }
    windows
}

#[cfg(test)]
mod window_tests {
    use super::*;

    fn frames(count: usize) -> Vec<replay_structs::GameFrame> {
        (0..count)
            .map(|index| replay_structs::GameFrame {
                time: index as f32 / 30.0,
                delta: 1.0 / 30.0,
                ..replay_structs::GameFrame::default()
            })
            .collect()
    }

    /// Goal windows tile the replay exactly, cutting just after each goal.
    #[test]
    fn goal_windows_tile_the_replay() {
        let parsed = ParsedReplay {
            frames: frames(1000),
            goal_frames: vec![300, 700],
            ..ParsedReplay::default()
        };
        let windows = goal_windows(&parsed);
        assert_eq!(windows, vec![0..301, 301..701, 701..1000]);
    }

    /// Time windows tile the replay; a short remainder joins the last window.
    #[test]
    fn time_windows_tile_the_replay() {
        let parsed = ParsedReplay {
            frames: frames(30 * 130),
            ..ParsedReplay::default()
        };
        let windows = time_windows(&parsed, 60.0);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows.first().map(|w| w.start), Some(0));
        assert_eq!(windows.last().map(|w| w.end), Some(30 * 130));
        for pair in windows.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "windows must be contiguous");
        }
    }
}

#[cfg(test)]
mod situational_tests {
    use std::sync::Arc;

    use replay_structs::{ActorState, BallState, GameFrame, PlayerState, Team};

    use super::*;

    fn car(name: &str, team: Team, x: f32, y: f32) -> PlayerState {
        PlayerState {
            actor_id: 0,
            name: Arc::new(name.to_string()),
            team,
            actor_state: ActorState {
                position: Vector3 { x, y, z: 17.0 },
                rotation: Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                ..ActorState::default()
            },
        }
    }

    fn stat(values: &[f32; MATCH_STAT_COUNT], stat: MatchStat) -> f32 {
        values[stat.index()]
    }

    /// Ball deep in blue's half, an orange attacker on it. Blue: b0 in net (last back,
    /// goal-side), b1 shadowing goal-side near the ball, b2 stranded upfield (overcommitted).
    #[test]
    fn defending_roles_are_credited_to_the_right_players() {
        let frame = GameFrame {
            time: 0.0,
            delta: 1.0 / 30.0,
            seconds_remaining: 200,
            ball: BallState {
                position: Vector3 {
                    x: 0.0,
                    y: -3000.0,
                    z: 93.0,
                },
                ..BallState::default()
            },
            players: vec![
                car("b0", Team::Blue, 0.0, -5000.0),
                car("b1", Team::Blue, 300.0, -3800.0),
                car("b2", Team::Blue, 0.0, 2000.0),
                car("o0", Team::Orange, 0.0, -2900.0),
                car("o1", Team::Orange, 1000.0, 0.0),
                car("o2", Team::Orange, 0.0, 4000.0),
            ],
            demolitions: Vec::new(),
        };
        let mut frame = frame;
        // b1 backs off toward its goal: shadowing, not diving in.
        frame.players[1].actor_state.velocity.y = -500.0;
        let parsed = ParsedReplay {
            frames: vec![frame; 60],
            ..ParsedReplay::default()
        };
        let roster = PlayerRoster {
            names: core::array::from_fn(|slot| match slot {
                0..3 => format!("b{slot}"),
                _ if (SLOTS_PER_TEAM..SLOTS_PER_TEAM + 3).contains(&slot) => {
                    format!("o{}", slot - SLOTS_PER_TEAM)
                }
                _ => String::new(),
            }),
        };
        let stats = compute_player_match_stats(&parsed, &roster);
        let values = |slot: usize| {
            stats[slot]
                .as_ref()
                .map_or([0.0; MATCH_STAT_COUNT], |s| s.values)
        };

        assert!((stat(&values(0), MatchStat::LastBackWhenDefendingFraction) - 1.0).abs() < 1e-6);
        assert!((stat(&values(1), MatchStat::LastBackWhenDefendingFraction)).abs() < 1e-6);
        assert!((stat(&values(0), MatchStat::GoalSideWhenDefendingFraction) - 1.0).abs() < 1e-6);
        assert!((stat(&values(2), MatchStat::GoalSideWhenDefendingFraction)).abs() < 1e-6);
        assert!((stat(&values(1), MatchStat::ShadowDefenseFraction) - 1.0).abs() < 1e-6);
        assert!((stat(&values(0), MatchStat::ShadowDefenseFraction)).abs() < 1e-6);
        assert!((stat(&values(2), MatchStat::OvercommitFraction) - 1.0).abs() < 1e-6);
        assert!((stat(&values(0), MatchStat::OvercommitFraction)).abs() < 1e-6);
        // Orange is attacking, not defending.
        assert!(
            (stat(
                &values(SLOTS_PER_TEAM),
                MatchStat::GoalSideWhenDefendingFraction
            ))
            .abs()
                < 1e-6
        );
    }
}
