//! Replay parser crate for Rocket League replays.
//!
//! This crate wraps the `boxcars` library to parse `.replay` files
//! into structured Rust types suitable for ML feature extraction.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use boxcars::{Attribute, HeaderProp, ParserBuilder, Replay};
use replay_structs::{
    ActorState, BallState, DemolitionEvent, GameFrame, GoalEvent, HeaderPlayerStats, MatchFormat,
    MatchOutcome, ParsedReplay, PlayerState, Team, UnsupportedReplayMatch,
};

pub mod match_validation;

/// Parsing or policy errors when loading a replay.
#[derive(Debug)]
pub enum ReplayAcceptanceError {
    /// Replay is valid but not allowed for this product.
    Unsupported(UnsupportedReplayMatch),
    /// Low-level parse or IO failure.
    Parse(anyhow::Error),
}

impl std::fmt::Display for ReplayAcceptanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(details) => write!(f, "{}", details.user_message()),
            Self::Parse(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ReplayAcceptanceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(err) => Some(err.as_ref()),
            Self::Unsupported(_) => None,
        }
    }
}

/// Internal player info used during parsing (different from `replay_structs::PlayerInfo`).
#[derive(Debug, Default)]
struct PlayerInfo {
    name: String,
    team: Option<Team>,
}

/// Recursively find PRI actor for a given car.
/// Strategy: Find components that reference the car, then check what those components reference.
fn find_pri_for_car(
    car_id: i32,
    actor_links: &HashMap<i32, Vec<i32>>, // Maps actor -> what it references via ActiveActor
    reverse_actor_links: &HashMap<i32, Vec<i32>>, // Maps actor -> what references it
    player_infos: &HashMap<i32, PlayerInfo>,
    visited: &mut std::collections::HashSet<i32>,
) -> Option<i32> {
    if visited.contains(&car_id) {
        return None; // Cycle detection
    }
    visited.insert(car_id);

    // Find all actors that reference this car (components)
    if let Some(referencers) = reverse_actor_links.get(&car_id) {
        for &referencer in referencers {
            // Check if this referencer is a PRI (name may come later)
            if player_infos.contains_key(&referencer) {
                return Some(referencer);
            }

            // Check all actors that this referencer references - one might be a PRI
            if let Some(referenced_list) = actor_links.get(&referencer) {
                for &referenced_by_component in referenced_list {
                    // Check if the referenced actor is a PRI
                    if player_infos.contains_key(&referenced_by_component) {
                        return Some(referenced_by_component);
                    }
                    // Recursively check
                    if let Some(pri_id) = find_pri_for_car(
                        referenced_by_component,
                        actor_links,
                        reverse_actor_links,
                        player_infos,
                        visited,
                    ) {
                        return Some(pri_id);
                    }
                }
            }
        }
    }

    None
}

/// Update `car_to_pri` mapping for a specific car.
fn update_car_to_pri_mapping(
    car_id: i32,
    actor_links: &HashMap<i32, Vec<i32>>,
    reverse_actor_links: &HashMap<i32, Vec<i32>>,
    car_to_actor_id: &mut HashMap<i32, i32>,
    player_infos: &HashMap<i32, PlayerInfo>,
) {
    let mut visited = std::collections::HashSet::new();
    if let Some(pri_id) = find_pri_for_car(
        car_id,
        actor_links,
        reverse_actor_links,
        player_infos,
        &mut visited,
    ) {
        car_to_actor_id.insert(car_id, pri_id);
    }
}

/// Parses a replay file from the given path.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed, or the match is not supported.
pub fn parse_replay(path: &Path) -> Result<ParsedReplay, ReplayAcceptanceError> {
    let raw_data = std::fs::read(path).map_err(|io| ReplayAcceptanceError::Parse(io.into()))?;
    parse_replay_from_bytes(&raw_data)
}

/// Parses a replay from raw bytes without applying ranked 3v3 policy checks.
///
/// Use for tooling that needs header or network statistics (for example player-count diagnosis).
///
/// # Errors
///
/// Returns an error if the data cannot be parsed.
pub fn parse_boxcars_replay_without_policy(data: &[u8]) -> Result<Replay, ReplayAcceptanceError> {
    ParserBuilder::new(data)
        .must_parse_network_data()
        .parse()
        .map_err(|error| ReplayAcceptanceError::Parse(anyhow::anyhow!(error)))
}

/// Parses a replay from raw bytes.
///
/// # Errors
///
/// Returns an error if the data cannot be parsed, or the replay is not a supported match.
pub fn parse_replay_from_bytes(data: &[u8]) -> Result<ParsedReplay, ReplayAcceptanceError> {
    let replay = parse_boxcars_replay_without_policy(data)?;
    let match_format = match_validation::validate_supported_match(&replay)
        .map_err(ReplayAcceptanceError::Unsupported)?;
    let mut parsed = parse_boxcars_replay(&replay).map_err(ReplayAcceptanceError::Parse)?;
    parsed.match_format = match_format;
    Ok(parsed)
}

fn parse_boxcars_replay(replay: &Replay) -> anyhow::Result<ParsedReplay> {
    let goals = extract_goals(replay);
    let goal_frames: Vec<usize> = goals.iter().map(|goal| goal.frame).collect();

    // Extract player name -> team mapping from header's PlayerStats
    // This is the authoritative source and helps with reconnecting players
    let header_player_teams = extract_player_teams_from_header(replay);

    // Find object IDs for important types
    let ball_object_id = find_object_id(&replay.objects, "Archetypes.Ball.Ball_Default");
    let car_object_id = find_object_id(&replay.objects, "Archetypes.Car.Car_Default");
    let boost_object_id = find_object_id(
        &replay.objects,
        "Archetypes.CarComponents.CarComponent_Boost",
    );

    // Find attribute IDs for important properties
    let player_name_attr =
        find_object_id(&replay.objects, "Engine.PlayerReplicationInfo:PlayerName");
    let team_attr = find_object_id(&replay.objects, "Engine.PlayerReplicationInfo:Team");
    let seconds_remaining_attr = find_object_id(
        &replay.objects,
        "TAGame.GameEvent_Soccar_TA:SecondsRemaining",
    );
    // Replays recorded before the `ReplicatedBoost` struct existed carry the amount as a plain
    // byte. Without this, every car in those replays reads boost = 0 for the whole match —
    // about three quarters of the bronze-1 training replays (they are old because bronze
    // uploads are rare), which turned "no boost" into a bronze marker.
    let legacy_boost_amount_attr = find_object_id(
        &replay.objects,
        "TAGame.CarComponent_Boost_TA:ReplicatedBoostAmount",
    );
    let input_attributes = InputAttributeIds::find(&replay.objects);
    let component_archetypes = ComponentArchetypeIds::find(&replay.objects);

    let network_frames = replay
        .network_frames
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No network frames in replay"))?;

    // Track active actors
    let mut ball_actors: HashMap<i32, ActorState> = HashMap::new();
    let mut car_actors: HashMap<i32, ActorState> = HashMap::new();
    let mut boost_actors: HashMap<i32, i32> = HashMap::new(); // boost_actor_id -> car_actor_id
    let mut boost_amounts: HashMap<i32, f32> = HashMap::new(); // car_actor_id -> boost (0-1)
    // Car component actor id -> its kind, and -> the car it belongs to.
    let mut component_kinds: HashMap<i32, CarComponentKind> = HashMap::new();
    let mut component_cars: HashMap<i32, i32> = HashMap::new();
    let mut player_infos: HashMap<i32, PlayerInfo> = HashMap::new(); // PRI actor_id -> PlayerInfo
    let mut car_to_actor_id: HashMap<i32, i32> = HashMap::new(); // car_actor_id -> PRI actor_id
    // Forward mapping: which actors each actor references via ActiveActor (can be multiple)
    // Maps: actor_id -> Vec<referenced_actor_id>
    let mut actor_links: HashMap<i32, Vec<i32>> = HashMap::new();
    // Reverse mapping: which actors reference a given actor
    // Maps: referenced_actor_id -> Vec<actors_that_reference_it>
    let mut reverse_actor_links: HashMap<i32, Vec<i32>> = HashMap::new();

    // Cache for player names to reuse Arc<String> across frames
    let mut name_cache: HashMap<String, Arc<String>> = HashMap::new();

    // Persistent cache: player name -> team mapping (survives reconnections)
    // Initialize with authoritative data from replay header
    let mut player_name_to_team: HashMap<String, Team> = header_player_teams;

    // State tracking for frame reconstruction
    let mut current_seconds_remaining: i32 = 300;
    let mut frames = Vec::with_capacity(network_frames.frames.len());
    let mut kickoff_frames = Vec::new();
    let mut last_kickoff_detected = false;

    for (frame_idx, frame) in network_frames.frames.iter().enumerate() {
        let mut frame_demolitions: Vec<DemolitionEvent> = Vec::new();
        // Handle new actors
        for new_actor in &frame.new_actors {
            let actor_id = new_actor.actor_id.0;
            let object_id = new_actor.object_id.0 as usize;

            if Some(object_id) == ball_object_id {
                ball_actors.insert(actor_id, ActorState::default());
            } else if Some(object_id) == car_object_id {
                car_actors.insert(actor_id, ActorState::default());
            } else if Some(object_id) == boost_object_id {
                // Boost component - we'll link it to a car via Vehicle attribute
                boost_actors.insert(actor_id, 0);
            }
            if let Some(kind) = component_archetypes.kind_of(object_id) {
                component_kinds.insert(actor_id, kind);
            }

            // Check if this is a PRI (player replication info) based on object name
            if let Some(obj_name) = replay.objects.get(object_id)
                && obj_name.contains("PRI_TA")
                && !obj_name.contains(':')
            {
                player_infos.insert(actor_id, PlayerInfo::default());
            }
        }

        // Handle deleted actors
        for deleted_actor in &frame.deleted_actors {
            let actor_id = deleted_actor.0;
            ball_actors.remove(&actor_id);
            car_actors.remove(&actor_id);
            boost_actors.remove(&actor_id);
            component_kinds.remove(&actor_id);
            component_cars.remove(&actor_id);
            player_infos.remove(&actor_id);
            car_to_actor_id.remove(&actor_id);
            // Also remove any car_to_actor_id entries that point TO this deleted actor (PRI deletion)
            car_to_actor_id.retain(|_, &mut pri_id| pri_id != actor_id);
            actor_links.remove(&actor_id);
            // Remove any links that point to this actor
            for links in actor_links.values_mut() {
                links.retain(|&x| x != actor_id);
            }
            reverse_actor_links.remove(&actor_id);
            // Remove this actor from reverse links
            for referencers in reverse_actor_links.values_mut() {
                referencers.retain(|&x| x != actor_id);
            }
        }

        // Handle updated actors
        for update in &frame.updated_actors {
            let actor_id = update.actor_id.0;

            match &update.attribute {
                Attribute::RigidBody(rigid_body) => {
                    let state = ActorState {
                        position: rigid_body.location.into(),
                        velocity: rigid_body
                            .linear_velocity
                            .map(Into::into)
                            .unwrap_or_default(),
                        rotation: rigid_body.rotation.into(),
                        angular_velocity: rigid_body
                            .angular_velocity
                            .map(Into::into)
                            .unwrap_or_default(),
                        ..Default::default()
                    };

                    if let std::collections::hash_map::Entry::Occupied(mut entry) =
                        ball_actors.entry(actor_id)
                    {
                        entry.insert(state);
                    } else if car_actors.contains_key(&actor_id)
                        && let Some(existing) = car_actors.get_mut(&actor_id)
                    {
                        existing.position = state.position;
                        existing.velocity = state.velocity;
                        existing.rotation = state.rotation;
                        existing.angular_velocity = state.angular_velocity;
                    }
                }
                Attribute::ReplicatedBoost(boost) => {
                    // This is a boost component actor
                    if let Some(&car_id) = boost_actors.get(&actor_id)
                        && car_id != 0
                    {
                        let boost_normalized = f32::from(boost.boost_amount) / 255.0;
                        boost_amounts.insert(car_id, boost_normalized);
                        if let Some(car) = car_actors.get_mut(&car_id) {
                            car.boost = boost_normalized;
                        }
                    }
                }
                Attribute::String(name) => {
                    // Player name - check if this is the PlayerName attribute
                    // Check object_id, not stream_id (object_id identifies the attribute type)
                    let object_id = update.object_id.0 as usize;
                    if Some(object_id) == player_name_attr && !name.is_empty() {
                        // Create PRI entry if it doesn't exist
                        let info = player_infos.entry(actor_id).or_default();
                        info.name.clone_from(name);

                        // If we already know this player's team from a previous session, restore it
                        if info.team.is_none() {
                            if let Some(&cached_team) = player_name_to_team.get(name) {
                                info.team = Some(cached_team);
                            }
                        } else if let Some(team) = info.team {
                            // Cache the player name -> team mapping
                            player_name_to_team.insert(name.clone(), team);
                        }

                        // Update all cars that might link to this PRI
                        for &car_id in car_actors.keys() {
                            if !car_to_actor_id.contains_key(&car_id) {
                                update_car_to_pri_mapping(
                                    car_id,
                                    &actor_links,
                                    &reverse_actor_links,
                                    &mut car_to_actor_id,
                                    &player_infos,
                                );
                            }
                        }
                    }
                }
                Attribute::TeamPaint(team_paint) => {
                    // Also check if this is a player info
                    if let Some(info) = player_infos.get_mut(&actor_id) {
                        let team = Team::from(team_paint.team);
                        info.team = Some(team);
                        // Cache the player name -> team mapping for reconnection handling
                        if !info.name.is_empty() {
                            player_name_to_team.insert(info.name.clone(), team);
                        }
                    }
                }
                Attribute::FlaggedByte(flagged, team_id) => {
                    // Team assignment via flagged byte (Team attribute)
                    let stream_id = update.stream_id.0 as usize;
                    if Some(stream_id) == team_attr
                        && *flagged
                        && let Some(info) = player_infos.get_mut(&actor_id)
                    {
                        // team_id typically encodes the team (0 or 1)
                        let team = Team::from(*team_id % 2);
                        info.team = Some(team);
                        // Cache the player name -> team mapping for reconnection handling
                        if !info.name.is_empty() {
                            player_name_to_team.insert(info.name.clone(), team);
                        }
                    }
                }
                Attribute::ActiveActor(active_actor) => {
                    let referenced_actor = active_actor.actor.0;

                    // Build forward mapping: track which actor references which other actor
                    // Store all links (an actor can have multiple ActiveActor attributes)

                    let links = actor_links.entry(actor_id).or_default();

                    if !links.contains(&referenced_actor) {
                        links.push(referenced_actor);
                    }

                    // Build reverse mapping: track which actors reference this one
                    let reverse_links = reverse_actor_links.entry(referenced_actor).or_default();

                    if !reverse_links.contains(&actor_id) {
                        reverse_links.push(actor_id);
                    }

                    // This links various components together
                    // For boost components: boost_actor -> car_actor
                    if boost_actors.contains_key(&actor_id) {
                        let car_id = referenced_actor;
                        boost_actors.insert(actor_id, car_id);
                    }
                    if component_kinds.contains_key(&actor_id) {
                        component_cars.insert(actor_id, referenced_actor);
                    }

                    // KEY INSIGHT: A car actor can directly reference its PRI via ActiveActor!
                    // If THIS actor is a car and the referenced actor is a PRI, link them directly
                    // Note: We link even if the PRI doesn't have a name yet - the name may come later
                    if car_actors.contains_key(&actor_id)
                        && player_infos.contains_key(&referenced_actor)
                    {
                        car_to_actor_id.insert(actor_id, referenced_actor);
                    }

                    // If a car is being referenced by this component, check if this component also references a PRI
                    if car_actors.contains_key(&referenced_actor) {
                        // Check all actors that this component references to find a PRI
                        if let Some(component_refs) = actor_links.get(&actor_id) {
                            for &component_ref in component_refs {
                                if component_ref != referenced_actor
                                    && player_infos.contains_key(&component_ref)
                                {
                                    car_to_actor_id.insert(referenced_actor, component_ref);
                                    break;
                                }
                            }
                        }
                        // Also try recursive search
                        let mut visited = std::collections::HashSet::new();
                        if let Some(pri_id) = find_pri_for_car(
                            referenced_actor,
                            &actor_links,
                            &reverse_actor_links,
                            &player_infos,
                            &mut visited,
                        ) {
                            car_to_actor_id.insert(referenced_actor, pri_id);
                        }
                    }

                    // If this component references a PRI, check if any cars reference this component
                    if player_infos.contains_key(&referenced_actor) {
                        // This component references a PRI - check if any cars reference this component
                        if let Some(car_referencers) = reverse_actor_links.get(&actor_id) {
                            for &car_id in car_referencers {
                                if car_actors.contains_key(&car_id) {
                                    car_to_actor_id.insert(car_id, referenced_actor);
                                }
                            }
                        }
                        // Also check if this component references any cars, and link those cars to this PRI
                        if let Some(component_refs) = actor_links.get(&actor_id) {
                            for &component_ref in component_refs {
                                if car_actors.contains_key(&component_ref) {
                                    car_to_actor_id.insert(component_ref, referenced_actor);
                                }
                            }
                        }
                    }

                    // Also update any cars that might now link to a PRI through this new link
                    for &car_id in car_actors.keys() {
                        if !car_to_actor_id.contains_key(&car_id) {
                            update_car_to_pri_mapping(
                                car_id,
                                &actor_links,
                                &reverse_actor_links,
                                &mut car_to_actor_id,
                                &player_infos,
                            );
                        }
                    }
                }
                Attribute::Byte(value) => {
                    let object_id = update.object_id.0 as usize;
                    let amount = value;
                    if Some(object_id) == input_attributes.replicated_active
                        && let Some(&kind) = component_kinds.get(&actor_id)
                        && let Some(&car_id) = component_cars.get(&actor_id)
                        && let Some(car) = car_actors.get_mut(&car_id)
                    {
                        let active = value % 2 == 1;
                        match kind {
                            CarComponentKind::Boost => car.controls.boosting = active,
                            CarComponentKind::Jump => car.controls.jumping = active,
                            CarComponentKind::DoubleJump => car.controls.double_jumping = active,
                            CarComponentKind::Dodge => car.controls.dodging = active,
                        }
                    } else if Some(object_id) == input_attributes.throttle
                        && let Some(car) = car_actors.get_mut(&actor_id)
                    {
                        car.controls.throttle = byte_axis(*value);
                    } else if Some(object_id) == input_attributes.steer
                        && let Some(car) = car_actors.get_mut(&actor_id)
                    {
                        car.controls.steer = byte_axis(*value);
                    } else if Some(object_id) == legacy_boost_amount_attr
                        && let Some(&car_id) = boost_actors.get(&actor_id)
                        && car_id != 0
                    {
                        let boost_normalized = f32::from(*amount) / 255.0;
                        boost_amounts.insert(car_id, boost_normalized);
                        if let Some(car) = car_actors.get_mut(&car_id) {
                            car.boost = boost_normalized;
                        }
                    }
                }
                Attribute::Boolean(value) => {
                    let object_id = update.object_id.0 as usize;
                    if Some(object_id) == input_attributes.handbrake
                        && let Some(car) = car_actors.get_mut(&actor_id)
                    {
                        car.controls.handbrake = *value;
                    }
                }
                Attribute::Location(vector) => {
                    let object_id = update.object_id.0 as usize;
                    if Some(object_id) == input_attributes.dodge_torque
                        && let Some(&car_id) = component_cars.get(&actor_id)
                        && let Some(car) = car_actors.get_mut(&car_id)
                    {
                        car.controls.dodge_torque = (*vector).into();
                    }
                }
                Attribute::Int(seconds) => {
                    let object_id = update.object_id.0 as usize;
                    if Some(object_id) == seconds_remaining_attr {
                        current_seconds_remaining = *seconds;
                    }
                }
                Attribute::Demolish(demo) => {
                    // Mark car as demolished
                    let victim_id = demo.victim.0;
                    frame_demolitions.push(DemolitionEvent {
                        attacker_actor_id: demo.attacker.0,
                        victim_actor_id: victim_id,
                    });
                    if let Some(car) = car_actors.get_mut(&victim_id) {
                        car.is_demolished = true;
                    }
                }
                Attribute::DemolishExtended(demo) => {
                    // Newer replays only send this variant. Before it was handled, demolitions
                    // in those replays were invisible: no demolished flag, no event.
                    let victim_id = demo.victim.actor.0;
                    frame_demolitions.push(DemolitionEvent {
                        attacker_actor_id: demo.attacker.actor.0,
                        victim_actor_id: victim_id,
                    });
                    if let Some(car) = car_actors.get_mut(&victim_id) {
                        car.is_demolished = true;
                    }
                }
                Attribute::DemolishFx(demo) => {
                    // Mark car as demolished (extended version)
                    let victim_id = demo.victim.0;
                    frame_demolitions.push(DemolitionEvent {
                        attacker_actor_id: demo.attacker.0,
                        victim_actor_id: victim_id,
                    });
                    if let Some(car) = car_actors.get_mut(&victim_id) {
                        car.is_demolished = true;
                    }
                }
                Attribute::Reservation(res) => {
                    // Link player to team via reservation
                    if let Some(info) = player_infos.get_mut(&actor_id)
                        && let Some(name) = &res.name
                    {
                        info.name.clone_from(name);
                    }
                }
                _ => {}
            }
        }

        // Detect kickoff (ball at center position with low velocity)
        let ball_at_center = ball_actors.values().any(|b| {
            b.position.x.abs() < 10.0
                && b.position.y.abs() < 10.0
                && b.velocity.x.abs() < 1.0
                && b.velocity.y.abs() < 1.0
        });

        if ball_at_center && !last_kickoff_detected && frame_idx > 0 {
            kickoff_frames.push(frame_idx);
        }
        last_kickoff_detected = ball_at_center;

        // Build frame state
        let ball_state = ball_actors
            .values()
            .next()
            .map(|b| BallState {
                position: b.position,
                velocity: b.velocity,
            })
            .unwrap_or_default();

        let mut players: Vec<PlayerState> = car_actors
            .iter()
            .map(|(&car_id, car)| {
                // Try to find player info for this car
                let pri_id = car_to_actor_id.get(&car_id);
                let player_info = pri_id.and_then(|id| player_infos.get(id));

                let boost = boost_amounts.get(&car_id).copied().unwrap_or(car.boost);

                // Get the player name if available
                let player_name = player_info.filter(|p| !p.name.is_empty()).map(|p| &p.name);

                // Determine team - try player info first, then cached team by name, finally position heuristic
                let team = player_info
                    .and_then(|p| p.team)
                    .or_else(|| {
                        // Try the persistent name -> team cache (from header or previous frames)
                        player_name.and_then(|n| player_name_to_team.get(n).copied())
                    })
                    .unwrap_or_else(|| {
                        // Heuristic: cars on positive Y side at start are team 1
                        Team::from(u8::from(car.position.y > 0.0))
                    });

                let mut actor_state = car.clone();
                actor_state.boost = boost;

                // Get or create Arc<String> for player name, reusing across frames
                let name = if let Some(info) = player_info.filter(|p| !p.name.is_empty()) {
                    name_cache
                        .entry(info.name.clone())
                        .or_insert_with(|| Arc::new(info.name.clone()))
                        .clone()
                } else {
                    let default_name = format!("Player_{car_id}");
                    name_cache
                        .entry(default_name.clone())
                        .or_insert_with(|| Arc::new(default_name))
                        .clone()
                };

                PlayerState {
                    actor_id: car_id,
                    name,
                    team,
                    actor_state,
                }
            })
            .collect();

        // Sort players by team then name for consistent ordering.
        // Name is stable across respawns (a player keeps their name even when a new car
        // spawns after a demolish or goal reset, which changes the actor_id).
        players.sort_by(|a, b| a.team.cmp(&b.team).then(a.name.cmp(&b.name)));

        // Reset demolished state after capturing it
        for car in car_actors.values_mut() {
            car.is_demolished = false;
        }

        frames.push(GameFrame {
            time: frame.time,
            delta: frame.delta,
            seconds_remaining: current_seconds_remaining,
            ball: ball_state,
            players,
            demolitions: frame_demolitions,
        });
    }

    // Add initial kickoff if we have frames
    if !frames.is_empty() && (kickoff_frames.is_empty() || kickoff_frames.first() != Some(&0)) {
        kickoff_frames.insert(0, 0);
    }

    // Final pass: try to link any remaining cars to PRIs
    // Check all components that reference each car to see if they also reference a PRI
    for &car_id in car_actors.keys() {
        if car_to_actor_id.contains_key(&car_id) {
            continue;
        }
        // Find components that reference this car
        if let Some(components) = reverse_actor_links.get(&car_id) {
            for &component_id in components {
                // Check if this component references a PRI
                let Some(component_refs) = actor_links.get(&component_id) else {
                    continue;
                };

                for &component_ref in component_refs {
                    if component_ref != car_id && player_infos.contains_key(&component_ref) {
                        car_to_actor_id.insert(car_id, component_ref);
                        break;
                    }
                }
            }
        }
        // Also try recursive search
        update_car_to_pri_mapping(
            car_id,
            &actor_links,
            &reverse_actor_links,
            &mut car_to_actor_id,
            &player_infos,
        );
    }

    Ok(ParsedReplay {
        frames,
        goals,
        goal_frames,
        kickoff_frames,
        header_player_stats: extract_header_player_stats(replay),
        match_format: MatchFormat::default(),
        outcome: extract_outcome(replay),
    })
}

/// Forfeit flag and winning team from the header. Replays saved after the match carry
/// `WinningTeam`; most others keep the default name ending in "Win" or "Loss" from the
/// recorder's side, whose team is `PrimaryPlayerTeam`.
fn extract_outcome(replay: &Replay) -> MatchOutcome {
    let property = |key: &str| {
        replay
            .properties
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    };
    let integer = |key: &str| match property(key) {
        Some(HeaderProp::Int(value)) => Some(*value),
        _ => None,
    };
    let forfeit = matches!(property("bForfeit"), Some(HeaderProp::Bool(true)));
    let from_name = || {
        let Some(HeaderProp::Str(name)) = property("ReplayName") else {
            return None;
        };
        let recorder = Team::from(integer("PrimaryPlayerTeam")?);
        let other = match recorder {
            Team::Blue => Team::Orange,
            Team::Orange => Team::Blue,
        };
        if name.ends_with(" Win") {
            Some(recorder)
        } else if name.ends_with(" Loss") {
            Some(other)
        } else {
            None
        }
    };
    MatchOutcome {
        forfeit,
        winning_team: integer("WinningTeam").map(Team::from).or_else(from_name),
    }
}

fn extract_goals(replay: &Replay) -> Vec<GoalEvent> {
    let mut goals = Vec::new();

    for (key, value) in &replay.properties {
        if key == "Goals"
            && let HeaderProp::Array(goal_array) = value
        {
            for goal_props in goal_array {
                let mut frame = 0;
                let mut player_name = String::new();
                let mut player_team = 0u8;

                for (prop_key, prop_value) in goal_props {
                    match (prop_key.as_str(), prop_value) {
                        ("frame", HeaderProp::Int(f)) => frame = *f as usize,
                        ("PlayerName", HeaderProp::Str(name)) => player_name.clone_from(name),
                        ("PlayerTeam", HeaderProp::Int(t)) => player_team = *t as u8,
                        _ => {}
                    }
                }

                goals.push(GoalEvent {
                    frame,
                    player_name,
                    player_team: Team::from(player_team),
                });
            }
        }
    }

    goals
}

fn find_object_id(objects: &[String], name: &str) -> Option<usize> {
    objects.iter().position(|o| o == name)
}

/// Reads every player's end-of-match scoreboard line from the header's `PlayerStats`.
fn extract_header_player_stats(replay: &Replay) -> Vec<HeaderPlayerStats> {
    match_validation::player_stats_row_slices(replay)
        .into_iter()
        .map(|player_props| {
            let mut stats = HeaderPlayerStats::default();
            for (prop_key, prop_value) in player_props {
                match (prop_key.as_str(), prop_value) {
                    ("Name", HeaderProp::Str(name)) => stats.name.clone_from(name),
                    ("Team", HeaderProp::Int(team)) => stats.team = Some(Team::from(*team as u8)),
                    ("Score", HeaderProp::Int(value)) => stats.score = *value,
                    ("Goals", HeaderProp::Int(value)) => stats.goals = *value,
                    ("Assists", HeaderProp::Int(value)) => stats.assists = *value,
                    ("Saves", HeaderProp::Int(value)) => stats.saves = *value,
                    ("Shots", HeaderProp::Int(value)) => stats.shots = *value,
                    _ => {}
                }
            }
            stats
        })
        .collect()
}

/// Extract player name -> team mapping from the replay header's `PlayerStats`.
/// This is the authoritative source for team assignments when present.
fn extract_player_teams_from_header(replay: &Replay) -> HashMap<String, Team> {
    let mut player_teams = HashMap::new();

    for player_props in match_validation::player_stats_row_slices(replay) {
        let mut name = String::new();
        let mut team: Option<u8> = None;

        for (prop_key, prop_value) in player_props {
            match (prop_key.as_str(), prop_value) {
                ("Name", HeaderProp::Str(n)) => name.clone_from(n),
                ("Team", HeaderProp::Int(t)) => team = Some(*t as u8),
                _ => {}
            }
        }

        if !name.is_empty()
            && let Some(t) = team
        {
            player_teams.insert(name, Team::from(t));
        }
    }

    player_teams
}

#[cfg(test)]
mod tests {
    use tracing::info;

    use super::*;

    #[test]
    fn test_parse_real_replay() {
        let _tracing = tracing_subscriber::fmt()
            .with_test_writer()
            .with_max_level(tracing::Level::INFO)
            .try_init();

        let replay_path = Path::new("../../test_data/2af51380-05b5-44ac-8b31-94b8b0f8da84.replay");
        assert!(replay_path.exists(), "Test replay not found");

        let parsed = parse_replay(replay_path).expect("Failed to parse replay");
        info!(parsed_frames = parsed.frames.len(), "Parsed frames");

        assert!(parsed.frames.len() > 1000, "Expected many frames");

        // Check goals
        info!("=== GOALS ===");
        for goal in &parsed.goals {
            info!(
                frame = goal.frame,
                player = %goal.player_name,
                team = ?goal.player_team,
                "Goal scored"
            );
        }
        assert!(!parsed.goals.is_empty(), "Expected goals in replay");

        // Check kickoffs
        info!("=== KICKOFFS ===");
        info!(
            kickoff_frames = ?&parsed.kickoff_frames[..parsed.kickoff_frames.len().min(10)],
            "Kickoff frames"
        );

        // Check a sample frame in the middle
        let mid_idx = parsed.frames.len() / 2;
        if let Some(frame) = parsed.frames.get(mid_idx) {
            info!("=== SAMPLE FRAME {} ===", mid_idx);
            info!(time = frame.time, "Time (seconds)");
            info!(
                seconds_remaining = frame.seconds_remaining,
                "Seconds remaining"
            );
            info!(
                ball_x = frame.ball.position.x,
                ball_y = frame.ball.position.y,
                ball_z = frame.ball.position.z,
                "Ball position"
            );
            info!(player_count = frame.players.len(), "Players");
            for player in &frame.players {
                info!(
                    name = %player.name,
                    team = ?player.team,
                    pos_x = player.actor_state.position.x,
                    pos_y = player.actor_state.position.y,
                    pos_z = player.actor_state.position.z,
                    velocity_x = player.actor_state.velocity.x,
                    velocity_y = player.actor_state.velocity.y,
                    velocity_z = player.actor_state.velocity.z,
                    boost = player.actor_state.boost * 100.0,
                    "Player state"
                );
            }
        }

        // Check player names are properly parsed in player_name field
        info!("=== PLAYER NAMES ===");

        // Expected players: Blue (team 0) and Orange (team 1)
        let expected_blue_players = ["Dtwlve1", "Rip.the.Trip", "Ah perro!"];
        let expected_orange_players = ["************", "Mooski17", "Olin"];

        // Check frames to find player names in PlayerState.name
        // Also verify each frame that player names match expected teams
        for frame in &parsed.frames {
            assert!(
                frame.players.len() >= 5,
                "There should be at least 5 players in each frame in this replay"
            );
            for player in &frame.players {
                // Verify player name is in the expected list for their team
                match player.team {
                    Team::Blue => {
                        assert!(
                            expected_blue_players.contains(&player.name.as_str()),
                            "Player '{}' is on Blue team but not in expected_blue_players. Time: {}. Remaining seconds: {}",
                            player.name,
                            frame.time,
                            frame.seconds_remaining
                        );
                    }
                    Team::Orange => {
                        assert!(
                            expected_orange_players.contains(&player.name.as_str()),
                            "Player '{}' is on Orange team but not in expected_orange_players. Time: {}. Remaining seconds: {}",
                            player.name,
                            frame.time,
                            frame.seconds_remaining
                        );
                    }
                }
            }
        }
    }

    /// The recorder (orange) forfeited a 1-1 duel: their replay is named "... Loss".
    #[test]
    fn forfeit_and_winner_come_from_the_header() {
        let path =
            Path::new("../../test_data/ranked-duels-a9feb907-75e9-4514-b52f-44766e0374f1.replay");
        let parsed = parse_replay_from_bytes(&std::fs::read(path).expect("test replay exists"))
            .expect("test replay parses");
        assert_eq!(
            parsed.outcome,
            MatchOutcome {
                forfeit: true,
                winning_team: Some(Team::Blue),
            }
        );
    }

    #[test]
    fn dump_smurf_replay_timing() {
        let replay_path = Path::new("../../data/smurf/ab793727-e03c-4fc7-aab9-5de6ed5c1c19.replay");
        if !replay_path.exists() {
            eprintln!("smurf replay not found, skipping");
            return;
        }
        let parsed = parse_replay(replay_path).expect("Failed to parse smurf replay");
        let total = parsed.frames.len();
        assert!(total > 1000);
        let last = parsed.frames.last().unwrap();
        assert!(
            last.seconds_remaining < 300,
            "seconds_remaining should decrease during a match"
        );
    }
}

/// Maps a replicated input byte (128 = centred) onto `[-1, 1]`.
fn byte_axis(value: u8) -> f32 {
    ((f32::from(value) - 128.0) / 127.0).clamp(-1.0, 1.0)
}

/// Car components whose `ReplicatedActive` byte is tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CarComponentKind {
    Boost,
    Jump,
    DoubleJump,
    Dodge,
}

/// Object ids of the archetypes that spawn each tracked car component.
struct ComponentArchetypeIds {
    boost: Option<usize>,
    jump: Option<usize>,
    double_jump: Option<usize>,
    dodge: Option<usize>,
}

impl ComponentArchetypeIds {
    fn find(objects: &[String]) -> Self {
        Self {
            boost: find_object_id(objects, "Archetypes.CarComponents.CarComponent_Boost"),
            jump: find_object_id(objects, "Archetypes.CarComponents.CarComponent_Jump"),
            double_jump: find_object_id(
                objects,
                "Archetypes.CarComponents.CarComponent_DoubleJump",
            ),
            dodge: find_object_id(objects, "Archetypes.CarComponents.CarComponent_Dodge"),
        }
    }

    fn kind_of(&self, object_id: usize) -> Option<CarComponentKind> {
        let id = Some(object_id);
        if id == self.boost {
            Some(CarComponentKind::Boost)
        } else if id == self.jump {
            Some(CarComponentKind::Jump)
        } else if id == self.double_jump {
            Some(CarComponentKind::DoubleJump)
        } else if id == self.dodge {
            Some(CarComponentKind::Dodge)
        } else {
            None
        }
    }
}

/// Object ids of the input-related attributes.
struct InputAttributeIds {
    throttle: Option<usize>,
    steer: Option<usize>,
    handbrake: Option<usize>,
    replicated_active: Option<usize>,
    dodge_torque: Option<usize>,
}

impl InputAttributeIds {
    fn find(objects: &[String]) -> Self {
        Self {
            throttle: find_object_id(objects, "TAGame.Vehicle_TA:ReplicatedThrottle"),
            steer: find_object_id(objects, "TAGame.Vehicle_TA:ReplicatedSteer"),
            handbrake: find_object_id(objects, "TAGame.Vehicle_TA:bReplicatedHandbrake"),
            replicated_active: find_object_id(objects, "TAGame.CarComponent_TA:ReplicatedActive"),
            dodge_torque: find_object_id(objects, "TAGame.CarComponent_Dodge_TA:DodgeTorque"),
        }
    }
}
