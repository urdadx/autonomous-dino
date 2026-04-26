from __future__ import annotations

import random
from dataclasses import dataclass, field

from dino.collision import circle_hits_obstacle
from dino.config import GameConfig
from dino.obstacles import Obstacle, spawn_obstacle


@dataclass
class PlayerState:
    y: float
    velocity_y: float = 0.0
    is_on_ground: bool = True
    jumps_remaining: int = 0
    coyote_timer: float = 0.0
    jump_buffer_timer: float = 0.0
    jump_input_held: bool = False


@dataclass
class RunnerState:
    player: PlayerState
    world_speed: float
    spawn_timer: float
    ground_offset: float = 0.0
    obstacles: list[Obstacle] = field(default_factory=list)
    score: int = 0
    survival_time: float = 0.0
    game_over: bool = False


@dataclass(frozen=True)
class StepResult:
    passed_obstacles: int = 0
    collision: bool = False
    game_over: bool = False


class RunnerGame:
    observation_size = 9
    action_count = 2

    def __init__(self, config: GameConfig | None = None, seed: int | None = None) -> None:
        self.config = config or GameConfig()
        self.rng = random.Random(seed)
        self.state = self._new_state()

    def reset(self, seed: int | None = None) -> tuple[float, ...]:
        if seed is not None:
            self.rng.seed(seed)
        self.state = self._new_state()
        return self.get_observation()

    def step(self, jump_held: bool, dt: float | None = None) -> StepResult:
        if self.state.game_over:
            return StepResult(collision=True, game_over=True)

        step_dt = self.config.fixed_timestep if dt is None else dt
        player = self.state.player
        jump_pressed = jump_held and not player.jump_input_held
        player.jump_input_held = jump_held

        if jump_pressed:
            player.jump_buffer_timer = self.config.jump_buffer_time
        else:
            player.jump_buffer_timer = max(0.0, player.jump_buffer_timer - step_dt)

        if player.is_on_ground:
            player.coyote_timer = self.config.coyote_time
            player.jumps_remaining = self.config.max_jumps
        else:
            player.coyote_timer = max(0.0, player.coyote_timer - step_dt)
            if player.coyote_timer == 0 and player.jumps_remaining == self.config.max_jumps:
                player.jumps_remaining = self.config.max_jumps - 1

        can_ground_jump = player.is_on_ground or player.coyote_timer > 0
        if player.jump_buffer_timer > 0:
            if can_ground_jump and player.jumps_remaining == self.config.max_jumps:
                player.velocity_y = -self.config.jump_speed
                player.is_on_ground = False
                player.coyote_timer = 0.0
                player.jump_buffer_timer = 0.0
                player.jumps_remaining -= 1
            elif not can_ground_jump and player.jumps_remaining > 0:
                player.velocity_y = -self.config.jump_speed
                player.jump_buffer_timer = 0.0
                player.jumps_remaining -= 1

        current_gravity = self.config.gravity
        if player.velocity_y < 0 and not jump_held:
            current_gravity = self.config.jump_cut_gravity
        elif player.velocity_y > 0:
            current_gravity = self.config.fall_gravity

        player.velocity_y = min(
            player.velocity_y + current_gravity * step_dt,
            self.config.max_fall_speed,
        )
        player.y += player.velocity_y * step_dt

        if player.y >= self.config.floor_y:
            player.y = self.config.floor_y
            player.velocity_y = 0.0
            player.is_on_ground = True
        else:
            player.is_on_ground = False

        self.state.world_speed = min(
            self.config.max_world_speed,
            self.state.world_speed + self.config.speed_ramp * step_dt,
        )
        dash_cycle = self.config.ground_dash_width + self.config.ground_dash_gap
        self.state.ground_offset = (self.state.ground_offset + self.state.world_speed * step_dt) % dash_cycle

        self.state.spawn_timer -= step_dt
        if self.state.spawn_timer <= 0:
            self.state.obstacles.append(spawn_obstacle(self.rng, self.config))
            self.state.spawn_timer = self._sample_spawn_delay()

        passed_obstacles = 0
        for obstacle in self.state.obstacles:
            obstacle.x -= self.state.world_speed * step_dt

            if not obstacle.passed and obstacle.right < self.config.player_x - self.config.ball_radius:
                obstacle.passed = True
                passed_obstacles += 1
                self.state.score += 1

            if circle_hits_obstacle(
                self.config.player_x,
                player.y,
                self.config.ball_radius,
                obstacle,
                self.config,
            ):
                self.state.game_over = True

        self.state.obstacles = [obstacle for obstacle in self.state.obstacles if obstacle.right > -60]
        self.state.survival_time += step_dt

        return StepResult(
            passed_obstacles=passed_obstacles,
            collision=self.state.game_over,
            game_over=self.state.game_over,
        )

    def get_observation(self) -> tuple[float, ...]:
        obstacles = sorted(
            (obstacle for obstacle in self.state.obstacles if obstacle.right >= self.config.player_x),
            key=lambda obstacle: obstacle.x,
        )
        next_obstacle = obstacles[0] if obstacles else None
        second_obstacle = obstacles[1] if len(obstacles) > 1 else None

        # Keep the observation compact and normalized so either NEAT or a future
        # DQN can learn from the same interface.
        return (
            self.state.player.y / self.config.floor_y,
            self._normalize_velocity(self.state.player.velocity_y),
            1.0 if self.state.player.is_on_ground else 0.0,
            self.state.player.jumps_remaining / self.config.max_jumps,
            self._normalize_distance(self._distance_to_player(next_obstacle)),
            self._normalize_width(next_obstacle.width if next_obstacle else 0),
            self._normalize_height(next_obstacle.height if next_obstacle else 0),
            self._normalize_distance(self._distance_to_player(second_obstacle)),
            self.state.world_speed / self.config.max_world_speed,
        )

    def _distance_to_player(self, obstacle: Obstacle | None) -> float:
        if obstacle is None:
            return self.config.screen_width * 1.5
        return obstacle.x - (self.config.player_x + self.config.ball_radius)

    def _normalize_distance(self, distance: float) -> float:
        return distance / self.config.screen_width

    def _normalize_width(self, width: int) -> float:
        return width / self.config.max_obstacle_width

    def _normalize_height(self, height: int) -> float:
        return height / self.config.max_obstacle_height

    def _normalize_velocity(self, velocity: float) -> float:
        return velocity / self.config.max_vertical_speed

    def _new_state(self) -> RunnerState:
        return RunnerState(
            player=PlayerState(
                y=float(self.config.floor_y),
                jumps_remaining=self.config.max_jumps,
            ),
            world_speed=self.config.start_world_speed,
            spawn_timer=self._sample_spawn_delay(),
        )

    def _sample_spawn_delay(self) -> float:
        return self.rng.uniform(*self.config.spawn_delay_range)
