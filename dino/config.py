from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    screen_width: int = 1280
    screen_height: int = 720
    target_fps: int = 60
    fixed_timestep: float = 1 / 60

    ball_radius: int = 32
    player_x: int = 180
    ground_height: int = 110
    ground_dash_width: int = 40
    ground_dash_gap: int = 26

    gravity: float = 2200
    fall_gravity: float = 3200
    jump_cut_gravity: float = 4200
    jump_speed: float = 900
    max_fall_speed: float = 1400
    coyote_time: float = 0.1
    jump_buffer_time: float = 0.1
    # Keep the runner honest: one grounded jump only.
    max_jumps: int = 1

    start_world_speed: float = 520
    max_world_speed: float = 920
    speed_ramp: float = 14
    spawn_delay_range: tuple[float, float] = (0.7, 1.15)
    obstacle_width_range: tuple[int, int] = (30, 55)
    obstacle_height_range: tuple[int, int] = (60, 125)
    obstacle_spawn_padding_range: tuple[int, int] = (0, 90)

    @property
    def ground_top(self) -> int:
        return self.screen_height - self.ground_height

    @property
    def floor_y(self) -> int:
        return self.ground_top - self.ball_radius

    @property
    def max_obstacle_width(self) -> int:
        return self.obstacle_width_range[1]

    @property
    def max_obstacle_height(self) -> int:
        return self.obstacle_height_range[1]

    @property
    def max_vertical_speed(self) -> float:
        return max(self.jump_speed, self.max_fall_speed)
