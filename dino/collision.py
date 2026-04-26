from __future__ import annotations

from dino.config import GameConfig
from dino.obstacles import Obstacle


def circle_hits_obstacle(
    center_x: float,
    center_y: float,
    radius: float,
    obstacle: Obstacle,
    config: GameConfig,
) -> bool:
    obstacle_left = obstacle.x
    obstacle_right = obstacle.x + obstacle.width
    obstacle_top = config.ground_top - obstacle.height
    obstacle_bottom = config.ground_top

    # Clamp the circle center to the obstacle bounds to find the nearest point.
    closest_x = max(obstacle_left, min(center_x, obstacle_right))
    closest_y = max(obstacle_top, min(center_y, obstacle_bottom))
    dx = center_x - closest_x
    dy = center_y - closest_y
    return dx * dx + dy * dy < radius * radius
