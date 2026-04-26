from __future__ import annotations

import random
from dataclasses import dataclass

from dino.config import GameConfig


@dataclass
class Obstacle:
    x: float
    width: int
    height: int
    passed: bool = False

    @property
    def right(self) -> float:
        return self.x + self.width


def spawn_obstacle(rng: random.Random, config: GameConfig) -> Obstacle:
    width = rng.randint(*config.obstacle_width_range)
    height = rng.randint(*config.obstacle_height_range)
    x = config.screen_width + rng.randint(*config.obstacle_spawn_padding_range)
    return Obstacle(x=x, width=width, height=height)
