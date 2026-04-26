from __future__ import annotations

from dino.actions import ACTION_COUNT, ACTION_JUMP
from dino.config import GameConfig
from dino.game import RunnerGame


class RunnerEnv:
    def __init__(self, config: GameConfig | None = None, seed: int | None = None) -> None:
        self.game = RunnerGame(config=config, seed=seed)

    @property
    def action_count(self) -> int:
        return ACTION_COUNT

    @property
    def observation_size(self) -> int:
        return self.game.observation_size

    def reset(self, seed: int | None = None) -> tuple[float, ...]:
        return self.game.reset(seed=seed)

    def step(self, action: int) -> tuple[tuple[float, ...], float, bool, dict[str, float | int]]:
        jump_held = action == ACTION_JUMP
        result = self.game.step(jump_held=jump_held)
        reward = 0.1 + result.passed_obstacles * 5.0
        if result.collision:
            reward -= 10.0

        return (
            self.game.get_observation(),
            reward,
            result.game_over,
            {
                "score": self.game.state.score,
                "survival_time": self.game.state.survival_time,
            },
        )
