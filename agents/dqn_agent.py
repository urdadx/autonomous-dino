from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from dino.actions import ACTION_JUMP

QValueFn = Callable[[tuple[float, ...]], Sequence[float]]


@dataclass
class GreedyDQNAgent:
    # The concrete network stays outside this class so future PyTorch or JAX
    # code can plug into the same game interface.
    predict_q_values: QValueFn

    def act(self, observation: tuple[float, ...]) -> bool:
        q_values = self.predict_q_values(observation)
        best_action = max(range(len(q_values)), key=q_values.__getitem__)
        return best_action == ACTION_JUMP
