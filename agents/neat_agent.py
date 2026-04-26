from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import neat


@dataclass
class NeatAgent:
    network: Any
    threshold: float = 0.5

    @classmethod
    def from_genome(cls, genome: Any, config: neat.Config) -> "NeatAgent":
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        return cls(network=network)

    def act(self, observation: tuple[float, ...]) -> bool:
        output = self.network.activate(observation)
        return output[0] > self.threshold
