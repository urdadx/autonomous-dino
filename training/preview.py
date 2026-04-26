from __future__ import annotations

from dataclasses import dataclass

import pygame

from agents.neat_agent import NeatAgent
from dino.config import GameConfig
from dino.game import RunnerGame
from dino.render import GameRenderer


@dataclass(frozen=True)
class PreviewResult:
    score: int
    survival_time: float


class TrainingPreviewWindow:
    def __init__(self, config: GameConfig) -> None:
        pygame.init()
        self.config = config
        self.screen = pygame.display.set_mode((config.screen_width, config.screen_height))
        pygame.display.set_caption("Dino Raw - NEAT Training")
        self.clock = pygame.time.Clock()
        self.renderer = GameRenderer(self.screen, config)

    def close(self) -> None:
        pygame.quit()

    def preview_generation(
        self,
        agent: NeatAgent,
        generation: int,
        preview_seed: int,
        max_seconds: float | None,
        best_fitness: float,
        best_score: int,
        target_score: int,
    ) -> PreviewResult:
        game = RunnerGame(config=self.config, seed=preview_seed)

        while not game.state.game_over and (
            max_seconds is None or game.state.survival_time < max_seconds
        ):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            jump_held = agent.act(game.get_observation())
            game.step(jump_held)

            self.renderer.draw(
                game.state,
                agent_label="NEAT Training",
                status_lines=[
                    f"Generation: {generation}",
                    f"Preview score: {game.state.score}",
                    f"Best score: {best_score}",
                    f"Best fitness: {best_fitness:.2f}",
                    f"Target score: {target_score}",
                ],
            )
            pygame.display.flip()
            self.clock.tick(self.config.target_fps)

        return PreviewResult(score=game.state.score, survival_time=game.state.survival_time)
