from __future__ import annotations

import pygame

from dino.config import GameConfig
from dino.game import RunnerState
from dino.obstacles import Obstacle


class GameRenderer:
    def __init__(self, screen: pygame.Surface, config: GameConfig) -> None:
        self.screen = screen
        self.config = config
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)

    def draw(
        self,
        state: RunnerState,
        agent_label: str | None = None,
        status_lines: list[str] | None = None,
    ) -> None:
        self.screen.fill((247, 247, 247))
        pygame.draw.rect(
            self.screen,
            (232, 232, 232),
            (0, self.config.ground_top, self.config.screen_width, self.config.ground_height),
        )
        pygame.draw.line(
            self.screen,
            (60, 60, 60),
            (0, self.config.ground_top),
            (self.config.screen_width, self.config.ground_top),
            3,
        )

        self._draw_ground_dashes(state.ground_offset)

        for obstacle in state.obstacles:
            pygame.draw.rect(
                self.screen,
                (30, 140, 70),
                self._obstacle_rect(obstacle),
                border_radius=6,
            )

        pygame.draw.circle(
            self.screen,
            (220, 60, 60),
            (self.config.player_x, round(state.player.y)),
            self.config.ball_radius,
        )

        score_text = self.font.render(f"Score: {state.score}", True, (40, 40, 40))
        speed_text = self.font.render(f"Speed: {int(state.world_speed)}", True, (40, 40, 40))
        self.screen.blit(score_text, (24, 20))
        self.screen.blit(speed_text, (24, 54))

        if agent_label:
            agent_text = self.font.render(f"Agent: {agent_label}", True, (40, 40, 40))
            self.screen.blit(agent_text, (24, 88))

        if status_lines:
            # Keep training/debug overlays separate from the core game HUD so the
            # same renderer can be reused for manual play, eval, and live training.
            for index, line in enumerate(status_lines):
                line_text = self.font.render(line, True, (40, 40, 40))
                line_rect = line_text.get_rect(topright=(self.config.screen_width - 24, 20 + index * 32))
                self.screen.blit(line_text, line_rect)

        if state.game_over:
            game_over_text = self.big_font.render("Game Over", True, (40, 40, 40))
            restart_text = self.font.render("Press Space or R to restart", True, (40, 40, 40))
            game_over_rect = game_over_text.get_rect(center=(self.config.screen_width / 2, 220))
            restart_rect = restart_text.get_rect(center=(self.config.screen_width / 2, 270))
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(restart_text, restart_rect)

    def _draw_ground_dashes(self, ground_offset: float) -> None:
        dash_x = -self.config.ground_dash_width
        dash_cycle = self.config.ground_dash_width + self.config.ground_dash_gap

        while dash_x < self.config.screen_width + self.config.ground_dash_width:
            pygame.draw.rect(
                self.screen,
                (120, 120, 120),
                (
                    dash_x - ground_offset,
                    self.config.ground_top + 32,
                    self.config.ground_dash_width,
                    8,
                ),
            )
            dash_x += dash_cycle

    def _obstacle_rect(self, obstacle: Obstacle) -> pygame.Rect:
        return pygame.Rect(
            round(obstacle.x),
            self.config.ground_top - obstacle.height,
            obstacle.width,
            obstacle.height,
        )
