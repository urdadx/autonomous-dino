from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import neat
import pygame

from agents.neat_agent import NeatAgent
from dino.config import GameConfig
from dino.game import RunnerGame
from dino.render import GameRenderer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "neat.ini"
DEFAULT_MODEL_PATH = ROOT / "models" / "neat_winner.pkl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch a trained NEAT agent play.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reset-delay", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    game_config = GameConfig()

    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(args.config),
    )

    with args.model.open("rb") as model_file:
        genome = pickle.load(model_file)

    agent = NeatAgent.from_genome(genome, neat_config)

    pygame.init()
    screen = pygame.display.set_mode((game_config.screen_width, game_config.screen_height))
    pygame.display.set_caption("Dino Raw - NEAT Eval")
    clock = pygame.time.Clock()
    renderer = GameRenderer(screen, game_config)
    game = RunnerGame(config=game_config)
    game_over_time = 0.0
    running = True

    while running:
        dt = clock.tick(game_config.target_fps) / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game.state.game_over:
            game_over_time += dt
            if game_over_time >= args.reset_delay:
                game.reset()
                game_over_time = 0.0
        else:
            jump_held = agent.act(game.get_observation())
            game.step(jump_held)

        renderer.draw(game.state, agent_label="NEAT")
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
