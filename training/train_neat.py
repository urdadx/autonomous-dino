from __future__ import annotations

import argparse
import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import neat

from agents.neat_agent import NeatAgent
from dino.config import GameConfig
from dino.game import RunnerGame
from training.preview import TrainingPreviewWindow


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "neat.ini"
DEFAULT_MODEL_PATH = ROOT / "models" / "neat_winner.pkl"
DEFAULT_CHECKPOINT_PREFIX = ROOT / "models" / "neat-checkpoint-"


@dataclass(frozen=True)
class GenomeSummary:
    fitness: float
    average_score: float
    best_score: int
    average_survival_time: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a NEAT agent for Dino Raw.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--checkpoint-prefix", type=Path, default=DEFAULT_CHECKPOINT_PREFIX)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--validation-episodes", type=int, default=3)
    parser.add_argument("--validation-seed-base", type=int, default=100_000)
    parser.add_argument("--target-score", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--preview-seed", type=int, default=200_000)
    parser.add_argument("--preview-max-seconds", type=float)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def run_episode(
    agent: NeatAgent,
    game_config: GameConfig,
    seed: int,
    max_seconds: float | None,
) -> tuple[int, float, bool]:
    game = RunnerGame(config=game_config, seed=seed)

    while _episode_is_running(game, max_seconds):
        jump_held = agent.act(game.get_observation())
        game.step(jump_held)

    return game.state.score, game.state.survival_time, game.state.game_over


def summarize_genome(
    genome: Any,
    neat_config: neat.Config,
    game_config: GameConfig,
    episodes: int,
    max_seconds: float | None,
    seed_base: int,
) -> GenomeSummary:
    agent = NeatAgent.from_genome(genome, neat_config)
    total_fitness = 0.0
    total_score = 0
    total_survival_time = 0.0
    best_score = 0

    for episode_index in range(episodes):
        score, survival_time, collision = run_episode(
            agent,
            game_config,
            seed=seed_base + episode_index,
            max_seconds=max_seconds,
        )
        total_fitness += survival_time + score * 8.0
        total_score += score
        total_survival_time += survival_time
        best_score = max(best_score, score)

        if collision:
            total_fitness -= 5.0

    return GenomeSummary(
        fitness=total_fitness / episodes,
        average_score=total_score / episodes,
        best_score=best_score,
        average_survival_time=total_survival_time / episodes,
    )


def save_winner(output_path: Path, winner: Any) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as model_file:
        pickle.dump(winner, model_file)

    return output_path


def _episode_is_running(game: RunnerGame, max_seconds: float | None) -> bool:
    if game.state.game_over:
        return False
    if max_seconds is None:
        return True
    return game.state.survival_time < max_seconds


def run_training(args: argparse.Namespace) -> Path:
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(args.config),
    )

    if args.resume_checkpoint is not None:
        population = neat.Checkpointer.restore_checkpoint(str(args.resume_checkpoint))
        neat_config = population.config
    else:
        population = neat.Population(neat_config)

    # Training ends on our explicit score target, not NEAT's built-in fitness threshold.
    neat_config.no_fitness_termination = True
    args.checkpoint_prefix.parent.mkdir(parents=True, exist_ok=True)

    stats_reporter = neat.StatisticsReporter()
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(stats_reporter)
    population.add_reporter(
        neat.Checkpointer(
            args.checkpoint_every,
            filename_prefix=str(args.checkpoint_prefix),
        )
    )

    game_config = GameConfig()
    preview = TrainingPreviewWindow(game_config) if args.render else None
    best_winner: Any | None = None
    best_summary: GenomeSummary | None = None

    try:
        for _ in range(args.generations):
            generation_seed_base = args.seed_base + population.generation * 1_000

            def evaluate_genomes(genomes: list[tuple[int, Any]], current_config: neat.Config) -> None:
                for _, genome in genomes:
                    summary = summarize_genome(
                        genome,
                        current_config,
                        game_config,
                        episodes=args.episodes,
                        max_seconds=args.max_seconds,
                        seed_base=generation_seed_base,
                    )
                    genome.fitness = summary.fitness

            population.run(evaluate_genomes, 1)
            generation_index = population.generation - 1
            generation_best = copy.deepcopy(stats_reporter.most_fit_genomes[-1])
            generation_summary = summarize_genome(
                generation_best,
                neat_config,
                game_config,
                episodes=args.validation_episodes,
                max_seconds=args.max_seconds,
                seed_base=args.validation_seed_base,
            )

            improved = (
                best_summary is None
                or generation_summary.best_score > best_summary.best_score
                or (
                    generation_summary.best_score == best_summary.best_score
                    and generation_summary.fitness > best_summary.fitness
                )
            )
            if improved:
                best_winner = generation_best
                best_summary = generation_summary
                save_winner(args.output, best_winner)

            print(
                "Validation: "
                f"generation={generation_index} "
                f"avg_score={generation_summary.average_score:.2f} "
                f"best_score={generation_summary.best_score} "
                f"avg_survival={generation_summary.average_survival_time:.2f}s"
            )

            if preview is not None:
                preview_agent = NeatAgent.from_genome(generation_best, neat_config)
                preview_result = preview.preview_generation(
                    preview_agent,
                    generation=generation_index,
                    preview_seed=args.preview_seed,
                    max_seconds=args.preview_max_seconds,
                    best_fitness=best_summary.fitness if best_summary else generation_summary.fitness,
                    best_score=best_summary.best_score if best_summary else generation_summary.best_score,
                    target_score=args.target_score,
                )
                print(
                    "Preview: "
                    f"generation={generation_index} "
                    f"score={preview_result.score} "
                    f"survival={preview_result.survival_time:.2f}s"
                )

            if best_summary is not None and best_summary.best_score >= args.target_score:
                print(f"Reached target score {args.target_score} at generation {generation_index}.")
                break
        else:
            print(f"Stopped after {args.generations} generations without hitting score {args.target_score}.")
    except KeyboardInterrupt:
        print("Training interrupted. Saving best genome seen so far.")
    finally:
        if preview is not None:
            preview.close()

    if best_winner is None:
        raise RuntimeError("Training did not produce a winner.")

    return save_winner(args.output, best_winner)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path = run_training(args)
    print(f"Saved NEAT winner to {output_path}")


if __name__ == "__main__":
    main()
