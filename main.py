import pygame

from agents.manual import ManualAgent
from dino.config import GameConfig
from dino.game import RunnerGame
from dino.render import GameRenderer


def main() -> None:
    pygame.init()

    config = GameConfig()
    screen = pygame.display.set_mode((config.screen_width, config.screen_height))
    pygame.display.set_caption("Dino Raw")
    clock = pygame.time.Clock()
    renderer = GameRenderer(screen, config)
    game = RunnerGame(config=config)
    agent = ManualAgent()
    running = True

    while running:
        restart_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_r, pygame.K_SPACE):
                restart_requested = True

        if game.state.game_over and restart_requested:
            game.reset()

        if not game.state.game_over:
            game.step(agent.act())

        renderer.draw(game.state, agent_label="Manual")
        pygame.display.flip()
        clock.tick(config.target_fps)

    pygame.quit()


if __name__ == "__main__":
    main()
