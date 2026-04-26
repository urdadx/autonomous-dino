import pygame


class ManualAgent:
    def act(self) -> bool:
        keys = pygame.key.get_pressed()
        return bool(keys[pygame.K_SPACE] or keys[pygame.K_UP])
