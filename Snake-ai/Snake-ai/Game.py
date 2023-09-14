import pygame
import os


def game (WIN,WIDTH,HEIGHT,FPS,SCENARIO) :
    run = True

    def draw_window(WIN) :
        WIN.fill((50,50,50))
        pygame.display.update()

    while run :

        draw_window(WIN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                pygame.quit()
                os._exit(0)
            if event.type == pygame.KEYDOWN :
                if event.key == pygame.K_ESCAPE :
                    run = False

