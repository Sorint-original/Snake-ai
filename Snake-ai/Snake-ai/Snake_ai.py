import pygame 
import Menu

pygame.init()

screen = pygame.display.Info()
WIDTH = screen.current_w
HEIGHT = screen.current_h
FPS = 60
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
#pygame_icon = pygame.image.load('Assets/Units/Marine.png')
#pygame.display.set_icon(pygame_icon)

#it enters in the menu
Menu.menu(WIN,WIDTH,HEIGHT,FPS)
