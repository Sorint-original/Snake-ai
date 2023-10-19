import pygame 
import Menu
from torch import nn, tensor
import time
import numpy as np
import torch
from Networks import SNAKE_Q_NET

pygame.init()

screen = pygame.display.Info()
WIDTH = screen.current_w
HEIGHT = screen.current_h
FPS = 60
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
#pygame_icon = pygame.image.load('Assets/Units/Marine.png')
#pygame.display.set_icon(pygame_icon)

'''
model = SNAKE_Q_NET(15)

tile_map =[]
aux_matrix1 = []
aux_matrix2 = []
#setup map tiles
for i in range(15):
    aux1 = []
    aux2 = []
    for j in range(15):
        aux1.append(0.0)
        aux2.append(0.0)
    aux_matrix1.append(aux1)
    aux_matrix2.append(aux2)
tile_map = [aux_matrix1,aux_matrix2]
test=model(torch.tensor(tile_map),torch.tensor([0.5]))
'''


#it enters in the menu
Menu.menu(WIN,WIDTH,HEIGHT,FPS)
