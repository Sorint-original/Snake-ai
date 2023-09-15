import pygame
import os

import map_class


tile_pixel = 50
map_border = 5

def game (WIN,WIDTH,HEIGHT,FPS,SCENARIO) :
    run = True

    #DECLARATION OF TE MAP
    
    Map = map_class.map_class(15)
    map_segment_size = map_border*2 + tile_pixel*Map.size + Map.size +1

    WIN.fill((50,50,50))
    pygame.draw.rect(WIN,(50,200,50),pygame.Rect(WIDTH/2-map_segment_size/2,HEIGHT/2-map_segment_size/2,map_segment_size,map_segment_size))
    pygame.draw.rect(WIN,(0,0,0),pygame.Rect(WIDTH/2-map_segment_size/2 + map_border,HEIGHT/2-map_segment_size/2+map_border,map_segment_size-2*map_border,map_segment_size-2*map_border))
    def draw_window(WIN) :

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

