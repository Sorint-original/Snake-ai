import pygame
import os

import map_classes
from map_classes import tile_pixel, map_border,deafoult_size


def game (WIN,WIDTH,HEIGHT,FPS,SCENARIO) :
    run = True

    #DECLARATION OF TE MAP
    map_segment_size = map_border*2 + tile_pixel*deafoult_size + deafoult_size +1
    WIN.fill((50,50,50))
    pygame.draw.rect(WIN,(50,200,50),pygame.Rect(WIDTH/2-map_segment_size/2,HEIGHT/2-map_segment_size/2,map_segment_size,map_segment_size))
    pygame.draw.rect(WIN,(0,0,0),pygame.Rect(WIDTH/2-map_segment_size/2 + map_border,HEIGHT/2-map_segment_size/2+map_border,map_segment_size-2*map_border,map_segment_size-2*map_border))
    Map = map_classes.map_class(deafoult_size,SCENARIO,WIN)
    def draw_window(WIN) :
        pygame.draw.rect(WIN,(0,0,0),pygame.Rect(WIDTH/2-map_segment_size/2 + map_border,HEIGHT/2-map_segment_size/2+map_border,map_segment_size-2*map_border,map_segment_size-2*map_border))
        Map.draw_everything(WIN)
        pygame.display.update()

    #moves every twenty frames
    game_speed = 15
    speed_counter = game_speed

    clock = pygame.time.Clock()
    while run :
        clock.tick(FPS)

        #Move check
        speed_counter -=1
        if speed_counter == 0:
            #move the snakes
            Map.update_move()
            speed_counter = game_speed
        draw_window(WIN)


        #The event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                pygame.quit()
                os._exit(0)
            if event.type == pygame.KEYDOWN :
                if event.key == pygame.K_ESCAPE :
                    run = False
                if Map.first_snake != None and Map.first_snake.direction_changed == False :
                    if event.key == pygame.K_w and Map.first_snake.direction %2 == 0 :
                        Map.first_snake.direction = 1
                        Map.first_snake.direction_changed = True
                    elif event.key == pygame.K_s and Map.first_snake.direction %2 == 0  :
                        Map.first_snake.direction = 3
                        Map.first_snake.direction_changed = True
                    elif event.key == pygame.K_a and Map.first_snake.direction %2 == 1 :
                        Map.first_snake.direction = 2
                        Map.first_snake.direction_changed = True
                    elif event.key == pygame.K_d and Map.first_snake.direction %2 == 1  :
                        Map.first_snake.direction = 4
                        Map.first_snake.direction_changed = True


