import pygame
import os

import map_classes
from map_classes import tile_pixel, map_border,deafoult_size
import game_over
import time


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
    Status = None

    clock = pygame.time.Clock()
    while run :
        clock.tick(FPS)

        #Move check
        speed_counter -=1
        if speed_counter <= 0:
            #move the snakes
            Status, score = Map.update_move()
            if Status == "first_died" or Status == "second_died" :
                run = False 
                #Enter in game over part
                if SCENARIO == 0 :
                    game_over.gover(WIN,WIDTH,HEIGHT,SCENARIO,Status,score,None)
                elif SCENARIO == 1 :
                    if Status == "first_died" :
                        other_score = Map.second_snake.size - 3
                    else :
                        other_score = Map.first_snake.size - 3
                    game_over.gover(WIN,WIDTH,HEIGHT,SCENARIO,Status,score,other_score)
            elif SCENARIO == 0 :
                speed_counter = game_speed - 0.25*(Map.first_snake.size/2)
            elif SCENARIO == 1 :
                speed_counter = game_speed - 0.125*(Map.first_snake.size+ Map.second_snake.size)
        draw_window(WIN)


        #The event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                pygame.quit()
                os._exit(0)
            if event.type == pygame.KEYDOWN :
                if event.key == pygame.K_ESCAPE :
                    run = False
                #First snake manual movements
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
                #Second snake manual movements
                if Map.second_snake != None and Map.second_snake.direction_changed == False :
                    if event.key == pygame.K_UP and Map.second_snake.direction %2 == 0 :
                        Map.second_snake.direction = 1
                        Map.second_snake.direction_changed = True
                    elif event.key == pygame.K_DOWN and Map.second_snake.direction %2 == 0  :
                        Map.second_snake.direction = 3
                        Map.second_snake.direction_changed = True
                    elif event.key == pygame.K_LEFT and Map.second_snake.direction %2 == 1 :
                        Map.second_snake.direction = 2
                        Map.second_snake.direction_changed = True
                    elif event.key == pygame.K_RIGHT and Map.second_snake.direction %2 == 1  :
                        Map.second_snake.direction = 4
                        Map.second_snake.direction_changed = True



