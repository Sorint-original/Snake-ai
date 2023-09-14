
import pygame
import os

import button
import Game

def menu(WIN,WIDTH,HEIGHT,FPS):
 
    run  =True

    title = "Snake Game"
    #the declaration of the buttons that are in the menu\
    FontButton = pygame.font.Font(None,30)
    Buttons = []
    B = button.Button((WIDTH/2-150,HEIGHT*2/5,250,75),(230,230,230),None,**{"text":"Play","font":FontButton})
    Buttons.append(B)
    B = button.Button((WIDTH/2-150,HEIGHT*2/5+75*2,250,75),(230,230,230),None,**{"text":"Quit","font":FontButton})
    Buttons.append(B)

    def draw_window(WIN) :
        WIN.fill((50,50,50))
        for butt in Buttons :
            butt.update(WIN)
        pygame.display.update()
        
    while run :
        draw_window(WIN)


        for event in pygame.event.get() :
            if event.type == pygame.QUIT :
                pygame.quit()
                os._exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN :
                for i in range(len(Buttons)) :
                    if Buttons[i].on_click(event) :
                        if i == 1 :
                            pygame.quit()
                            os._exit(0)
                        elif i == 0 :
                            Game.game(WIN,WIDTH,HEIGHT,FPS,0)

