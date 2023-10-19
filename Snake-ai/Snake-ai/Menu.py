
import pygame
import os

import button
import Game
import Training

def menu(WIN,WIDTH,HEIGHT,FPS):
 
    run  =True

    title = "Snake Game"
    #the declaration of the buttons that are in the menu\
    FontButton = pygame.font.Font(None,30)
    Buttons = []
    B = button.Button((WIDTH/2-300,HEIGHT*2/5,250,75),(230,230,230),None,**{"text":"SinglePlay","font":FontButton})
    Buttons.append(B)
    B = button.Button((WIDTH/2-300,HEIGHT*2/5+75*2,250,75),(230,230,230),None,**{"text":"Play 1v1","font":FontButton})
    Buttons.append(B)
    B = button.Button((WIDTH/2+50,HEIGHT*2/5,250,75),(230,230,230),None,**{"text":"Training","font":FontButton})
    Buttons.append(B)
    B = button.Button((WIDTH/2-125,HEIGHT*2/5+75*4,250,75),(230,230,230),None,**{"text":"Quit","font":FontButton})
    Buttons.append(B)


    def draw_window(WIN) :
        WIN.fill((50,50,50))
        for butt in Buttons :
            butt.update(WIN)
        pygame.display.update()
        

    clock = pygame.time.Clock()
    while run :
        clock.tick(FPS)

        draw_window(WIN)

        for event in pygame.event.get() :
            if event.type == pygame.QUIT :
                pygame.quit()
                os._exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN :
                for i in range(len(Buttons)) :
                    if Buttons[i].on_click(event) :
                        if i == 3 :
                            pygame.quit()
                            os._exit(0)
                        elif i == 2 :
                            WIN = Training.training_ai(WIN,WIDTH,HEIGHT,FPS,i)
                        else :
                            Game.game(WIN,WIDTH,HEIGHT,FPS,i)

