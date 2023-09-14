
import pygame
import os
import button

def menu(WIN,WIDTH,HEIGHT,FPS):
 
    run  =True

    title = "Snake Game"
    #the declaration of the buttons that are in the menu
    Buttons = []
    B = button.Button((WIDTH/2-150,HEIGHT/3,300,150),(230,230,230),None,**{"text":"Play"})
    Buttons.append(B)

    def draw_window(WIN) :
        WIN.fill((150,205,205))
        for butt in Buttons :
            butt.update(WIN)
        pygame.display.update()
        
    while run :
        draw_window(WIN)


        for event in pygame.event.get() :
            if event.type == pygame.QUIT :
                pygame.quit()
                os._exit(0)

