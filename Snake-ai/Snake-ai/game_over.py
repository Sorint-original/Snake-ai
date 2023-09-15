import pygame 
import os

FontGover = pygame.font.Font(None,35)

def gover(WIN,WIDTH,HEIGHT, SCENARIO,score) :
    if SCENARIO == 0 :
        Govertext = FontGover.render("You lost, your score is "+str(score)+" press Esc to return to the menu",True,(255,0,0))
        Gover_rect = Govertext.get_rect()
        WIN.blit(Govertext,(WIDTH/2-Gover_rect[2]/2,25))
        pygame.display.update()

    run = True
    while run :
        for event in pygame.event.get() :
            if event.type == pygame.QUIT :
                pygame.quit()
                os._exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE :
                run = False



