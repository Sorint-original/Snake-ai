import pygame 
import os

FontGover = pygame.font.Font(None,35)

def gover(WIN,WIDTH,HEIGHT, SCENARIO,Status,score,other_score) :
    if SCENARIO == 0 :
        Govertext = FontGover.render("You lost, your score is "+str(score)+", press Esc to return to the menu",True,(255,0,0))
        Gover_rect = Govertext.get_rect()
        WIN.blit(Govertext,(WIDTH/2-Gover_rect[2]/2,25))
        pygame.display.update()
    elif SCENARIO == 1 :
        if Status == "first_died" :
            Govertext = FontGover.render("Plyer 1 lost with a score of "+str(score)+", Player 2 WON with a score of "+str(other_score)+", press Esc to return to the menu",True,(255,0,0))
        else :
            Govertext = FontGover.render("Plyer 2 lost with a score of "+str(score)+", Player 1 WON with a score of "+str(other_score)+", press Esc to return to the menu",True,(255,0,0))
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



