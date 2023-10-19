import pygame
import random
import numpy as np

pygame.init()

screen = pygame.display.Info()
WIDTH = screen.current_w
HEIGHT = screen.current_h

tile_pixel = 50
map_border = 5
deafoult_size=15
map_segment_size = map_border*2 + tile_pixel*deafoult_size + deafoult_size +1
x_start =WIDTH/2-map_segment_size/2 + map_border
y_start = HEIGHT/2-map_segment_size/2+map_border

class snake_class(object):
    def __init__ (self,control) :
        self.size = 3
        self.control = control 
        self.segments_pos = []
        self.direction_changed = False
        if control == 0 :
            self.direction = 3
            self.segments_pos.append((0,2))
            self.segments_pos.append((0,1))
            self.segments_pos.append((0,0))
        else :
            self.direction = 1
            self.segments_pos.append((deafoult_size-1,deafoult_size-3))
            self.segments_pos.append((deafoult_size-1,deafoult_size-2))
            self.segments_pos.append((deafoult_size-1,deafoult_size-1))

class map_class(object):
    def __init__ (self,size,SCENARIO,WIN) :
        self.size = size
        self.tile_map =[]
        aux_matrix1 = []
        aux_matrix2 = []
        #setup map tiles
        for i in range(size):
            aux1 = []
            aux2 = []
            for j in range(size):
                aux1.append(0)
                aux2.append(0)
            aux_matrix1.append(aux1)
            aux_matrix2.append(aux2)
        self.tile_map = [aux_matrix1,aux_matrix2]
        #scenario managment
        self.first_snake = None
        self.second_snake = None
        if SCENARIO == 0 or SCENARIO == 1 :
            #spawn snake
            self.first_snake = snake_class(0)
            for i in range(len(self.first_snake.segments_pos)) :
                self.tile_map[0][self.first_snake.segments_pos[i][0]][self.first_snake.segments_pos[i][1]] = 2
                self.tile_map[1][self.first_snake.segments_pos[i][0]][self.first_snake.segments_pos[i][1]] = self.first_snake.size-i
        if SCENARIO >= 1 :
            #spawn second snake
            self.second_snake = snake_class(1)
            for i in range(len(self.second_snake.segments_pos)) :
                self.tile_map[0][self.second_snake.segments_pos[i][0]][self.second_snake.segments_pos[i][1]] = 3
                self.tile_map[1][self.second_snake.segments_pos[i][0]][self.second_snake.segments_pos[i][1]] = self.second_snake.size-i
        #Spawn apple indferent of the scenario
        self.spawn_apple()
        if  WIN :
            self.draw_everything(WIN)


    def spawn_apple(self) :
        xs = []
        for i in range(self.size):
            xs.append(i)
        choice = None
        while choice == None and len(xs) > 0 :
            x = random.choice(xs)
            xs.remove(x)
            ys = []
            for i in range(self.size):
                if self.tile_map[0][x][i] == 0 :
                    ys.append(i)
            if len(ys) > 0 :
                choice = [x,random.choice(ys)]
        if choice != None :
            self.tile_map[0][choice[0]][choice[1]] = 1
            self.tile_map[1][choice[0]][choice[1]] = 0
            self.apple = choice

    def draw_everything(self,WIN) :
        #draw apple
        pygame.draw.rect(WIN,(255,255,0),(x_start+self.apple[0]*tile_pixel+self.apple[0]+1,y_start+self.apple[1]*tile_pixel+self.apple[1]+1,tile_pixel,tile_pixel))
        #draw the first snake
        if self.first_snake != None :
            for i in range(self.first_snake.size) :
                if i == 0 :
                    pygame.draw.rect(WIN,(29,162,0),(x_start+self.first_snake.segments_pos[i][0]*tile_pixel+self.first_snake.segments_pos[i][0]+1,y_start+self.first_snake.segments_pos[i][1]*tile_pixel+self.first_snake.segments_pos[i][1]+1,tile_pixel,tile_pixel))
                else :
                    pygame.draw.rect(WIN,(43,255,0),(x_start+self.first_snake.segments_pos[i][0]*tile_pixel+self.first_snake.segments_pos[i][0]+1,y_start+self.first_snake.segments_pos[i][1]*tile_pixel+self.first_snake.segments_pos[i][1]+1,tile_pixel,tile_pixel))
                    #determine direction of conection
                    if self.first_snake.segments_pos[i][0] == self.first_snake.segments_pos[i-1][0] :
                        #up or down
                        if self.first_snake.segments_pos[i][1] > self.first_snake.segments_pos[i-1][1] :
                            #down
                            pygame.draw.rect(WIN,(43,255,0),(x_start+self.first_snake.segments_pos[i][0]*tile_pixel+self.first_snake.segments_pos[i][0]+1,y_start+self.first_snake.segments_pos[i][1]*tile_pixel+self.first_snake.segments_pos[i][1],tile_pixel,1))
                        else :
                            #up
                            pygame.draw.rect(WIN,(43,255,0),(x_start+self.first_snake.segments_pos[i][0]*tile_pixel+self.first_snake.segments_pos[i][0]+1,y_start+self.first_snake.segments_pos[i][1]*tile_pixel+self.first_snake.segments_pos[i][1]+1+tile_pixel,tile_pixel,1))
                    else :
                        #left or right
                        if self.first_snake.segments_pos[i][0] > self.first_snake.segments_pos[i-1][0] :
                            #right
                            pygame.draw.rect(WIN,(43,255,0),(x_start+self.first_snake.segments_pos[i][0]*tile_pixel+self.first_snake.segments_pos[i][0],y_start+self.first_snake.segments_pos[i][1]*tile_pixel+self.first_snake.segments_pos[i][1]+1,1,tile_pixel))
                        else :
                            #left
                            pygame.draw.rect(WIN,(43,255,0),(x_start+self.first_snake.segments_pos[i][0]*tile_pixel+self.first_snake.segments_pos[i][0]+1+tile_pixel,y_start+self.first_snake.segments_pos[i][1]*tile_pixel+self.first_snake.segments_pos[i][1]+1,1,tile_pixel))
        #Draw second snake
        if self.second_snake != None :
            for i in range(self.second_snake.size) :
                if i == 0 :
                    pygame.draw.rect(WIN,(162,29,0),(x_start+self.second_snake.segments_pos[i][0]*tile_pixel+self.second_snake.segments_pos[i][0]+1,y_start+self.second_snake.segments_pos[i][1]*tile_pixel+self.second_snake.segments_pos[i][1]+1,tile_pixel,tile_pixel))
                else :
                    pygame.draw.rect(WIN,(255,43,0),(x_start+self.second_snake.segments_pos[i][0]*tile_pixel+self.second_snake.segments_pos[i][0]+1,y_start+self.second_snake.segments_pos[i][1]*tile_pixel+self.second_snake.segments_pos[i][1]+1,tile_pixel,tile_pixel))
                    #determine direction of conection
                    if self.second_snake.segments_pos[i][0] == self.second_snake.segments_pos[i-1][0] :
                        #up or down
                        if self.second_snake.segments_pos[i][1] > self.second_snake.segments_pos[i-1][1] :
                            #down
                            pygame.draw.rect(WIN,(255,43,0),(x_start+self.second_snake.segments_pos[i][0]*tile_pixel+self.second_snake.segments_pos[i][0]+1,y_start+self.second_snake.segments_pos[i][1]*tile_pixel+self.second_snake.segments_pos[i][1],tile_pixel,1))
                        else :
                            #up
                            pygame.draw.rect(WIN,(255,43,0),(x_start+self.second_snake.segments_pos[i][0]*tile_pixel+self.second_snake.segments_pos[i][0]+1,y_start+self.second_snake.segments_pos[i][1]*tile_pixel+self.second_snake.segments_pos[i][1]+1+tile_pixel,tile_pixel,1))
                    else :
                        #left or right
                        if self.second_snake.segments_pos[i][0] > self.second_snake.segments_pos[i-1][0] :
                            #right
                            pygame.draw.rect(WIN,(255,43,0),(x_start+self.second_snake.segments_pos[i][0]*tile_pixel+self.second_snake.segments_pos[i][0],y_start+self.second_snake.segments_pos[i][1]*tile_pixel+self.second_snake.segments_pos[i][1]+1,1,tile_pixel))
                        else :
                            #left
                            pygame.draw.rect(WIN,(255,43,0),(x_start+self.second_snake.segments_pos[i][0]*tile_pixel+self.second_snake.segments_pos[i][0]+1+tile_pixel,y_start+self.second_snake.segments_pos[i][1]*tile_pixel+self.second_snake.segments_pos[i][1]+1,1,tile_pixel))

    def update_move(self):
        #moving the first snake
        if self.first_snake != None :
            if self.first_snake.direction %2 == 0 :
                #x -axis
                next_x = self.first_snake.segments_pos[0][0] + self.first_snake.direction - 3
                next_y = self.first_snake.segments_pos[0][1]
            else :
                next_x = self.first_snake.segments_pos[0][0]
                next_y = self.first_snake.segments_pos[0][1] + self.first_snake.direction - 2

            if next_x <0 or next_x >= self.size or next_y<0 or next_y >= self.size :
                #GAME OVER from wall
                score = self.first_snake.size - 3
                self.kill_snake(1)
                return "first_died",score
            
            #There si nothing in front of the snake or there is his tail which will move
            elif self.tile_map[0][next_x][next_y] == 0 or (self.tile_map[0][next_x][next_y] == 2 and self.tile_map[1][next_x][next_y] == 1) :
                #Snake moves foward
                for i in range(self.first_snake.size-1,-1,-1) :
                    segment = self.first_snake.segments_pos[i]
                    if i == self.first_snake.size-1 :
                        self.tile_map[0][segment[0]][segment[1]] = 0
                        self.tile_map[1][segment[0]][segment[1]] = 0
                        self.first_snake.segments_pos.remove(segment)
                    else :
                        self.tile_map[1][segment[0]][segment[1]] -= 1
                #append the new head
                self.first_snake.segments_pos.insert(0,[next_x,next_y])
                self.tile_map[0][next_x][next_y] = 2
                self.tile_map[1][next_x][next_y] = self.first_snake.size

            #hits a snake
            elif self.tile_map[0][next_x][next_y] > 1  :
                #possible Game over
                if self.tile_map[0][next_x][next_y] == 2 and self.tile_map[1][next_x][next_y] != 1 :
                    #GAME OVER 
                    score = self.first_snake.size - 3
                    self.kill_snake(1)
                    return "first_died", score
                if self.second_snake != None and self.tile_map[0][next_x][next_y] == 3 :
                    #GAME OVER 
                    score = self.first_snake.size - 3
                    self.kill_snake(1)
                    return "first_died", score
            #eats an apple
            elif self.tile_map[0][next_x][next_y] == 1 :
                #If the snake eats an apple
                self.first_snake.size +=1
                self.first_snake.segments_pos.insert(0,[next_x,next_y])
                self.tile_map[0][next_x][next_y] = 2
                self.tile_map[1][next_x][next_y] = self.first_snake.size
                self.spawn_apple()
            #End of first snake movement
            if self.first_snake != None :
                self.first_snake.direction_changed = False
        #moving second_snake
        if self.second_snake != None :
            if self.second_snake.direction %2 == 0 :
                #x -axis
                next_x = self.second_snake.segments_pos[0][0] + self.second_snake.direction - 3
                next_y = self.second_snake.segments_pos[0][1]
            else :
                next_x = self.second_snake.segments_pos[0][0]
                next_y = self.second_snake.segments_pos[0][1] + self.second_snake.direction - 2

            if next_x <0 or next_x >= self.size or next_y<0 or next_y >= self.size :
                #GAME OVER from wall
                score = self.second_snake.size - 3
                self.kill_snake(2)
                return "second_died",score
            elif self.tile_map[0][next_x][next_y] == 0 or (self.tile_map[0][next_x][next_y] == 3 and self.tile_map[1][next_x][next_y] == 1) :
                #Snake moves foward
                for i in range(self.second_snake.size-1,-1,-1) :
                    segment = self.second_snake.segments_pos[i]
                    if i == self.second_snake.size-1 :
                        self.tile_map[0][segment[0]][segment[1]] = 0
                        self.tile_map[1][segment[0]][segment[1]] = 0
                        self.second_snake.segments_pos.remove(segment)
                    else :
                        self.tile_map[1][segment[0]][segment[1]] -= 1
                #append the new head
                self.second_snake.segments_pos.insert(0,[next_x,next_y])
                self.tile_map[0][next_x][next_y] = 3
                self.tile_map[1][next_x][next_y] = self.second_snake.size
            elif self.tile_map[0][next_x][next_y] > 1  :
                #possible Game over
                if self.tile_map[0][next_x][next_y] == 3 and self.tile_map[1][next_x][next_y] != 1 :
                    #GAME OVER 
                    score = self.second_snake.size - 3
                    self.kill_snake(2)
                    return "second_died", score
                elif self.first_snake != None and self.tile_map[0][next_x][next_y] == 2 :
                    #GAME OVER 
                    score = self.second_snake.size - 3
                    self.kill_snake(2)
                    return "second_died", score
            elif self.tile_map[0][next_x][next_y] == 1 :
                #If the snake eats an apple
                self.second_snake.size +=1
                self.second_snake.segments_pos.insert(0,[next_x,next_y])
                self.tile_map[0][next_x][next_y] = 3
                self.tile_map[1][next_x][next_y] = self.second_snake.size
                self.spawn_apple()
            #End of first snake movement
            if self.second_snake != None :
                self.second_snake.direction_changed = False

        return "nothing", None
            




    def kill_snake(self,which) :
        if which == 1 :
            #Kill the first snake
            for pos in self.first_snake.segments_pos :
                self.tile_map[0][pos[0]][pos[1]] = 0
                self.tile_map[1][pos[0]][pos[1]] = 0
            self.first_snake = None
        else :
            #kill second snake
            for pos in self.second_snake.segments_pos :
                self.tile_map[0][pos[0]][pos[1]] = 0
                self.tile_map[1][pos[0]][pos[1]] = 0
            self.second_snake = None




   

