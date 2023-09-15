import pygame
import random

pygame.init()

screen = pygame.display.Info()
WIDTH = screen.current_w
HEIGHT = screen.current_h

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


tile_pixel = 50
map_border = 5
deafoult_size=15
map_segment_size = map_border*2 + tile_pixel*deafoult_size + deafoult_size +1
x_start =WIDTH/2-map_segment_size/2 + map_border
y_start = HEIGHT/2-map_segment_size/2+map_border

class map_class(object):
    def __init__ (self,size,SCENARIO,WIN) :
        self.size = size
        self.tile_map =[]
        #setup map tiles
        for i in range(size):
            aux = []
            for j in range(size):
                aux.append([0,0])
            self.tile_map.append(aux)
        #scenario managment
        if SCENARIO == 0 :
            #spawn snake
            self.first_snake = snake_class(0)
            for i in range(len(self.first_snake.segments_pos)) :
                self.tile_map[self.first_snake.segments_pos[i][0]][self.first_snake.segments_pos[i][1]] = [2,self.first_snake.size-i]
        #Spawn apple indferent of the scenario
        self.spawn_apple()
        self.draw_everything(WIN)


    def spawn_apple(self) :
        xs = []
        for i in range(self.size):
            xs.append(i)
        choice = None
        while choice == None :
            x = random.choice(xs)
            xs.remove(x)
            ys = []
            for i in range(self.size):
                if self.tile_map[x][i][0] == 0 :
                    ys.append(i)
            if len(ys) > 0 :
                choice = [x,random.choice(ys)]
        if choice != None :
            self.tile_map[choice[0]][choice[1]] = [1,0]
            self.apple = choice

    def draw_everything(self,WIN) :
        #draw apple
        pygame.draw.rect(WIN,(255,27,27),(x_start+self.apple[0]*tile_pixel+self.apple[0]+1,y_start+self.apple[1]*tile_pixel+self.apple[1]+1,tile_pixel,tile_pixel))
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
            elif self.tile_map[next_x][next_y][0] > 1  :
                #possible Game over
                if self.tile_map[next_x][next_y][0] == 2 and self.tile_map[next_x][next_y][1] != self.first_snake.size :
                    #GAME OVER 
                    score = self.first_snake.size - 3
                    self.kill_snake(1)
                    return "first_died", score
            elif self.tile_map[next_x][next_y][0] == 1 :
                #If the snake eats an apple
                self.first_snake.size +=1
                self.first_snake.segments_pos.insert(0,[next_x,next_y])
                self.tile_map[next_x][next_y] = [2,self.first_snake.size]
                self.spawn_apple()
            else :
                #Snake moves foward
                for i in range(self.first_snake.size-1,0,-1) :
                    segment = self.first_snake.segments_pos[i]
                    if i == self.first_snake.size-1 :
                        self.tile_map[segment[0]][segment[1]] = [0,0]
                        self.first_snake.segments_pos.remove(segment)
                    else :
                        self.tile_map[segment[0]][segment[1]][1] -= i
                #append the new head
                self.first_snake.segments_pos.insert(0,[next_x,next_y])
                self.tile_map[next_x][next_y] = [2,self.first_snake.size]
            #End of first snake movement
            if self.first_snake != None :
                self.first_snake.direction_changed = False
        return "nothing", None
            




    def kill_snake(self,which) :
        if which == 1 :
            #Kill the first snake
            for pos in self.first_snake.segments_pos :
                self.tile_map[pos[0]][pos[1]] = [0,0]
            self.first_snake = None



   

