import pygame
import torch
from DQN_AGENT import DQN_Agent
from map_classes import deafoult_size, map_class
from copy import copy, deepcopy
import random
import numpy as np
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Learning_rate = 0.1
Gamma = 0.6

print(torch.cuda.is_available())

def training_ai(WIN,WIDTH,HEIGHT,FPS,SCENARIO) :
    #close the desplay for the training part to save processing power
    pygame.display.quit() 
    WIN = None
    print("load or new: ", end="")
    answer = input()
    if answer == "new":
        agent = DQN_Agent(deafoult_size,3,Learning_rate,Gamma,device)
    print("Epsilon decay: ", end="")
    epsilon = 1
    epsilon_min = 0.001
    epsilon_decay = float(input())
    print("number of episodes: ", end="")
    num_of_episodes = int(input())
    print("batch size: ", end="")
    batch_size = int(input())
    #logging vectors 
    Scores_log = []
    Epsilon_log = []
    Loss_log = []
    
    max_ep_time = 225

    for episode in range(1,num_of_episodes+1) :
        map = map_class(deafoult_size,SCENARIO,WIN)
        Status = None
        apple_count = 0
        episode_timer=max_ep_time
        
        procesing_matrix = deepcopy(map.tile_map)
        procesing_matrix[0] =  [ [x/3 for x in y] for y in procesing_matrix[0]]
        procesing_matrix[1] = [ [x/255 for x in y] for y in procesing_matrix[1]]
        procesing_matrix = torch.tensor([procesing_matrix])
        direction = torch.tensor([map.second_snake.direction/4])
        #the episode loop 
        while episode_timer > 0 :
            episode_timer -= 1
            #saveing apple
            ver_apple = deepcopy(map.apple)
            if np.random.rand() <= epsilon:
                action = torch.tensor([random.randint(0,2)],device=device,dtype = torch.long)
            else :
                action = agent.act(procesing_matrix,direction)
            if action.item() == 0:
                map.second_snake.direction -= 1;
                if map.second_snake.direction == 0 :
                    map.second_snake.direction = 4
            elif action.item() == 2:
                map.second_snake.direction += 1
                if map.second_snake.direction == 5 :
                    map.second_snake.direction = 1
            Status, score = map.update_move()
            #getting the reward for the action
            if Status == "nothing" :
                terminated = False
                if ver_apple != map.apple :
                    reward = 25 * map.second_snake.size
                    episode_timer = max_ep_time
                    apple_count += 1
                else :
                    reward = -1
            else :
                terminated = True
                reward = -100
            #getting new state
            nextprocesing_matrix = deepcopy(map.tile_map)
            nextprocesing_matrix[0] = [ [x/3 for x in y] for y in nextprocesing_matrix[0]]
            nextprocesing_matrix[1] = [ [x/255 for x in y] for y in nextprocesing_matrix[1]]
            nextprocesing_matrix = torch.tensor([nextprocesing_matrix])
            if terminated == False :
                nextdirection = torch.tensor([map.second_snake.direction/4])
            else:
                nextdirection = None
            
            reward = torch.tensor([reward])
            agent.store(procesing_matrix,direction,action,reward,nextprocesing_matrix,nextdirection,terminated)
            procesing_matrix = nextprocesing_matrix
            direction = nextdirection
            
            if terminated == True :
                break
            
        loss = agent.retrain(batch_size)
        print(f"Episode {episode}, Score: {apple_count}")
        Scores_log.append(apple_count);
        Epsilon_log.append(epsilon)
        Loss_log.append(loss)
        #Modify epsilon
        epsilon =  max(epsilon*epsilon_decay,epsilon_min)

    directory = "saved DQN/"
    #training finnished
    if answer == "new" :
        count = 0
        for path in os.listdir(directory):
            # check if current path is a file
            if os.path.isdir(os.path.join(directory, path)):
                count += 1

        Name = "DQN_NeNe_V" + str(count)
        aux_directory =  directory +Name
        os.mkdir(aux_directory)
        aux_directory = aux_directory + "/"+ Name
        iterations = range(1, num_of_episodes+1, 1)
        agent.save_model(iterations,Scores_log,Epsilon_log,Loss_log,aux_directory)
        




    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    
    return WIN