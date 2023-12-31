from concurrent.futures import thread
import math
import pygame
import torch
from DQN_AGENT import DQN_Agent
from map_classes import deafoult_size, map_class
from copy import copy, deepcopy
import random
import numpy as np
import time
import os
import psutil
import GPUtil 
import threading


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
process = psutil.Process()
total_memory_percent = (((psutil.virtual_memory().total)/(1024**3))/100)

ram = []
gpu = []
cpu = []

test = True
def get_process_data() :  
    global test,ram,cpu,gpu
    while test == True :
        ram.append((process.memory_info().rss/(1024**3))/total_memory_percent)
        GPUs = GPUtil.getGPUs()
        load = GPUs[0].load
        gpu.append(load*100)
        cpu.append(process.cpu_percent()/psutil.cpu_count())
        time.sleep(60)
        
    


def training_ai(WIN,WIDTH,HEIGHT,FPS,SCENARIO) :
    global test ,ram,cpu,gpu
    #clearing the comands
    os.system('cls')
    #reseting performance vectors
    ram = []
    gpu = []
    cpu = []
    test = True
    #close the desplay for the training part to save processing power
    pygame.display.quit() 
    WIN = None
    print("learning rate: ",end = "")
    Learning_rate = float(input())
    print("Gamma: ",end = "")
    Gamma = float(input())
    print("load numbe or (-1) new model :", end="")
    answer = input()
    if answer == "-1":
        agent = DQN_Agent(deafoult_size,3,Learning_rate,Gamma,device)
    else:
        agent = DQN_Agent(deafoult_size,3,Learning_rate,Gamma,device,answer)
    print("Epsilon start: ", end="")  
    epsilon = float(input())
    print("Epsilon decay: ", end="")
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

    max_apple_distance = round(math.sqrt(2*((deafoult_size-1)**2)))

    eval_thread = threading.Thread(target = get_process_data)
    eval_thread.start()

    for episode in range(1,num_of_episodes+1) :
        map = map_class(deafoult_size,SCENARIO,WIN)
        Status = None
        apple_count = 0
        episode_timer=max_ep_time
        

        procesing_matrix = deepcopy([map.tile_map[0]])
        procesing_matrix = torch.tensor([procesing_matrix],device=device,dtype = torch.float32)
        info = torch.tensor([[map.second_snake.direction/4,(map.apple[0]-map.second_snake.segments_pos[0][0])/(deafoult_size-1),(map.apple[1]-map.second_snake.segments_pos[0][1])/(deafoult_size-1)]],device=device,dtype = torch.float32)
        time = 0
        #the episode loop 
        while episode_timer > 0 :
            time+=1
            episode_timer -= 1
            #saveing apple
            ver_apple = deepcopy(map.apple)
            if np.random.rand() <= epsilon:
                action = torch.tensor([random.randint(0,2)],device=device,dtype = torch.long)
            else :
                action = agent.act(procesing_matrix,info)
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
                    reward = map.second_snake.size
                    episode_timer = max_ep_time
                    apple_count += 1
                else :
                    apple_distance = math.sqrt((map.apple[0]-map.second_snake.segments_pos[0][0])**2 + (map.apple[1]-map.second_snake.segments_pos[0][1])**2)
                    #reward = 1/apple_distance
                    reward = 1/(apple_distance-max_apple_distance)
            else :
                terminated = True
                reward = -10
            #getting new state
            nextprocesing_matrix = deepcopy([map.tile_map[0]])
            nextprocesing_matrix = torch.tensor([nextprocesing_matrix],device=device,dtype = torch.float32)
            if terminated == False :
                nextinfo = torch.tensor([[map.second_snake.direction/4,(map.apple[0]-map.second_snake.segments_pos[0][0])/(deafoult_size-1),(map.apple[1]-map.second_snake.segments_pos[0][1])/(deafoult_size-1)]],device=device,dtype = torch.float32)
            else:
                nextinfo = None
            
            reward = torch.tensor([reward],device=device)
            agent.store(procesing_matrix,info,action,reward,nextprocesing_matrix,nextinfo,terminated)
            procesing_matrix = nextprocesing_matrix
            info = nextinfo
            
            if terminated == True :
                break
            
        loss = agent.retrain(batch_size)
        print(f"Episode {episode}, Score: {apple_count}, Time: {time}, Epsilon: {epsilon}")


        
        Scores_log.append(apple_count);
        Epsilon_log.append(epsilon)
        Loss_log.append(loss)
        #Modify epsilon
        epsilon =  max(epsilon*epsilon_decay,epsilon_min)

    test = False
    directory = "saved DQN"
    
    #training finnished
    iterations = range(1, num_of_episodes+1, 1)

    eval_thread.join()
    
    if answer == "-1" :
        count = 0
        for path in os.listdir(directory):
            # check if current path is a file
            if os.path.isdir(os.path.join(directory, path)):
                count += 1

        Name = "DQN_NeNe_V" + str(count)
        aux_directory =  directory+"/" +Name
        os.mkdir(aux_directory)
        aux_directory = aux_directory + "/"+ Name
        agent.save_model(iterations,Scores_log,Epsilon_log,Loss_log,aux_directory,ram,gpu,cpu)
    else :
        generations = answer.split('.')
        aux_directory = directory
        for i in range(len(generations)) :
            if i == 0 :
                Name = "DQN_NeNe_V"+str(generations[i])
            else :
                Name = Name+"."+str(generations[i])
            aux_directory =  aux_directory+"/" +Name
        agent.save_model(iterations,Scores_log,Epsilon_log,Loss_log,aux_directory,ram,gpu,cpu)
        




    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    
    return WIN