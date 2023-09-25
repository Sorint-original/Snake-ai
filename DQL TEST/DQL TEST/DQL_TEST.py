from functools import lru_cache
import random
import gym
import numpy as np
import os
import torch
from memory_profiler import profile


from DQN_AGENT import DQN_Agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



env = gym.make("CartPole-v1")
states = env.observation_space.shape[0]
actions = env.action_space.n
Learning_rate = 1e-4
Gamma = 0.6

agent = DQN_Agent(states,actions,Learning_rate,Gamma,device)

batch_size = 256
num_of_episodes = 1000
timesteps_per_episode = 200
  

epsilon = 1
epsilon_decay = 0.995
epsilon_min = 0.001
Scores_log = []
Epsilon_log = []
Loss_log = []
#Training loop
for e in range(1, num_of_episodes+1):
    # Reset the enviroment
    state = env.reset()
    state = torch.tensor(state,dtype = torch.float32,device=device).unsqueeze(0)
    score = 0
    # Initialize variables
    terminated = False
    
    for timestep in range(timesteps_per_episode): 
        
        # Take action    
        if np.random.rand() <= epsilon:
            action = torch.tensor([[env.action_space.sample()]],device=device,dtype = torch.long)
        else :
            action = agent.act(state)
        next_state, reward, terminated, info = env.step(action.item()) 
        next_state = torch.tensor(next_state,dtype = torch.float32,device=device).unsqueeze(0)
        score += reward
        reward = torch.tensor([reward],device=device)
        agent.store(state, action, reward, next_state, terminated)
        state = next_state
        if terminated:
            break
            
        
    
    loss =agent.retrain(batch_size)
    print(f"Episode {e}, Score: {score}")
    Scores_log.append(score);
    Epsilon_log.append(epsilon*200)
    Loss_log.append(loss*200)
    #Modify epsilon
    epsilon =  max(epsilon*epsilon_decay,epsilon_min)

#saveing Neural Network model
directory = "Saved models/DQN/"

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


#visualization loop

for episode in range(1, 11) :
    state = env.reset()
    state = torch.tensor(state,dtype = torch.float32,device=device).unsqueeze(0)
    done = False
    score = 0
    
    while not done :
        action = agent.act(state)
        state, reward,done,_ = env.step(action.item())
        state = torch.tensor(state,dtype = torch.float32,device=device).unsqueeze(0)
        score += reward
        env.render()
        
    print(f"Episode {episode}, Score: {score}")
    
env.close()




