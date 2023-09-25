import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque, namedtuple
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import time

from memory_profiler import profile  



class Q_net(nn.Module):
    def __init__(self,_state_size,_action_size) :
        super(Q_net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(_state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,_action_size)
            )
        
    def forward(self, x):
        return self.model(x)











Expirience = namedtuple('Expirience',('state', 'action', 'next_state', 'reward'))

class DQN_Agent:
    def __init__(self,observation_space,action_space,learning_rate,gamma,device, trained_model = None):
        
        # Initialize base statistics
        self._state_size = observation_space
        self._action_size = action_space
        self.device = device

        self. network_sync_counter = 0
        self.network_sync_freq = 10
        
        self.expirience_replay = deque(maxlen=10000)
        
        # Initialize discount 
        self.gamma = gamma
        
        # Build main network and the target network
        self.q_network = Q_net(self._state_size,self._action_size).to(device)
        self.target_network = Q_net(self._state_size,self._action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self._optimizer = optim.Adam(self.q_network.parameters(),lr = learning_rate)
    
    #The function that stores past expiriences
    def store(self, state, action, reward, next_state, terminated):
        if terminated == True :
            next_state = None
        self.expirience_replay.append(Expirience(state,action,next_state,reward))
    
    #Get the action
    def act(self, state):
        with torch.no_grad():
            action = self.q_network(state).max(1)[1].view(1, 1)
            #print(action)
            return action

    
    #The ratraining 
    def retrain(self, batch_size):
        if len(self.expirience_replay) > batch_size :
            #align models
            self.network_sync_counter += 1  
            if(self.network_sync_counter == self.network_sync_freq):
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.network_sync_counter = 0
            #The first step in setting the optimization
            self._optimizer.zero_grad()
            self.q_network.train()
            self.target_network.eval()
            # Get variables from random expiriences and form a batch
            expiriences = random.sample(self.expirience_replay,batch_size)
            batch = Expirience(*zip(*expiriences))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            batch_index = np.arange(batch_size,dtype = np.int32)
            #the values taken by the network

            #state_action_values = self.q_network(state_batch).gather(1, action_batch)
            state_action_values =  self.q_network(state_batch).gather(1,action_batch)

            next_state_values = torch.zeros(batch_size,device=self.device)
        
            with torch.no_grad():
                next_state_values[non_final_mask] = self.q_network(non_final_next_states).max(1)[0]


            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            #Compute loss
            criterion = nn.SmoothL1Loss()
            loss = criterion( expected_state_action_values.unsqueeze(1),state_action_values)
            loss.backward()
            self._optimizer.step()
            torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
            return loss.item()
        else :
            return 0 
        
           

    def save_model(self,iterations,score_log,epsilon_log,loss_log, filename):
        #Save model
        torch.save(self.q_network.state_dict(), filename)
        #Save graphs
        plt.plot(iterations, epsilon_log, label = "Randomization", color = 'green')
        plt.plot(iterations,loss_log, label = "Loss",color="red")
        plt.plot(iterations, score_log, label = "Score", color = 'tab:blue')
        #calculate average
        average = []
        for i in range(24,len(score_log)):
            recent_score_sum = 0
            for j in range(i-24,i+1) :
                recent_score_sum += score_log[j]
            average.append(recent_score_sum/25)
        iterations = range(25, len(epsilon_log)+1, 1)
        plt.plot(iterations, average, label = "Score Average",color = 'tab:orange')
        plt.ylabel('Return/Randomization factor')
        plt.xlabel('Iterations')
        plt.ylim(top=250)
        plt.legend()
        plt.savefig(filename+".png")
        



