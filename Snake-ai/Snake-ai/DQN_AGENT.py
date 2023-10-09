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




class sumtree(object):
    data_pointer = 0
    one_loop = 0

    def __init__ (self,capacity) :
       
        self.capacity = capacity
        self.tree = np.zeros(2*capacity -1)
        self.data = np.zeros(capacity,dtype=object)

    #add new expierience with a set priority
    def add(self,priority,data) :
        tree_index = self.data_pointer + self.capacity - 1
        
        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update (tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  
            self.data_pointer = 0   
            self.one_loop += 1

    #update the binary tree on all levels
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    
    def get_leaf(self, value):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]




class Memory(object):  # stored as ( state, action, reward, next_state ) in SumTree
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree 
        self.tree = sumtree(capacity)
    
    #Save a specific expirience with max priority or error_set_priority
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)   # set the max priority for new priority
     
    def sample(self, batch_size):
        # Create a minibatch array that will contains the minibatch
        minibatch = []

        b_idx = np.empty((batch_size,), dtype=np.int32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.tree[0] / batch_size       # priority segment
        for i in range(batch_size):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)
            b_idx[i]= index

            minibatch.append(data)

        return b_idx, minibatch
    


    def batch_update(self, tree_idx, abs_errors):
        abs_errors = abs_errors.detach()
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p.item())


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
        
        self.expirience_replay = Memory(10000)
        
        # Initialize discount 
        self.gamma = gamma
        self.criterion = nn.SmoothL1Loss()
        
        # Build main network and the target network
        self.q_network = Q_net(self._state_size,self._action_size).to(device)
        self.target_network = Q_net(self._state_size,self._action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self._optimizer = optim.Adam(self.q_network.parameters(),lr = learning_rate)
    
    #The function that stores past expiriences
    def store(self, state, action, reward, next_state, terminated):
        if terminated == True :
            next_state = None
        self.expirience_replay.store(Expirience(state,action,next_state,reward))
    
    #Get the action
    def act(self, state):
        with torch.no_grad():
            action = self.q_network(state).max(1)[1].view(1, 1)
            #print(action)
            return action

    
    #The ratraining 
    def retrain(self, batch_size):
        if self.expirience_replay.tree.data_pointer > batch_size or self.expirience_replay.tree.one_loop >=1 :
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
            leaf_index,expiriences = self.expirience_replay.sample(batch_size)
            batch = Expirience(*zip(*expiriences))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            batch_index = np.arange(batch_size,dtype = np.int32)
            
            #the values taken by the network
            state_action_values =  self.q_network(state_batch).gather(1,action_batch)

            next_state_values = torch.zeros(batch_size,device=self.device)
        
            with torch.no_grad():
                next_state_values[non_final_mask] = self.q_network(non_final_next_states).max(1)[0]


            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            indices = np.arange(batch_size, dtype=np.int32)
            errors = torch.abs(state_action_values.squeeze(1) - expected_state_action_values)
            # Update priority
            self.expirience_replay.batch_update(leaf_index, errors)
            
            #Compute loss
            loss = self.criterion( expected_state_action_values.unsqueeze(1),state_action_values)
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
        plt.ylim(top=550)
        plt.legend()
        plt.savefig(filename+".png")
        



