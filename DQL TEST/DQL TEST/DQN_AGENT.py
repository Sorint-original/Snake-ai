
from tensorflow.keras.optimizers import Adam

import torch.nn as nn

from collections import deque
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import gc

from memory_profiler import profile  


class DQN_Agent:
    def __init__(self, optimizer,observation_space,action_space,trained_model = None):
        
        # Initialize base statistics
        self._state_size = observation_space
        self._action_size = action_space
        self._optimizer = optimizer
        
        self.expirience_replay = deque(maxlen=2000)
        
        # Initialize discount 
        self.gamma = 0.6
        
        # Build main network and the target network
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()
    #The function that stores past expiriences
    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    #Buld the neural network
    def _build_compile_model(self):
        model = nn.Sequential(
            nn.Linear(self._state_size,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,self._action_size)
            )
        return model
        '''
        model =  Sequential()
        model.add(Input(shape=(4,)))
        model.add(Dense(32,activation = "relu"))
        model.add(Dense(32,activation = "relu"))
        model.add(Dense(self._action_size,activation = "linear"))
        model.compile(loss='mse', optimizer=self._optimizer)
        return model
        '''    
    #Alighn the target network with the main network
    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    #Get the action
    def act(self, state):
        q_values = self.q_network.predict(state,verbose=0)
        _ = gc.collect()
        return np.argmax(q_values[0])
    #The ratraining 
    def retrain(self, batch_size):
        # Get variables from random expiriences and form a batch
        minibatch_indx = np.random.permutation(len(self.expirience_replay))[: batch_size]
        states      = np.concatenate([np.reshape(self.expirience_replay[i][0],(1,4)) for i in minibatch_indx], axis=0)
        actions     = np.concatenate([np.reshape(self.expirience_replay[i][1],(1)) for i in minibatch_indx], axis=0)
        rewards     = np.concatenate([np.reshape(self.expirience_replay[i][2],(1)) for i in minibatch_indx], axis=0)
        next_states = np.concatenate([np.reshape(self.expirience_replay[i][3],(1,4))  for i in minibatch_indx], axis=0)
        dones       = np.concatenate([np.reshape(self.expirience_replay[i][4],(1)) for i in minibatch_indx], axis=0)

        X_batch = np.copy(states)
        Y_batch = np.zeros((batch_size, self._action_size), dtype=np.float64)
        
        qValues_batch = self.q_network.predict(states, verbose=0)
        _ = gc.collect()
        qValuesNewState_batch = self.target_network.predict(next_states,verbose=0)
        _ = gc.collect()
        targetValue_batch = np.copy(rewards)
        targetValue_batch += (1 - dones) * self.gamma * np.amax(qValuesNewState_batch, axis=1)
        
        for idx in range(batch_size):
            targetValue = targetValue_batch[idx]
            Y_sample = qValues_batch[idx]
            Y_sample[actions[idx]] = targetValue
            Y_batch[idx] = Y_sample

            if dones[idx]:
                X_batch = np.append(X_batch, np.reshape(np.copy(next_states[idx]), (1, self._state_size)), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards[idx]] * self._action_size]), axis=0)

        self.q_network.fit(X_batch, Y_batch, batch_size=len(X_batch), epochs=1, verbose=0)
        

    def save_model(self,iterations,score_log,epsilon_log, filename):
        #Save model
        self.q_network.save(filename+".keras")
        #Save graphs
        plt.plot(iterations, epsilon_log, label = "Randomization", color = 'green')
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
        



