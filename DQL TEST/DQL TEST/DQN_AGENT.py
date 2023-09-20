from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten,Input, InputLayer
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random


class DQN_Agent:
    def __init__(self, optimizer,observation_space,action_space,trained_model = None):
        
        # Initialize atributes
        self._state_size = observation_space
        self._action_size = action_space
        self._optimizer = optimizer
        
        self.expirience_replay = deque(maxlen=2000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        model =  Sequential()
        model.add(Input(shape=(4,)))
        model.add(Dense(24,activation = "relu"))
        model.add(Dense(24,activation = "relu"))
        model.add(Dense(self._action_size,activation = "linear"))
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        q_values = self.q_network.predict(state,verbose=0)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch_indx = np.random.permutation(len(self.expirience_replay))[: batch_size]
        #print(states)
        states      = np.concatenate([np.reshape(self.expirience_replay[i][0],(1,4)) for i in minibatch_indx], axis=0)
        actions     = np.concatenate([np.reshape(self.expirience_replay[i][1],(1)) for i in minibatch_indx], axis=0)
        rewards     = np.concatenate([np.reshape(self.expirience_replay[i][2],(1)) for i in minibatch_indx], axis=0)
        next_states = np.concatenate([np.reshape(self.expirience_replay[i][3],(1,4))  for i in minibatch_indx], axis=0)
        dones       = np.concatenate([np.reshape(self.expirience_replay[i][4],(1)) for i in minibatch_indx], axis=0)

        X_batch = np.copy(states)
        Y_batch = np.zeros((batch_size, self._action_size), dtype=np.float64)
        
        qValues_batch = self.q_network.predict(states, verbose=0)
        qValuesNewState_batch = self.target_network.predict(next_states,verbose=0)

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



