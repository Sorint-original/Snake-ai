from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten,Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random


class DQN_Agent:
    def __init__(self, optimizer,observation_space,action_space):
        
        # Initialize atributes
        self._state_size = observation_space
        self._action_size = action_space
        self._optimizer = optimizer
        
        self.expirience_replay = deque(maxlen=2000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        model =  Sequential()
        model.add(Input(shape=(4,)))
        model.add(Dense(32,activation = "relu"))
        model.add(Dense(32,activation = "relu"))
        model.add(Dense(self._action_size,activation = "linear"))
        print(model.input_shape)
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state,env):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        print(len(state))
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)
        
        for state, action, reward, next_state, terminated in minibatch:
            
            target = self.q_network.predict(state)
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.q_network.fit(state, target, epochs=1, verbose=0)




