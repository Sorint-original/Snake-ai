import random
import gym
import numpy as np
import os
import gc

from memory_profiler import profile  
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from DQN_AGENT import DQN_Agent
@profile
def everything():

    env = gym.make("CartPole-v1")
    states = env.observation_space.shape[0]
    actions = env.action_space.n

    optimizer = Adam(learning_rate=0.01)


    agent = DQN_Agent(optimizer,states,actions)

    batch_size = 32
    num_of_episodes = 250
    timesteps_per_episode = 200
    episodes_between_training = 1
    episodes_between_alignment = 100
    alignment_count = episodes_between_alignment

    epsilon = 1
    epsilon_decay = 0.995
    epsilon_min = 0.001
    Scores_log = []
    Epsilon_log = []
    #Training loop
    for e in range(1, num_of_episodes+1):
        # Reset the enviroment
        state = env.reset()
        state = np.reshape(state,(1,4))
        score = 0
        # Initialize variables
        terminated = False
        alignment_count -= 1
    
        for timestep in range(timesteps_per_episode): 
        
            # Take action    
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else :
                action = agent.act(state)
            next_state, reward, terminated, info = env.step(action) 
            score += reward
            agent.store(state, action, reward, next_state, terminated)
        
            state = next_state
            state = np.reshape(state,(1,4))
        
            if terminated:
                break
            
        if len(agent.expirience_replay) > batch_size and e% episodes_between_training == 0:
            agent.retrain(batch_size)
        
    
        print(f"Episode {e}, Score: {score}")
        Scores_log.append(score);
        Epsilon_log.append(epsilon*200)
        #Modify epsilon
        epsilon =  max(epsilon*epsilon_decay,epsilon_min)

        if alignment_count <= 0 : 
            agent.alighn_target_model()

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
    agent.save_model(iterations,Scores_log,Epsilon_log,aux_directory)


    #visualization loop

    for episode in range(1, 11) :
        state = env.reset()
        state = np.reshape(state,(1,4))
        done = False
        score = 0
    
        while not done :
            q_values = agent.q_network(state)
            action = np.argmax(q_values[0])
            state, reward,done,_ = env.step(action)
            state = np.reshape(state,(1,4))
            score += reward
            env.render()
        
        print(f"Episode {episode}, Score: {score}")
    
    env.close()

everything()


