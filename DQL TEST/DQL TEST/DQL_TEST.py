import random
import gym
import numpy as np
import progressbar


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


from DQN_AGENT import DQN_Agent


env = gym.make("CartPole-v1")
states = env.observation_space.shape[0]
actions = env.action_space.n

#the structure of the neural network
model =  Sequential()
model.add(Flatten(input_shape=(1,states)))
model.add(Dense(24,activation = "relu"))
model.add(Dense(24,activation = "relu"))
model.add(Dense(actions,activation = "linear"))

optimizer = Adam(learning_rate=0.01)


agent = DQN_Agent(optimizer,states,actions)

batch_size = 32
num_of_episodes = 100
timesteps_per_episode = 1000

#Training loop
for e in range(0, num_of_episodes):
    # Reset the enviroment
    state = env.reset()
    
    # Initialize variables
    reward = 0
    terminated = False
    
    bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=\
                                  [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for timestep in range(timesteps_per_episode):
        # Run Action
        action = agent.act(state,env)
        
        # Take action    
        next_state, reward, terminated, info = env.step(action) 
        agent.store(state, action, reward, next_state, terminated)
        
        state = next_state
        
        if terminated:
            agent.alighn_target_model()
            break
            
        if len(agent.expirience_replay) > batch_size:
            agent.retrain(batch_size)
        
        if timestep%10 == 0:
            bar.update(timestep/10 + 1)
    
    bar.finish()
    if (e + 1) % 10 == 0:
        print("**********************************")
        print("Episode: {}".format(e + 1))
        env.render()
        print("**********************************")

#Old version of tensorflow 2.10
'''
#the structure of the neural network
model =  Sequential()
model.add(Flatten(input_shape=(1,states)))
model.add(Dense(24,activation = "relu"))
model.add(Dense(24,activation = "relu"))
model.add(Dense(actions,activation = "linear"))

#creating the DQNagent

agent = DQNAgent(
    model=model,
    memory = SequentialMemory(limit=50000,window_length = 1),
    policy = BoltzmannQPolicy(),
    nb_actions = actions,
    nb_steps_warmup = 10,
    target_model_update = 0.01
)

agent.compile(Adam(lr = 0.001))
agent.fit(env,nb_steps = 100000,visualize = False, verbose = 1)

results = agent.test(env, nb_episodes=10, visualize = True)
print(np.mean(results.history["episode_reward"]))

env.close()
'''



#Without deep learning version
"""

episodes = 10
for episode in range(1, episodes + 1) :
    state = env.reset()
    done = False
    score = 0
    
    while not done :
        action = random.choice([0,1])
        _, reward,done,_ = env.step(action)
        score += reward
        env.render()
        
    print(f"Episode {episode}, Score: {score}")
    
env.close()

"""