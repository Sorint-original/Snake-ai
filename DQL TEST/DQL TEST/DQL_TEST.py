import random
import gym

from tensorflow.keras.models import Sequential

env = gym.make("CartPole-v1")


episodes = 100
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