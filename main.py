import sys
import numpy as np
import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95 # how important are future reward over current reward
EPISODES = 25000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
                        
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) # 20x20x3

def get_discrete_state(state : list[float, float]):
    return tuple(((state - env.observation_space.low) / discrete_os_win_size).astype(np.int16))

discrete_state = get_discrete_state(env.reset()[0])

done = False

while not done:
    action = np.argmax(q_table[discrete_state])
    new_state, reward, done, _, _ = env.step(action)
    
    new_discrete_state = get_discrete_state(new_state)
    
    env.render()

    if not done:
        max_future_q = np.max(q_table[new_discrete_state])
    
env.close()