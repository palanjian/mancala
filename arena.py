from collections import deque
import numpy as np
from tqdm import tqdm
from agent import DQNAgent
from mancala_env import MancalaEnv
import os 

env = MancalaEnv()

p1 = DQNAgent("Peter")
p2 = DQNAgent("Bob")

EPISODES = 30000

# exploration settings
epsilon = 1
EPSILON_DECAY = 0.998
MIN_EPSILON = 0.01

one_episode_behind = None

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
    # these are for the agent currently being trained
    current_state, info = env.reset()

    done = False
    while not done:            
        current_player = info["current_player"]

        agent = p1 if current_player == 1 else p2

        # get the valid actions for player (mask)
        mask = env.get_action_mask()

        # if we're the agent being trained & we're 'exploring'
        if current_player == 1 and np.random.random() < epsilon:
            action = env.action_space.sample(mask)

        # non training agent should just exploit
        else:     
            qs = agent.get_qs(current_state) 
            masked_qs = np.where(mask, qs, -np.inf)
            action = np.argmax(masked_qs)
            
        new_state, reward, terminal, _, info = env.step(action)

        if (current_player == 1):
            #observation space, action, reward, new observation space, done or not
            agent.update_replay_memory((current_state, action, reward, new_state, terminal))
            agent.train(terminal)
        
        current_state = new_state
        done = terminal

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
        #print(f"New epsilon: {epsilon}")
    
    # set p2's weights to p1's old weights 
    if one_episode_behind: p2.model.set_weights(one_episode_behind)
    one_episode_behind = p1.target_model.get_weights()


print("Saving model.")
if os.path.exists("best.keras"):
    os.remove("best.keras")
p1.model.save("best.keras")
