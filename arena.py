from collections import deque
import numpy as np
from tqdm import tqdm
from agent import DQNAgent
from mancala_env import MancalaEnv
import os 

env = MancalaEnv()

p1 = DQNAgent("Peter")
p2 = DQNAgent("Bob")

EPISODES = 3000
ROUNDS_PER_EP = 10

# exploration settings
epsilon = 1
EPSILON_DECAY = 0.99925
MIN_EPSILON = 0.01

training_p1 = True

p1_win_history = deque(maxlen=100)
p1_pr = 0

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
    round = 0

    for round in range(0, ROUNDS_PER_EP):
        if round >= ROUNDS_PER_EP//2:
            training_p1 = not training_p1
        
        # these are for the agent currently being trained
        current_state, info = env.reset()

        done = False
        while not done:            
            current_player = info["current_player"]
            agent = p1 if current_player == 1 else p2

            if (training_p1 and current_player == 1) or (not training_p1 and current_player == 2):
                # if this is the agent being trained -> explore
                if np.random.random() > epsilon:
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    action = env.action_space.sample()
            else:
                # non training agent should just exploit
                action = np.argmax(agent.get_qs(current_state))
                
            new_state, reward, terminal, _, info = env.step(action)

            if (training_p1 and current_player == 1) or (not training_p1 and current_player == 2):
                #observation space, action, reward, new observation space, done or not
                agent.update_replay_memory((current_state, action, reward, new_state, terminal))
                agent.train(terminal)
            
            current_state = new_state
            done = terminal
        
        if current_player == 1 and reward == 1:
            p1_win_history.append(1)
        else: p1_win_history.append(0)

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
        print(f"New epsilon: {epsilon}")
    

    p1_win_rate = np.mean(p1_win_history)
    print(f"P1 avg win rate (last 100): {p1_win_rate:.2f}")

    if p1_win_rate > p1_pr:
        print("Saving model as we have peaked in average win rate")
        p1_pr = p1_win_rate
        if os.path.exists("best.keras"):
            os.remove("best.keras")
        p1.model.save("best.keras")
