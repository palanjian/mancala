import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np

from mancala import Mancala

# Register module as a gym environment. Once registered, id is usable in gym.make()
register(
    id='mancala',
    entry_point='mancala_env:MancalaEnv'
)

class MancalaEnv(gym.Env):
    # metadata is a required attribute
    # render_modes is either None or 'human'
    # render fps HAS to be declared as non-zero value (even if unused)
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        
        # Training code can call action_space.sample() to randomly select an action
        self.action_space = spaces.Discrete(6) # 6 possible moves at every turn

        # flattened 2x6 array, [...yours, ...theirs] 
        # TODO: do we need to add the banks?
        self.observation_space = spaces.Box(
            low=0,
            high=48,
            shape = (14,),
            dtype=np.int32
        )

        self.MAX_STEPS = 200
        self.current_player = 1
        self.mancala_game = Mancala()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # reset mancala
        self.mancala_game.reset()
        self.current_player = 1
        self.step_num = 0

        # construct the observation space
        obs = self.mancala_game.get_obs_for(self.current_player)
        
        #optional info to return
        info = {"current_player": self.current_player, "winner": None}

        if(self.render_mode == 'human'):
            self.render()

        return obs, info

    def step(self, action):
        self.step_num += 1
        # (Try to) perform action
        legal, next_player, terminal, won, scored = self.mancala_game.move(self.current_player, action)

        other = 2 if self.current_player == 1 else 1
        info = {
            "current_player": next_player, 
            "winner": self.current_player if won else other
        }

        self.current_player = next_player

        # going to try to add "or not legal" just to see if it works
        terminated = terminal or self.step_num > self.MAX_STEPS or not legal

        # construct the observation space
        obs = self.mancala_game.get_obs_for(self.current_player)

        reward = scored + 10 if won else scored 
        if terminated and not won: reward -= 10

        # check if its illegal
        if(self.render_mode == 'human'):
            self.render()
        
        return obs, reward, terminated, False, info
    
    def render(self):
        pass

    def get_action_mask(self):
        """returns 6x1 boolean mask"""
        return np.array([not self.mancala_game.is_illegal(self.current_player, i) for i in range(0, 6)], dtype=np.int8)
