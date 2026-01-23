from collections import deque
import numpy as np
class Mancala:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def __str__(self):
        b = self.board
        return f"""
        \n[{b[13]}]  [{b[12]}]  [{b[11]}]  [{b[10]}]  [{b[9]}]  [{b[8]}]  [{b[7]}]  [{b[6]}]\n[ ]  [{b[0]}]  [{b[1]}]  [{b[2]}]  [{b[3]}]  [{b[4]}]  [{b[5]}]  [ ]\n
        """

    def reset(self):
        """
        -------------------------------
           |6| [5] [4] [3] [2] [1] [0]  |13|   <p1 goes this way
           | | [7] [8] [9] [10][11][12] |  |   >p2 goes this way
        ----------------------------------- 
        """ 
        self.board = np.full((14,), self.rng.uniform(1, 6)) 
        self.board[6] = 0
        self.board[13] = 0
        self.steps = 0
    
    def get_obs_for(self, player):
        if player == 1:
            return self.board.copy()
        return np.concatenate((self.board[7:], self.board[:7])).copy()
        
    def get_reward(self, player):
        return self.board[6] if player == 1 else self.board[13]
        
    def is_illegal(self, player, action):
        pos = action if player == 1 else action + 7
        return self.board[pos] == 0
    
    def did_win(self, player):
        """Returns terminal, won"""
        p1_side = np.sum(self.board[slice(0, 6)])
        p2_side = np.sum(self.board[slice(7, 13)])

        if player == 1 and p1_side == 0:
            self.board[13] += p2_side
            return True, self.board[6] > self.board[13]
        
        elif player == 2 and p2_side == 0:
            self.board[6] += p2_side
            return True, self.board[13] > self.board[6]
        
        # have to explicitely check for this, I never considered the fact that the other player
        # could win on your turn
        return p1_side == 0 or p2_side == 0, False
    
    def move(self, player, action):
        scored = 0

        """
        only p1 should place on 6, and only p2 on 13
        returns legal, next_player, terminal, target_rearched, scored
        """
        pos = action if player == 1 else action + 7
        next_player = 2 if player == 1 else 1 
        
        # check if move is legal
        if self.is_illegal(player, action):
            print(f"Illegal move! by player {player}. Tried to make move {action}.\n {self}")
            quit()
            
        num_marbles = self.board[pos]
        self.board[pos] = 0

        while num_marbles > 0:
            pos = (pos + 1) % 14

            if (pos == 6 and player == 2) or (pos == 13 and player == 1):
                continue  
            
            self.board[pos] += 1
            num_marbles -= 1

        if (pos == 6 and player == 1) or (pos == 13 and player == 2):
            next_player = player
            scored = 0.1

        # if you land on a pit with 0 marbles on it
        elif (player == 1 and pos in range(0,6) and self.board[pos] == 1) or \
        (player == 2 and pos in range(7,13) and self.board[pos] == 1):
            num_in_adjacent = self.board[12-pos]
            #if there actual is something in there
            if num_in_adjacent: 
                scored = 0.1 * num_in_adjacent 

                if player == 1: self.board[6] += num_in_adjacent + 1
                else: self.board[13] += num_in_adjacent + 1

                self.board[pos] = 0
                self.board[12-pos] = 0

        terminal, won = self.did_win(player)
        
        return True, next_player, terminal, won, scored
