import numpy as np
class Mancala:
    def __init__(self):
        self.reset()

    def __str__(self):
        b = self.board
        return f"""
        \n[{b[6]}]  [{b[5]}]  [{b[4]}]  [{b[3]}]  [{b[2]}]  [{b[1]}]  [{b[0]}]  [{b[13]}]\n[ ]  [{b[7]}]  [{b[8]}]  [{b[9]}]  [{b[10]}]  [{b[11]}]  [{b[12]}]  [ ]\n
        """

    def reset(self):
        """
        -------------------------------
           |6| [5] [4] [3] [2] [1] [0]  |13|   <p1 goes this way
           | | [7] [8] [9] [10][11][12] |  |   >p2 goes this way
        ----------------------------------- 
        """ 
        self.board = np.full((14,), 6) 
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
        pits = slice(0, 6) if player == 1 else slice(7, 13)
        return np.sum(self.board[pits]) == 0
    
    def move(self, player, action):
        """
        only p1 should place on 6, and only p2 on 13
        """
        pos = action if player == 1 else action + 7

        # check if move is legal
        if self.is_illegal(player, action):
            return False, player, False
        
        num_marbles = self.board[pos]
        self.board[pos] = 0

        while num_marbles > 0:
            pos = (pos + 1) % 14

            if (pos == 6 and player == 2) or (pos == 13 and player == 1):
                continue  
            
            self.board[pos] += 1
            num_marbles -= 1

        next_player = 2 if player == 1 else 1 
        if (pos == 6 and player == 1) or (pos == 13 and player == 2):
            next_player = player

        elif self.board[pos] == 1:
            num_in_adjacent = self.board[12-pos]
            if num_in_adjacent > 0: next_player = player

            if player == 1: self.board[6] += num_in_adjacent
            else: self.board[13] += num_in_adjacent

        won = self.did_win(player)

        return True, next_player, won 
