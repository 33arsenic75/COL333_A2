import time
import math
import random
import numpy as np
from helper import *
from typing import Optional, Set, Tuple  # Add this import statement
from helper import check_win

class AIPlayer:

    def __init__(self, player_number: int, timer):
        """
        Intitialize the AIPlayer Agent

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game
        
        `timer: Timer`
            - a Timer object that can be used to fetch the remaining time for any player
            - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer

    def get_move(self, state: np.array) -> Tuple[int, int]:        
        valid_moves = self.get_valid_moves(state)
        for move in valid_moves:
            me_win, _ = check_win(state, move, self.player_number)
            opp_win, _ = check_win(state, move, 3 - self.player_number)
            if me_win:
                return move
            elif opp_win:
                return move

        return random.choice(valid_moves)

    def get_valid_moves(self, state: np.array) -> List[Tuple[int, int]]:
        """
        Get all valid moves for the current state.

        # Parameters
        `state (np.array)`: The current state of the game board

        # Returns
        `List[Tuple[int, int]]`: A list of valid moves (row, column)
        """
        valid_moves = []
        for row in range(state.shape[0]):
            for col in range(state.shape[1]):
                if state[row, col] == 0:  # Assuming 0 represents an empty cell
                    valid_moves.append((row, col))
        return valid_moves

    