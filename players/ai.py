import time
import math
import random
import numpy as np
from helper import *
from typing import Optional, Set, Tuple, List  # Add List to the import statement
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
        self.C = 1.2  # Exploration constant for UCT

    def get_move(self, state: np.array) -> Tuple[int, int]:        
        valid_moves = self.get_valid_moves(state)
        for move in valid_moves:
            me_win, _ = check_win(state, move, self.player_number)
            opp_win, _ = check_win(state, move, 3 - self.player_number)
            if me_win:
                return move
            elif opp_win:
                return move
        if state.shape[0] > 10 or state.shape[1] > 10:
            # Play randomly for the first few moves if the board size is greater than 10
            if np.count_nonzero(state) < 20:  # Adjust the number of moves as needed
                return random.choice(self.get_valid_moves(state))
            
        # return self.random_move(state)
        return self.mcts(state)

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

    def mcts(self, state: np.array) -> Tuple[int, int]:
        root = Node(state, None, None)
        self.visits = {root: 0}
        self.wins = {root: 0}
        for _ in range(1000):  # Number of iterations
            node = self.select(root)
            if node is None:
                continue
            self.expand(node)
            result = self.simulate(node)
            self.backpropagate(node, result)
        if not root.children:
            return random.choice(self.get_valid_moves(state))  # Fallback to a random move if no children
        return max(root.children, key=lambda n: self.visits.get(n, 0)).move

    def select(self, node):
        # UCT selection strategy
        best_value = -float('inf')
        best_node = None
        for child in node.children:
            if child not in self.visits:
                return child
            uct_value = (self.wins[child] / self.visits[child]) + \
                        self.C * np.sqrt(np.log(self.visits[node]) / self.visits[child])
            if uct_value > best_value:
                best_value = uct_value
                best_node = child
        return best_node

    def expand(self, node):
        valid_moves = self.get_valid_moves(node.state)
        for move in valid_moves:
            new_state = node.state.copy()
            new_state[move] = self.player_number
            new_node = Node(new_state, move, node)
            node.children.append(new_node)

    def simulate(self, node):
        current_state = node.state.copy()
        current_player = self.player_number
        while True:
            valid_moves = self.get_valid_moves(current_state)
            if not valid_moves:
                return 0  # Draw
            move = random.choice(valid_moves)
            current_state[move] = current_player
            if check_win(current_state, move, current_player):
                return 1 if current_player == self.player_number else -1
            current_player = 3 - current_player

    def backpropagate(self, node, result):
        while node is not None:
            if node not in self.visits:
                self.visits[node] = 0
                self.wins[node] = 0
            self.visits[node] += 1
            self.wins[node] += result
            node = node.parent
            result = -result

class Node:
    def __init__(self, state, move, parent):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        
    def __hash__(self):
        return hash(str(self.state.tostring()) + str(self.move))
    
    def __eq__(self, other):
        return isinstance(other, Node) and np.array_equal(self.state, other.state) and self.move == other.move
