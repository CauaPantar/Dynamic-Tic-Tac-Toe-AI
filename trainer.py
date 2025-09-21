from collections import deque
import random


class Board:
    def __init__(self):
        self.board = [['' for _ in range(3)] for _ in range(3)]

        self.pieces = {'X': deque(maxlen=3), 'O': deque(maxlen=3)}

        self.winner = None

    def get_state(self):
        board_tuple = tuple(item for row in self.board for item in row)
        x_pieces_tuple = tuple(self.pieces['X'])
        o_pieces_tuple = tuple(self.pieces['O'])
        
        # 3. Combina tudo numa única tupla imutável.
        return (board_tuple, x_pieces_tuple, o_pieces_tuple)


    def make_move(self, row, col, player):
        if self.board[row][col] != "":
            print("jogada invalida")
            return False

        if len(self.pieces[player]) == self.pieces[player].maxlen:
            oldest_row, oldest_col = self.pieces[player].popleft()
            self.board[oldest_row][oldest_col] = ""

        self.board[row][col] = player
        self.pieces[player].append((row, col))
        return True

    def check_winner(self):
        for i in range(3): # verifica linhas
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != "":
                self.winner = self.board[i][0]
                return self.board[i][0]
        
        for i in range(3): # verifica coluna
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != "":
                self.winner = self.board[0][i]
                return self.board[0][i]
        
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != "": # verifica diagonal principal
                self.winner = self.board[0][0]
                return self.board[0][0]
        
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != "": # verifica diagonal secundaria
                self.winner = self.board[0][2]
                return self.board[0][2]

        return None

    def get_possible_moves(self):
        possible = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == "":
                    possible.append((i, j))

        return possible

    def reset(self):
        self.__init__()

    def print_board(self):
        print("-" * 13)
        for row in self.board:
            print(f"| {' | '.join(p if p != '' else ' ' for p in row)} |")
            print("-" * 13)

class Agent:
    def __init__(self, player_symbol):
        self.player_symbol = player_symbol
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1
        self.history = []

    def choose_action(self, board):
        possible_moves = board.get_possible_moves()

        if not possible_moves:
            return None

        state = board.get_state()

        if random.random() < self.exploration_rate:
            action = random.choice(possible_moves)
        else:
            if state in self.q_table and self.q_table[state]:
                action = max(self.q_table[state], key=self.q_table[state].get)
            else:
                action = random.choice(possible_moves)

        self.history.append((state, action))
        return action
    
    def update_q_table(self, reward):
        current_reward = reward
        
        for state, action in reversed(self.history):
            old_q_value = self.get_q_value(state, action)

            new_q_value = old_q_value + self.learning_rate * (current_reward - old_q_value)

            self.q_table.setdefault(state, {})
            self.q_table[state][action] = new_q_value

            current_reward = new_q_value * self.discount_factor

        self.history = []

    def get_q_value(self, state, action):
        actions_dict = self.q_table.get(state, {})

        return actions_dict.get(action, 0.0)
