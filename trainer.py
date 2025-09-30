import pickle
from collections import deque
import random
import time
import os

class Board:
    def __init__(self):
        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.pieces = {'X': deque(maxlen=3), 'O': deque(maxlen=3)}
        self.winner = None

    def get_state(self):
        board_tuple = tuple(item for row in self.board for item in row)
        x_pieces_tuple = tuple(self.pieces['X'])
        o_pieces_tuple = tuple(self.pieces['O'])
        return (board_tuple, x_pieces_tuple, o_pieces_tuple)

    def make_move(self, row, col, player):
        if self.board[row][col] != '':
            return False
        if len(self.pieces[player]) == self.pieces[player].maxlen:
            oldest_r, oldest_c = self.pieces[player].popleft()
            self.board[oldest_r][oldest_c] = ''
        self.board[row][col] = player
        self.pieces[player].append((row, col))
        self.check_winner()
        return True

    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] and self.board[i][0] != '':
                self.winner = self.board[i][0]
                return self.winner
            if self.board[0][i] == self.board[1][i] == self.board[2][i] and self.board[0][i] != '':
                self.winner = self.board[0][i]
                return self.winner
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != '':
            self.winner = self.board[0][0]
            return self.winner
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != '':
            self.winner = self.board[0][2]
            return self.winner
        self.winner = None
        return None

    def get_possible_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == '']

    def reset(self):
        self.__init__()

class Agent:
    def __init__(self, player_symbol):
        self.player_symbol = player_symbol
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.history = []
        self.q_table = {}
        self.exploration_rate = 1.0

    def get_q_value(self, state, action):
        return self.q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, board):
        possible_moves = board.get_possible_moves()
        if not possible_moves:
            return None

        state = board.get_state()

        if random.random() < self.exploration_rate:
            action = random.choice(possible_moves)
        else:
            known_actions = self.q_table.get(state, {})
            valid_known_actions = {a: v for a, v in known_actions.items() if a in possible_moves}
            if valid_known_actions:
                action = max(valid_known_actions, key=valid_known_actions.get)
            else:
                action = random.choice(possible_moves)

        self.history.append((state, action))
        
        return action

    def update_q_table(self, reward):
        current_reward = reward
        for state, action in reversed(self.history):
            old_q = self.get_q_value(state, action)
            new_q = old_q + self.learning_rate * (current_reward - old_q)
            self.q_table.setdefault(state, {})
            self.q_table[state][action] = new_q
            current_reward = new_q * self.discount_factor
        self.history = []

def train(games_this_session=100_000):
    start_time = time.time()
    print(f"Iniciando treino de especialistas por {games_this_session} partidas...")

    player_x = Agent(player_symbol='X')
    player_o = Agent(player_symbol='O')

    dir_x = "IA X"
    dir_o = "IA O"
    os.makedirs(dir_x, exist_ok=True)
    os.makedirs(dir_o, exist_ok=True)

    start_rate = 1.0
    min_rate = 0.01
    decay_games = int(games_this_session * 0.80)
    decay_rate = (start_rate - min_rate) / decay_games if decay_games > 0 else 0
    feedback_interval = games_this_session // 20 if games_this_session >= 20 else 1

    for i in range(1, games_this_session + 1):
        board = Board()
        current_symbol = 'X'
        move_count = 0
        while board.winner is None and move_count < 100:
            current_player = player_x if current_symbol == 'X' else player_o
            action = current_player.choose_action(board)
            if action:
                board.make_move(action[0], action[1], current_symbol)
                move_count += 1
            else: break
            if board.winner: break
            current_symbol = 'O' if current_symbol == 'X' else 'X'
        if board.winner == 'X':
            player_x.update_q_table(1); player_o.update_q_table(-1)
        elif board.winner == 'O':
            player_x.update_q_table(-1); player_o.update_q_table(1)
        else:
            player_x.update_q_table(0); player_o.update_q_table(0)
        if i < decay_games and player_x.exploration_rate > min_rate:
            player_x.exploration_rate -= decay_rate
            player_o.exploration_rate -= decay_rate
        if feedback_interval > 0 and i % feedback_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"  Progresso: {i/games_this_session:.0%} | Epsilon: {player_o.exploration_rate:.4f} | Tempo: {elapsed_time:.1f}s")

    level = f"{games_this_session // 1_000_000}M" if games_this_session >= 1_000_000 else f"{games_this_session // 1000}k"
    brain_file_x = os.path.join(dir_x, f'ia_{level}_specialist_X.pkl')
    brain_file_o = os.path.join(dir_o, f'ia_{level}_specialist_O.pkl')
    
    with open(brain_file_x, 'wb') as f:
        pickle.dump({'q_table': player_x.q_table}, f)
    with open(brain_file_o, 'wb') as f:
        pickle.dump({'q_table': player_o.q_table}, f)

    total_time = time.time() - start_time
    print(f"\nTreino de especialistas concluído em {total_time/60:.2f} minutos!")
    print(f"Cérebros finais guardados em '{brain_file_x}' e '{brain_file_o}'.")

if __name__ == '__main__':
    # exemplo de como treinar uma IA
    train(games_this_session=100_000)