from collections import deque
import random
import pickle


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
        if self.board[row][col] != "":
            print("jogada invalida")
            return False

        if len(self.pieces[player]) == self.pieces[player].maxlen:
            oldest_row, oldest_col = self.pieces[player].popleft()
            self.board[oldest_row][oldest_col] = ""

        self.board[row][col] = player
        self.pieces[player].append((row, col))
        self.check_winner()
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
    def __init__(self, player_symbol, filename=None):
        self.player_symbol = player_symbol
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.history = []
        self.q_table = {}
        self.games_trained = 0
        self.exploration_rate = 1.0

        if filename:
            try:
                with open(filename, 'rb') as f:
                    brain_data = pickle.load(f)
                    self.q_table = brain_data['q_table']
                    self.games_trained = brain_data['games_trained']
                    self.exploration_rate = brain_data['exploration_rate']
                    print(f"Cérebro de {self.player_symbol} carregado! ({self.games_trained} partidas já treinadas)")
            except FileNotFoundError:
                print(f"Ficheiro não encontrado. A começar cérebro novo para {self.player_symbol}.")

    def choose_action(self, board):
        possible_moves = board.get_possible_moves()

        if not possible_moves:
            return None

        state = board.get_state()
        known_actions = self.q_table.get(state, {})
        valid_known_actions = {action: value for action, value in known_actions.items() if action in possible_moves}

        if random.random() < self.exploration_rate:
            action = random.choice(possible_moves)
        else:
            if valid_known_actions:
                action = max(valid_known_actions, key=valid_known_actions.get)
            else:
                action = random.choice(possible_moves)

        self.history.append((state, action))
        return action
    
    def update_q_table(self, reward):
        current_reward = reward

        for state, action in reversed(self.history):
            old_q_value = self.get_q_value(state, action)

            new_q_value = old_q_value + (self.learning_rate * (current_reward - old_q_value))

            self.q_table.setdefault(state, {})
            self.q_table[state][action] = new_q_value

            current_reward = new_q_value * self.discount_factor

        self.history = []

    def get_q_value(self, state, action):
        actions_dict = self.q_table.get(state, {})
        return actions_dict.get(action, 0.0)

def train(games_this_session=200000):
    print(f"Iniciando uma sessão de treino de {games_this_session} partidas...")

    brain_file_x = 'ia_cerebro_X.pkl'
    brain_file_o = 'ia_cerebro_O.pkl'
    player_x = Agent(player_symbol='X', filename=brain_file_x)
    player_o = Agent(player_symbol='O', filename=brain_file_o)

    total_target_games = player_o.games_trained + games_this_session
    save_points = {
        int(total_target_games * 0.10): 'ia_nivel_facil.pkl',
        int(total_target_games * 0.50): 'ia_nivel_medio.pkl',
        total_target_games: 'ia_nivel_dificil.pkl'
    }
    print(f"Novos save points calculados: {save_points}")

    for i in range(1, games_this_session + 1):
        board = Board()
        current_player_symbol = 'X'
        move_counter = 0

        while board.winner is None:
            current_player = player_x if current_player_symbol == 'X' else player_o

            action = current_player.choose_action(board)

            if action:
                board.make_move(action[0], action[1], current_player_symbol)
                move_counter += 1
            else:
                break

            if board.check_winner():
                break
            
            if move_counter >= 100:
                break

            current_player_symbol = 'O' if current_player_symbol == 'X' else 'X'

        player_x.games_trained += 1
        player_o.games_trained += 1

        if board.winner == 'X':
            player_x.update_q_table(reward=1)
            player_o.update_q_table(reward=-1)
        elif board.winner == 'O':
            player_x.update_q_table(reward=-1)
            player_o.update_q_table(reward=1)
        else: # Empate
            player_x.update_q_table(reward=0)
            player_o.update_q_table(reward=0)

        player_x.exploration_rate *= 0.99999
        player_o.exploration_rate *= 0.99999
        
        if i % 10000 == 0:
            print(f"Sessão atual: Partida {i}/{games_this_session}... (Epsilon atual: {player_o.exploration_rate:.4f})")
            
        if player_o.games_trained in save_points:
            filename = save_points[player_o.games_trained]
            brain_data_o = {
                'q_table': player_o.q_table,
                'games_trained': player_o.games_trained,
                'exploration_rate': player_o.exploration_rate
            }
            with open(filename, 'wb') as f:
                pickle.dump(brain_data_o, f)
            print(f"--- Cérebro da IA salvo em '{filename}'! (Total de {player_o.games_trained} partidas) ---")

    final_brain_data_x = {'q_table': player_x.q_table, 'games_trained': player_x.games_trained, 'exploration_rate': player_x.exploration_rate}
    final_brain_data_o = {'q_table': player_o.q_table, 'games_trained': player_o.games_trained, 'exploration_rate': player_o.exploration_rate}
    
    with open(brain_file_x, 'wb') as f:
        pickle.dump(final_brain_data_x, f)
    with open(brain_file_o, 'wb') as f:
        pickle.dump(final_brain_data_o, f)
        
    print(f"Treinamento concluído! Cérebros finais guardados.")

if __name__ == '__main__':
    train()
