import pickle
import random
import os
import sys
from trainer import Board, Agent 

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

class GameCLI:
    def __init__(self):
        self.board = Board()
        self.player_human = None
        self.player_ai = None
        self.ai_brain = None

    def print_board_with_info(self):
        print("=== Jogo da Velha Dinâmico com IA ===\n")
        display_board = [[' ' for _ in range(3)] for _ in range(3)]
        for player, pieces_deque in self.board.pieces.items():
            for i, (r, c) in enumerate(pieces_deque):
                display_board[r][c] = f"{player}{i+1}"
        print(" " + "-" * 13)
        for row in range(3):
            row_str = [cell.center(3) for cell in display_board[row]]
            print(f"|{'|'.join(row_str)}|")
            print(" " + "-" * 13)
        print("\nPeças em jogo (a mais antiga é a primeira):")
        human_pieces = self.board.pieces.get(self.player_human, [])
        ai_pieces = self.board.pieces.get(self.player_ai, [])
        print(f"  Você ('{self.player_human}'): {list(human_pieces)}")
        print(f"  IA    ('{self.player_ai}'): {list(ai_pieces)}")
        print("-" * 35)

    def get_human_move(self):
        while True:
            try:
                move_str = input(f"Sua jogada como '{self.player_human}' (linha,coluna), ex: 1,2 -> ")
                row, col = map(int, move_str.split(','))
                if 0 <= row <= 2 and 0 <= col <= 2 and self.board.board[row][col] == '':
                    return (row, col)
                else:
                    print("Jogada inválida! A casa não está vazia ou as coordenadas estão erradas.")
            except (ValueError, IndexError):
                print("Formato inválido! Por favor, use o formato 'linha,coluna'.")

    def run(self):
        clear_screen()
        print("Bem-vindo ao Duelo contra a IA!")
        player_choice = ''
        while player_choice.upper() not in ['X', 'O']:
            player_choice = input("Deseja jogar como 'X' (primeiro a jogar) ou 'O' (segundo a jogar)? [X/O] -> ")
        self.player_human = player_choice.upper()
        self.player_ai = 'O' if self.player_human == 'X' else 'X'
        print(f"\nÓtimo! Você jogará como '{self.player_human}'. A IA será '{self.player_ai}'.")

        dir_x = "IA X"
        dir_o = "IA O"
        
        if self.player_ai == 'X':
            brain_file = os.path.join(dir_x, 'ia_1B_specialist_X.pkl')
        else:
            brain_file = os.path.join(dir_o, 'ia_1B_specialist_O.pkl')
        
        print(f"A carregar IA Mestre ('{self.player_ai}')...")
        try:
            with open(brain_file, 'rb') as f:
                self.ai_brain = pickle.load(f)['q_table']
        except FileNotFoundError:
            print(f"ERRO CRÍTICO: O ficheiro '{brain_file}' não foi encontrado.")
            sys.exit()
            
        self.board.reset()
        current_player = 'X'
        while self.board.winner is None:
            self.print_board_with_info()
            if current_player == self.player_human:
                print(f"É a sua vez de jogar ('{self.player_human}').")
                row, col = self.get_human_move()
                self.board.make_move(row, col, self.player_human)
            else:
                print(f"IA ('{self.player_ai}') está a pensar...")
                ai_agent = Agent(self.player_ai)
                ai_agent.q_table = self.ai_brain
                ai_agent.exploration_rate = 0.0
                action = ai_agent.choose_action(self.board)
                if action:
                    self.board.make_move(action[0], action[1], self.player_ai)
                else: break
            if self.board.winner: break
            if len(self.board.get_possible_moves()) == 0 and len(self.board.pieces['X']) == 3 and len(self.board.pieces['O']) == 3: break
            current_player = 'O' if current_player == 'X' else 'X'
        self.print_board_with_info()
        print("\n--- FIM DE JOGO ---")
        if self.board.winner:
            if self.board.winner == self.player_human:
                print(f"Parabéns! Você venceu a IA Mestre!")
            else:
                print(f"A IA Mestre venceu. O vencedor é o jogador '{self.board.winner}'!")
        else:
            print("A partida terminou em empate!")

if __name__ == '__main__':
    game = GameCLI()
    game.run()
