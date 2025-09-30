import pickle
import random
import os
from trainer import Board, Agent

def choose_action_competitiva(q_table, state, possible_moves):
    known_actions = q_table.get(state, {})
    valid_known_actions = {a: v for a, v in known_actions.items() if a in possible_moves}
    if not valid_known_actions:
        return random.choice(possible_moves) if possible_moves else None
    max_q = max(valid_known_actions.values())
    best = [a for a, v in valid_known_actions.items() if v == max_q]
    return random.choice(best)

def run_tournament(player_1_id, player_2_id, num_games=1000):
    p1_name = os.path.basename(player_1_id).replace('.pkl', '')
    p2_name = os.path.basename(player_2_id).replace('.pkl', '')
    print(f"\n INICIANDO TORNEIO: {p1_name} (X) vs {p2_name} (O) --- [{num_games} partidas]")

    try:
        with open(player_1_id, 'rb') as f: brain_x = pickle.load(f)['q_table']
        with open(player_2_id, 'rb') as f: brain_o = pickle.load(f)['q_table']
    except FileNotFoundError as e: print(f"ERRO: {e.filename} não encontrado!"); return
    
    stats = {'X': 0, 'O': 0, 'Draw': 0}
    for i in range(num_games):
        board = Board()
        current_symbol = 'X'; move_count = 0
        while board.winner is None and move_count < 100:
            brain = brain_x if current_symbol == 'X' else brain_o
            moves = board.get_possible_moves()
            if not moves: break
            action = choose_action_competitiva(brain, board.get_state(), moves)
            if not action: break
            board.make_move(action[0], action[1], current_symbol)
            move_count += 1
            if board.winner: break
            current_symbol = 'O' if current_symbol == 'X' else 'X'
        if board.winner:
            stats[board.winner] += 1
        else:
            stats['Draw'] += 1
        if (i + 1) % 100 == 0: print(f"  ... Partidas: {i + 1}/{num_games}")
    print(f"Vitórias de {p1_name} (X): {stats['X']} ({stats['X']/num_games:.1%})")
    print(f"Vitórias de {p2_name} (O): {stats['O']} ({stats['O']/num_games:.1%})")
    print(f"Empates: {stats['Draw']} ({stats['Draw']/num_games:.1%})")

if __name__ == '__main__':
    # exemplo de teste entre IAs
    dir_x = "IA X"
    dir_o = "IA O"
    
    mestre_100k_X = os.path.join(dir_x, 'ia_100k_specialist_X.pkl')
    mestre_1B_O = os.path.join(dir_o, 'ia_1B_specialist_O.pkl')
    
    print("="*60)
    print("O DUELO ENTRE IAs")
    print("="*60)
    
    run_tournament(mestre_100k_X, mestre_1B_O)