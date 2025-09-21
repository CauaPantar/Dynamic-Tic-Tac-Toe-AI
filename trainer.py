from collections import deque


class Board:
    def __init__(self):
        self.board = [['' for _ in range(3)] for _ in range(3)]

        self.pieces = {'X': deque(maxlen=3), 'O': deque(maxlen=3)}

        self.winner = None

    def get_state(self):
        return tuple(item for row in self.board for item in row)

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
