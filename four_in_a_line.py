#!/usr/bin/env python3
"""4-in-a-line on an 8x8 board.

Computer = X, human = O.  Four in a row or column wins (diagonals do not count).

Search: iterative-deepening minimax with alpha-beta pruning.  A cut-off test
(time limit) replaces the terminal test; if time runs out mid-iteration the
unfinished iteration is discarded and the best move from the last completed
depth is played.

Evaluation: every 4-cell window (row or column) containing pieces of only one
player is worth a weight that grows with the piece count (positive for X,
negative for O).  The total is maintained incrementally, so evaluating a node
is O(1).
"""
import sys
import time

N = 8
EMPTY, X, O = 0, 1, 2
WIN = 1_000_000
INF = 10 ** 9
WEIGHT = [0, 1, 10, 100]  # value of a window holding k pieces of one player

# ---- precomputed geometry --------------------------------------------------
WINDOWS = []
for r in range(N):
    for c in range(N - 3):
        WINDOWS.append([r * N + c + k for k in range(4)])
for c in range(N):
    for r in range(N - 3):
        WINDOWS.append([(r + k) * N + c for k in range(4)])

CELL_WINDOWS = [[] for _ in range(N * N)]
for wi, w in enumerate(WINDOWS):
    for cell in w:
        CELL_WINDOWS[cell].append(wi)

NEIGHBORS = [[] for _ in range(N * N)]
for r in range(N):
    for c in range(N):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < N and 0 <= c + dc < N:
                    NEIGHBORS[r * N + c].append((r + dr) * N + c + dc)

# cells ordered centre-first (good for pruning)
ORDER = sorted(range(N * N), key=lambda i: abs(i // N - 3.5) + abs(i % N - 3.5))

# VAL[cx][co]: contribution of a window with cx X's and co O's (X's viewpoint)
VAL = [[0] * 5 for _ in range(5)]
for cx in range(5):
    for co in range(5):
        if cx and co:
            continue
        if cx:
            VAL[cx][co] = WEIGHT[cx] if cx < 4 else 0
        elif co:
            VAL[cx][co] = -WEIGHT[co] if co < 4 else 0


class Timeout(Exception):
    pass


class Game:
    def __init__(self):
        self.board = [EMPTY] * (N * N)
        self.cnt = [[0, 0, 0] for _ in WINDOWS]  # [_, #X, #O] per window
        self.score = 0
        self.near = [0] * (N * N)  # number of occupied neighbours
        self.filled = 0
        # search state
        self.deadline = 0.0
        self.nodes = 0

    # -- board mutation (returns True if the move completes 4 in a line) -----
    def make(self, cell, p):
        self.board[cell] = p
        self.filled += 1
        won = False
        cnt, score = self.cnt, self.score
        for wi in CELL_WINDOWS[cell]:
            c = cnt[wi]
            score -= VAL[c[1]][c[2]]
            c[p] += 1
            if c[p] == 4:
                won = True
            score += VAL[c[1]][c[2]]
        self.score = score
        for nb in NEIGHBORS[cell]:
            self.near[nb] += 1
        return won

    def undo(self, cell, p):
        self.board[cell] = EMPTY
        self.filled -= 1
        cnt, score = self.cnt, self.score
        for wi in CELL_WINDOWS[cell]:
            c = cnt[wi]
            score -= VAL[c[1]][c[2]]
            c[p] -= 1
            score += VAL[c[1]][c[2]]
        self.score = score
        for nb in NEIGHBORS[cell]:
            self.near[nb] -= 1

    def candidates(self):
        b, near = self.board, self.near
        if self.filled == 0:
            return [3 * N + 3]
        return [i for i in ORDER if b[i] == EMPTY and near[i]]

    # -- alpha-beta -----------------------------------------------------------
    def ab(self, depth, alpha, beta, player, ply):
        self.nodes += 1
        if (self.nodes & 1023) == 0 and time.time() >= self.deadline:
            raise Timeout
        if depth == 0 or self.filled == N * N:
            return self.score  # cut-off test: depth limit reached
        if player == X:
            best = -INF
            for m in self.candidates():
                if self.make(m, X):
                    v = WIN - ply
                else:
                    v = self.ab(depth - 1, alpha, beta, O, ply + 1)
                self.undo(m, X)
                if v > best:
                    best = v
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
            return best
        best = INF
        for m in self.candidates():
            if self.make(m, O):
                v = -WIN + ply
            else:
                v = self.ab(depth - 1, alpha, beta, X, ply + 1)
            self.undo(m, O)
            if v < best:
                best = v
            if best < beta:
                beta = best
            if alpha >= beta:
                break
        return best

    def best_move(self, seconds):
        """Iterative deepening alpha-beta.  Returns (move, depth_completed)."""
        self.deadline = time.time() + seconds - 0.05
        self.nodes = 0
        moves = self.candidates()
        best_move, best_depth = moves[0], 0
        depth = 1
        while depth <= N * N - self.filled:
            try:
                alpha, cur_best, cur_val = -INF, None, -INF
                for m in moves:
                    if self.make(m, X):
                        v = WIN
                    else:
                        try:
                            v = self.ab(depth - 1, alpha, INF, O, 1)
                        except Timeout:
                            self.undo(m, X)
                            raise
                    self.undo(m, X)
                    if v > cur_val:
                        cur_val, cur_best = v, m
                    if v > alpha:
                        alpha = v
            except Timeout:
                break
            best_move, best_depth = cur_best, depth
            if cur_val >= WIN - 100:  # forced win found
                break
            # search the best move first on the next iteration
            moves.remove(cur_best)
            moves.insert(0, cur_best)
            if cur_val <= -WIN + 100 and depth > 2:
                break  # lost anyway; stop wasting time
            depth += 1
        return best_move, best_depth


# ---- user interface --------------------------------------------------------
def show(game):
    print("\n  " + " ".join(str(c + 1) for c in range(N)))
    sym = {EMPTY: "-", X: "X", O: "O"}
    for r in range(N):
        print(chr(ord("A") + r) + " " + " ".join(sym[game.board[r * N + c]] for c in range(N)))
    print()


def name(cell):
    return chr(ord("A") + cell // N) + str(cell % N + 1)


def parse(text):
    t = text.strip().upper().replace(" ", "")
    if len(t) != 2 or not ("A" <= t[0] <= "H") or not ("1" <= t[1] <= "8"):
        return None
    return (ord(t[0]) - 65) * N + int(t[1]) - 1


def ask_yes_no(prompt):
    while True:
        a = input(prompt).strip().lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False
        print("Please answer y or n.")


def ask_time():
    while True:
        a = input("How long should the computer think about its moves (in seconds, max 30, "
                  "blank = 5)? : ").strip()
        if a == "":
            return 5.0
        try:
            t = float(a)
        except ValueError:
            print("Please enter a number.")
            continue
        if 0 < t <= 30:
            return t
        print("Time must be greater than 0 and at most 30 seconds.")


def main():
    game = Game()
    human_first = ask_yes_no("Would you like to go first? (y/n): ")
    seconds = ask_time()
    show(game)
    turn = O if human_first else X
    while True:
        if turn == O:
            while True:
                try:
                    text = input("Choose your next move: ")
                except EOFError:
                    return
                cell = parse(text)
                if cell is None:
                    print("Invalid move. Enter a row letter A-H followed by a column number 1-8 (e.g. D4).")
                elif game.board[cell] != EMPTY:
                    print("That square is already taken. Try again.")
                else:
                    break
            won = game.make(cell, O)
            show(game)
            if won:
                print("You win!\nGame Over!")
                return
        else:
            start = time.time()
            move, depth = game.best_move(seconds)
            won = game.make(move, X)
            print(f"Computer plays {name(move)}  (searched {depth} plies, "
                  f"{game.nodes} nodes, {time.time() - start:.2f}s)")
            show(game)
            if won:
                print("Computer wins!\nGame Over!")
                return
        if game.filled == N * N:
            print("It's a draw!\nGame Over!")
            return
        turn = X if turn == O else O


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
