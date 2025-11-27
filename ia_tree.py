from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
from chess import WHITE
import random

# ————————————————————————————————
#   OUVERTURE (book étendu et original)
# ————————————————————————————————

OPENING_BOOK = {
    # Ouvertures BLANC
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1': [
        'e4', 'd4', 'Nf3', 'b3', 'f4', 'Nc3'
    ],

    # Noirs vs e4
    'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2': [
        'e5', 'c5', 'e6', 'c6',
        'Qd6',  # Scandinave moderne
        'Nc6',  # Nimzowitsch
    ],

    # Noirs vs d4
    'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1': [
        'd5', 'Nf6', 'e6',
        'e5',   # Englund
        'c5',   # Benoni
    ]
}


# ————————————————————————————————
#      Valeurs matérielles
# ————————————————————————————————

PIECE_VALUES = {
    PAWN: 100,
    KNIGHT: 300,
    BISHOP: 325,
    ROOK: 500,
    QUEEN: 950,
    KING: 20000,
}

# ————————————————————————————————
#      Piece-Square Tables
# ————————————————————————————————

# Pions
PAWN_TABLE_WHITE = [...]
PAWN_TABLE_BLACK = list(reversed(PAWN_TABLE_WHITE))

# Cavaliers (inchangé)
KNIGHT_TABLE = [...]

# Fous – NEW
BISHOP_TABLE_WHITE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  5,  5,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5, 10, 10, 10, 10,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5, 10, 10, 10, 10,  5,-10,
    -10,  0,  0,  5,  5,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
BISHOP_TABLE_BLACK = list(reversed(BISHOP_TABLE_WHITE))

# Roi – milieu de jeu (NEW)
KING_MG_TABLE_WHITE = [...]
KING_MG_TABLE_BLACK = list(reversed(KING_MG_TABLE_WHITE))

# Roi – finale (NEW)
KING_EG_TABLE_WHITE = [...]
KING_EG_TABLE_BLACK = list(reversed(KING_EG_TABLE_WHITE))


# ————————————————————————————————
#                    IA
# ————————————————————————————————

class TreeIA:
    def __init__(self, depth=2):
        self.depth = depth
        self.transposition_table = {}
        self.opening_moves_played = 0

    # ——————————————————————————————
    #      ÉVALUATION AMÉLIORÉE
    # ——————————————————————————————
    def evaluate(self) -> int:

        # 1. Mat et nuls
        if self.board.is_checkmate():
            return -100000 if self.board.turn == WHITE else 100000
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0

        score = 0

        # 2. Matériel
        for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN]:
            score += PIECE_VALUES[pt] * (
                len(self.board.pieces(pt, WHITE)) -
                len(self.board.pieces(pt, not WHITE))
            )

        # 3. Pions (PST + passés)
        for sq in self.board.pieces(PAWN, WHITE):
            score += PAWN_TABLE_WHITE[sq]
        for sq in self.board.pieces(PAWN, not WHITE):
            score -= PAWN_TABLE_BLACK[sq]

        # 4. Cavaliers PST
        for sq in self.board.pieces(KNIGHT, WHITE):
            score += KNIGHT_TABLE[sq]
        for sq in self.board.pieces(KNIGHT, not WHITE):
            score -= KNIGHT_TABLE[63 - sq]

        # 5. Fous PST (NEW)
        for sq in self.board.pieces(BISHOP, WHITE):
            score += BISHOP_TABLE_WHITE[sq]
        for sq in self.board.pieces(BISHOP, not WHITE):
            score -= BISHOP_TABLE_BLACK[sq]

        # 6. Roi PST selon phase
        total_material = sum(
            PIECE_VALUES[pt] * len(self.board.pieces(pt, True))
            for pt in [QUEEN, ROOK, BISHOP, KNIGHT]
        )

        white_king = self.board.king(WHITE)
        black_king = self.board.king(not WHITE)

        if total_material > 2000:  # middlegame
            score += KING_MG_TABLE_WHITE[white_king]
            score -= KING_MG_TABLE_BLACK[black_king]
        else:  # endgame
            score += KING_EG_TABLE_WHITE[white_king]
            score -= KING_EG_TABLE_BLACK[black_king]

        return score

    # ——————————————————————————————
    #           PIONS PASSÉS
    # ——————————————————————————————
    def _is_passed_pawn(self, square, color):
        file = square % 8
        rank = square // 8

        for cf in [file - 1, file, file + 1]:
            if 0 <= cf < 8:
                if color == WHITE:
                    for r in range(rank + 1, 8):
                        sq = r * 8 + cf
                        piece = self.board.piece_at(sq)
                        if piece and piece.piece_type == PAWN and not self.board.color_at(sq):
                            return False
                else:
                    for r in range(rank - 1, -1, -1):
                        sq = r * 8 + cf
                        piece = self.board.piece_at(sq)
                        if piece and piece.piece_type == PAWN and self.board.color_at(sq):
                            return False
        return True

    # ——————————————————————————————
    #       STRUCTURE DES PIONS
    # ——————————————————————————————
    def _evaluate_pawn_structure(self):
        score = 0

        white_pawn_files = {}
        black_pawn_files = {}

        # comptage blancs
        for sq in self.board.pieces(PAWN, WHITE):
            f = sq % 8
            white_pawn_files[f] = white_pawn_files.get(f, 0) + 1

        # comptage noirs
        for sq in self.board.pieces(PAWN, not WHITE):
            f = sq % 8
            black_pawn_files[f] = black_pawn_files.get(f, 0) + 1

        # pions doublés
        for f, c in white_pawn_files.items():
            if c > 1:
                score -= 20 * (c - 1)
        for f, c in black_pawn_files.items():
            if c > 1:
                score += 20 * (c - 1)

        # pions isolés
        for f in white_pawn_files:
            if f - 1 not in white_pawn_files and f + 1 not in white_pawn_files:
                score -= 15
        for f in black_pawn_files:
            if f - 1 not in black_pawn_files and f + 1 not in black_pawn_files:
                score += 15

        return score

    # ——————————————————————————————
    #        CONTRÔLE DU CENTRE
    # ——————————————————————————————
    def _evaluate_center_control(self):
        score = 0
        center = [27, 28, 35, 36]
        ext_center = [18,19,20,21,26,29,34,37,42,43,44,45]

        for sq in center:
            p = self.board.piece_at(sq)
            if p:
                if self.board.color_at(sq) == WHITE: score += 30
                else: score -= 30

        for sq in ext_center:
            p = self.board.piece_at(sq)
            if p:
                if self.board.color_at(sq) == WHITE: score += 10
                else: score -= 10

        return score

    # ——————————————————————————————
    #       ORDONNANCEMENT DES COUPS
    # ——————————————————————————————
    def _order_moves(self, moves, maximizing):
        scored = []

        for move in moves:
            score = 0

            # Captures prioritaires
            if self.board.is_capture(move):
                target = self.board.piece_at(move.to_square)
                if target:
                    score += PIECE_VALUES.get(target.piece_type, 0) * 10

            # Promotions
            if move.promotion:
                score += 900

            # Menace d'échec
            self.board.push(move)
            if self.board.is_check():
                score += 50
            self.board.pop()

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for s, m in scored]

    # ——————————————————————————————
    #  EXTENSION SI DANS UNE POSITION
    #          CRITIQUE (ÉCHEC)
    # ——————————————————————————————
    def _should_extend_search(self):
        return self.board.is_check()

    # ——————————————————————————————
    #              MINIMAX
    # ——————————————————————————————
    def minimax(self, depth, alpha, beta, maximizing):
        key = (self.board.fen(), depth, maximizing)
        if key in self.transposition_table:
            return self.transposition_table[key]

        if depth == 0 and self._should_extend_search():
            depth = 1

        if depth == 0 or self.board.is_game_over():
            return self.evaluate(), None

        best_move = None
        moves = self._order_moves(self.board.legal_moves, maximizing)

        # ————————————————
        #     MAXIMIZATION
        # ————————————————
        if maximizing:
            maxEval = -10**9
            for move in moves:
                self.board.push(move)
                eval, _ = self.minimax(depth - 1, alpha, beta, False)
                self.board.pop()

                if eval > maxEval:
                    maxEval = eval
                    best_move = move

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            self.transposition_table[key] = (maxEval, best_move)
            return maxEval, best_move

        # ————————————————
        #     MINIMIZATION
        # ————————————————
        else:
            minEval = 10**9
            for move in moves:
                self.board.push(move)
                eval, _ = self.minimax(depth - 1, alpha, beta, True)
                self.board.pop()

                if eval < minEval:
                    minEval = eval
                    best_move = move

                beta = min(beta, eval)
                if beta <= alpha:
                    break

            self.transposition_table[key] = (minEval, best_move)
            return minEval, best_move

    # ——————————————————————————————
    #   COUPS D'OUVERTURE (BOOK)
    # ——————————————————————————————
    def get_opening_move(self, board):
        fen = board.fen()

        if fen in OPENING_BOOK:
            moves = OPENING_BOOK[fen]
            chosen = random.choice(moves)

            try:
                mv = board.push_san(chosen)
                board.pop()
                return chosen
            except:
                return None

        return None

    # ——————————————————————————————
    #           COUP FINAL
    # ——————————————————————————————
    def coup(self, board):
        # Ouverture pendant 8 coups
        if self.opening_moves_played < 8:
            mv = self.get_opening_move(board)
            if mv:
                self.opening_moves_played += 1
                return mv

        # Sinon IA complète
        self.board = board
        maximizing = board.turn == WHITE

        _, best = self.minimax(self.depth, -10**9, 10**9, maximizing)

        if best is None:
            raise ValueError("Aucun coup trouvé !")

        return board.san(best)
