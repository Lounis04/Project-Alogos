import json
import os
import random
import chess

from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK

from PST import (
    PAWN_TABLE_WHITE, PAWN_TABLE_BLACK,
    KNIGHT_TABLE,
    BISHOP_TABLE_WHITE, BISHOP_TABLE_BLACK,
    ROOK_TABLE_WHITE, ROOK_TABLE_BLACK,
    QUEEN_TABLE_WHITE, QUEEN_TABLE_BLACK,
    KING_MG_TABLE_WHITE, KING_MG_TABLE_BLACK,
    KING_EG_TABLE_WHITE, KING_EG_TABLE_BLACK
)

from Ouvertures import OPENING_BOOK


PIECE_VALUES = {
    PAWN: 100,
    KNIGHT: 300,
    BISHOP: 325,
    ROOK: 500,
    QUEEN: 950,
    KING: 20000,
}


class TreeIA:
    def __init__(self, depth=2, transpo_file="coups.json", train_mode=True):
        self.depth = depth
        self.transpo_file = transpo_file
        self.train_mode = train_mode

        # Compteur d’ouverture
        self.opening_moves_played = 0

        # Chargement du JSON
        if os.path.exists(transpo_file):
            try:
                with open(transpo_file, "r") as f:
                    self.transposition_table = json.load(f)
            except:
                self.transposition_table = {}
        else:
            self.transposition_table = {}

    # Sauvegarde conditionnelle (uniquement en mode entraînement)
    def save_transpo(self):
        if not self.train_mode:
            return
        with open(self.transpo_file, "w") as f:
            json.dump(self.transposition_table, f, indent=2)

    # ==============================================================
    #                        ÉVALUATION
    # ==============================================================

    def evaluate(self):

        if self.board.is_checkmate():
            return -100000 if self.board.turn == WHITE else 100000

        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0

        score = 0

        # Matériel
        for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN]:
            score += PIECE_VALUES[pt] * (
                len(self.board.pieces(pt, WHITE))
                - len(self.board.pieces(pt, BLACK))
            )

        # Pions PST
        for sq in self.board.pieces(PAWN, WHITE):
            score += PAWN_TABLE_WHITE[sq]
        for sq in self.board.pieces(PAWN, BLACK):
            score -= PAWN_TABLE_BLACK[sq]

        # Cavaliers PST
        for sq in self.board.pieces(KNIGHT, WHITE):
            score += KNIGHT_TABLE[sq]
        for sq in self.board.pieces(KNIGHT, BLACK):
            score -= KNIGHT_TABLE[sq ^ 56]  # correction du flip

        # Fous PST
        for sq in self.board.pieces(BISHOP, WHITE):
            score += BISHOP_TABLE_WHITE[sq]
        for sq in self.board.pieces(BISHOP, BLACK):
            score -= BISHOP_TABLE_BLACK[sq]

        # Tours PST
        for sq in self.board.pieces(ROOK, WHITE):
            score += ROOK_TABLE_WHITE[sq]
        for sq in self.board.pieces(ROOK, BLACK):
            score -= ROOK_TABLE_BLACK[sq]

        # Reines PST
        for sq in self.board.pieces(QUEEN, WHITE):
            score += QUEEN_TABLE_WHITE[sq]
        for sq in self.board.pieces(QUEEN, BLACK):
            score -= QUEEN_TABLE_BLACK[sq]

        # Roi PST (middlegame / endgame)
        total_material = sum(
            PIECE_VALUES[pt] * len(self.board.pieces(pt, WHITE))
            for pt in [QUEEN, ROOK, BISHOP, KNIGHT]
        )

        wking = self.board.king(WHITE)
        bking = self.board.king(BLACK)

        if total_material > 2000:  # middlegame
            score += KING_MG_TABLE_WHITE[wking]
            score -= KING_MG_TABLE_BLACK[bking]
        else:  # endgame
            score += KING_EG_TABLE_WHITE[wking]
            score -= KING_EG_TABLE_BLACK[bking]

        return score

    # ==============================================================
    #                        ORDRE DES COUPS
    # ==============================================================

    def _order_moves(self, moves, maximizing):
        scored = []

        for move in moves:
            score = 0

            # Captures
            if self.board.is_capture(move):
                target = self.board.piece_at(move.to_square)
                if target:
                    score += PIECE_VALUES[target.piece_type] * 10

            # Promotions
            if move.promotion:
                score += 900

            # Échec
            self.board.push(move)
            if self.board.is_check():
                score += 50
            self.board.pop()

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for s, m in scored]

    # ==============================================================
    #                           MINIMAX
    # ==============================================================

    def minimax(self, depth, alpha, beta, maximizing):
        fen = self.board.fen()

        # 1) Transposition (lecture seule)
        if fen in self.transposition_table:
            entry = self.transposition_table[fen]
            if entry["depth"] >= depth:
                return entry["score"], chess.Move.from_uci(entry["best_move"])

        # Feuille
        if depth == 0 or self.board.is_game_over():
            return self.evaluate(), None

        moves = self._order_moves(self.board.legal_moves, maximizing)
        best_move = None

        if maximizing:
            best_score = -10**9
            for move in moves:
                self.board.push(move)
                eval, _ = self.minimax(depth - 1, alpha, beta, False)
                self.board.pop()

                if eval > best_score:
                    best_score = eval
                    best_move = move

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

        else:
            best_score = 10**9
            for move in moves:
                self.board.push(move)
                eval, _ = self.minimax(depth - 1, alpha, beta, True)
                self.board.pop()

                if eval < best_score:
                    best_score = eval
                    best_move = move

                beta = min(beta, eval)
                if beta <= alpha:
                    break

        # Sauvegarde uniquement en mode entraînement
        if best_move and self.train_mode:
            self.transposition_table[fen] = {
                "best_move": best_move.uci(),
                "score": best_score,
                "depth": depth
            }

        return best_score, best_move

    # ==============================================================
    #                         OUVERTURES
    # ==============================================================

    def get_opening_move(self, board):
        fen = board.fen()
        if fen in OPENING_BOOK:
            moves = OPENING_BOOK[fen]
            return random.choice(moves)
        return None

    # ==============================================================
    #                           COUP
    # ==============================================================

    def coup(self, board):

        # Ouvertures
        if self.opening_moves_played < 8:
            mv = self.get_opening_move(board)
            if mv:
                self.opening_moves_played += 1
                return mv

        # IA complète
        self.board = board
        maximizing = board.turn == WHITE

        _, best = self.minimax(self.depth, -10**9, 10**9, maximizing)

        if best is None:
            raise ValueError("Aucun coup trouvé !")

        # Sauvegarde JSON si entraînement
        self.save_transpo()

        return board.san(best)
