import json
import math
import os
import random
import time
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
    PAWN:   100,
    KNIGHT: 320,
    BISHOP: 330,
    ROOK:   500,
    QUEEN:  900,
    KING:   20000,
}

EXACT      = 0
LOWERBOUND = 1
UPPERBOUND = 2

TT_MAX_SIZE = 500_000

# Futility Pruning
FUTILITY_MARGINS = {1: 320, 2: 500, 3: 900}

# Delta Pruning (Quiescence)
DELTA_MARGIN = 200

# Razoring
RAZORING_MARGIN = 300

# Probcut
PROBCUT_MARGIN = 100

# Singular Extension
SINGULAR_MARGIN = 50

# Contempt : valeur d'une nulle pour le camp qui cherche (points)
CONTEMPT = 25

# Late Move Pruning : nb de coups max à chercher par profondeur
LMP_COUNTS = {1: 5, 2: 10, 3: 18}

# ──────────────────────────────────────────────────────────────────────
#  Masques bitboard pour la détection rapide des pions passés
#  passed_mask_white[sq] = masque des cases devant (rangs sup.) sur
#  la colonne du pion et les colonnes adjacentes.
# ──────────────────────────────────────────────────────────────────────

def _build_passed_masks():
    white_masks = [0] * 64
    black_masks = [0] * 64
    for sq in range(64):
        rank, file = sq // 8, sq % 8
        # Masque blanc : rangs rank+1 … 7, colonnes file-1 … file+1
        for r in range(rank + 1, 8):
            for f in range(max(0, file - 1), min(7, file + 1) + 1):
                white_masks[sq] |= (1 << (r * 8 + f))
        # Masque noir : rangs 0 … rank-1
        for r in range(0, rank):
            for f in range(max(0, file - 1), min(7, file + 1) + 1):
                black_masks[sq] |= (1 << (r * 8 + f))
    return white_masks, black_masks

PASSED_MASK_WHITE, PASSED_MASK_BLACK = _build_passed_masks()


class TreeIA:
    def __init__(self, depth=2, transpo_file="coups.json", train_mode=True):
        self.depth        = depth
        self.transpo_file = transpo_file
        self.train_mode   = train_mode

        self.opening_moves_played = {True: 0, False: 0}

        if os.path.exists(transpo_file):
            try:
                with open(transpo_file, "r") as f:
                    self.transposition_table = json.load(f)
                if not isinstance(self.transposition_table, dict):
                    self.transposition_table = {}
            except Exception:
                self.transposition_table = {}
        else:
            self.transposition_table = {}

        self.killer_moves  = [[None, None] for _ in range(64)]
        self.history       = {}
        # Counter-move heuristic : counter_moves[from_sq][to_sq] = chess.Move
        self.counter_moves = [[None] * 64 for _ in range(64)]
        self.nodes_searched = 0

        self.time_limit         = 5.0
        self._search_start_time = 0.0
        self._time_exceeded     = False

        # Dernier coup joué par l'adversaire (pour counter-move)
        self._last_move = None

    # ------------------------------------------------------------------
    #   Clé Zobrist — string JSON-safe
    # ------------------------------------------------------------------

    @staticmethod
    def _zobrist(board):
        if hasattr(board, '_transposition_key'):
            return str(board._transposition_key())
        try:
            import chess.polyglot
            return str(chess.polyglot.zobrist_hash(board))
        except Exception:
            return board.fen()

    # ------------------------------------------------------------------
    #   Sauvegarde sur disque
    # ------------------------------------------------------------------

    def save_transpo(self):
        if not self.train_mode:
            return
        try:
            tmp = self.transpo_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.transposition_table, f, indent=2)
            os.replace(tmp, self.transpo_file)
        except Exception as e:
            print(f"[save_transpo] Erreur : {e}")

    # ==================================================================
    #                           ÉVALUATION
    # ==================================================================

    def evaluate(self):
        """
        Évaluation statique TOUJOURS du point de vue des BLANCS.
        Score positif = bon pour les Blancs, négatif = bon pour les Noirs.
        """
        if self.board.is_checkmate():
            return -100_000 if self.board.turn == WHITE else 100_000
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0

        score = 0

        for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN]:
            score += PIECE_VALUES[pt] * (
                len(self.board.pieces(pt, WHITE))
                - len(self.board.pieces(pt, BLACK))
            )

        for sq in self.board.pieces(PAWN, WHITE):   score += PAWN_TABLE_WHITE[sq]
        for sq in self.board.pieces(PAWN, BLACK):   score -= PAWN_TABLE_BLACK[sq]
        for sq in self.board.pieces(KNIGHT, WHITE): score += KNIGHT_TABLE[sq]
        for sq in self.board.pieces(KNIGHT, BLACK): score -= KNIGHT_TABLE[sq ^ 56]
        for sq in self.board.pieces(BISHOP, WHITE): score += BISHOP_TABLE_WHITE[sq]
        for sq in self.board.pieces(BISHOP, BLACK): score -= BISHOP_TABLE_BLACK[sq]

        if len(self.board.pieces(BISHOP, WHITE)) >= 2: score += 50
        if len(self.board.pieces(BISHOP, BLACK)) >= 2: score -= 50

        for sq in self.board.pieces(ROOK, WHITE):  score += ROOK_TABLE_WHITE[sq]
        for sq in self.board.pieces(ROOK, BLACK):  score -= ROOK_TABLE_BLACK[sq]
        for sq in self.board.pieces(QUEEN, WHITE): score += QUEEN_TABLE_WHITE[sq]
        for sq in self.board.pieces(QUEEN, BLACK): score -= QUEEN_TABLE_BLACK[sq]

        wking = self.board.king(WHITE)
        bking = self.board.king(BLACK)

        total_material = sum(
            PIECE_VALUES[pt] * (
                len(self.board.pieces(pt, WHITE)) + len(self.board.pieces(pt, BLACK))
            )
            for pt in [QUEEN, ROOK, BISHOP, KNIGHT]
        )
        is_endgame = total_material < 2600

        if is_endgame:
            score += KING_EG_TABLE_WHITE[wking]
            score -= KING_EG_TABLE_BLACK[bking]
        else:
            score += KING_MG_TABLE_WHITE[wking]
            score -= KING_MG_TABLE_BLACK[bking]

        score += self._evaluate_rook_placement()
        score += self._evaluate_pawn_structure()
        score += self._evaluate_center_control()
        score += self._evaluate_mobility_fast()
        score += self._evaluate_king_safety()

        if is_endgame:
            score += self._evaluate_endgame_king_activity()
        else:
            score += self._evaluate_castling_rights()

        score += self._evaluate_tactics()
        score += self._evaluate_king_exposure()
        score += self._evaluate_pins()

        # Bonus de simplification
        material_balance = sum(
            PIECE_VALUES[pt] * (
                len(self.board.pieces(pt, WHITE)) - len(self.board.pieces(pt, BLACK))
            )
            for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN]
        )
        if abs(material_balance) > 200:
            simplification_bonus = max(0, (6400 - total_material) // 200)
            if material_balance > 200:
                score += simplification_bonus
            else:
                score -= simplification_bonus

        return score

    # ------------------------------------------------------------------

    def _evaluate_castling_rights(self):
        score = 0
        if self.board.has_kingside_castling_rights(WHITE):  score += 15
        if self.board.has_queenside_castling_rights(WHITE): score += 10
        if self.board.has_kingside_castling_rights(BLACK):  score -= 15
        if self.board.has_queenside_castling_rights(BLACK): score -= 10
        wk = self.board.king(WHITE)
        bk = self.board.king(BLACK)
        if wk in [6, 2]:    score += 30
        if bk in [62, 58]:  score -= 30
        return score

    def _evaluate_endgame_king_activity(self):
        score = 0
        wk = self.board.king(WHITE)
        bk = self.board.king(BLACK)

        center = 3.5
        wd = abs(wk % 8 - center) + abs(wk // 8 - center)
        bd = abs(bk % 8 - center) + abs(bk // 8 - center)
        score += int((bd - wd) * 15)

        wp = sum(len(self.board.pieces(pt, WHITE)) for pt in [KNIGHT, BISHOP, ROOK, QUEEN])
        bp = sum(len(self.board.pieces(pt, BLACK)) for pt in [KNIGHT, BISHOP, ROOK, QUEEN])

        if wp == 0 and bp == 0:
            be = min(bk % 8, 7 - bk % 8, bk // 8, 7 - bk // 8)
            we = min(wk % 8, 7 - wk % 8, wk // 8, 7 - wk // 8)
            score -= be * 5
            score += we * 5

        wr, wf = wk // 8, wk % 8
        br, bf = bk // 8, bk % 8

        direct_opp = ((wr == br and abs(wf - bf) == 2)
                      or (wf == bf and abs(wr - br) == 2))
        dist_opp = (abs(wf - bf) % 2 == 0
                    and abs(wr - br) % 2 == 0
                    and abs(wf - bf) + abs(wr - br) > 2)

        if direct_opp:
            if self.board.turn == BLACK:
                score += 30
            else:
                score -= 30
        elif dist_opp:
            if self.board.turn == BLACK:
                score += 15
            else:
                score -= 15

        # Carré du pion passé (via bitboards)
        black_pawn_bb = int(self.board.pieces(PAWN, BLACK))
        white_pawn_bb = int(self.board.pieces(PAWN, WHITE))

        for sq in self.board.pieces(PAWN, WHITE):
            if PASSED_MASK_WHITE[sq] & black_pawn_bb:
                continue  # pas passé
            rank, file = sq // 8, sq % 8
            steps     = 7 - rank
            bk_r, bk_f = bk // 8, bk % 8
            king_dist = max(abs(bk_r - 7), abs(bk_f - file))
            extra     = 0 if self.board.turn == WHITE else 1
            if king_dist > steps + extra:
                score += 200

        for sq in self.board.pieces(PAWN, BLACK):
            if PASSED_MASK_BLACK[sq] & white_pawn_bb:
                continue
            rank, file = sq // 8, sq % 8
            steps     = rank
            wk_r, wk_f = wk // 8, wk % 8
            king_dist = max(abs(wk_r - 0), abs(wk_f - file))
            extra     = 0 if self.board.turn == BLACK else 1
            if king_dist > steps + extra:
                score -= 200

        # Cases clés devant les pions passés avancés
        for sq in self.board.pieces(PAWN, WHITE):
            file, rank = sq % 8, sq // 8
            if rank < 4:
                continue
            if PASSED_MASK_WHITE[sq] & black_pawn_bb:
                continue
            key_rank = min(7, rank + 2)
            for kf in [file - 1, file, file + 1]:
                if 0 <= kf < 8 and wk == key_rank * 8 + kf:
                    score += 50

        for sq in self.board.pieces(PAWN, BLACK):
            file, rank = sq % 8, sq // 8
            if rank > 3:
                continue
            if PASSED_MASK_BLACK[sq] & white_pawn_bb:
                continue
            key_rank = max(0, rank - 2)
            for kf in [file - 1, file, file + 1]:
                if 0 <= kf < 8 and bk == key_rank * 8 + kf:
                    score -= 50

        return score

    def _evaluate_rook_placement(self):
        score = 0
        for rook_sq in self.board.pieces(ROOK, WHITE):
            file = rook_sq % 8
            is_open = is_semi = True
            for rank in range(8):
                p = self.board.piece_at(rank * 8 + file)
                if p and p.piece_type == PAWN:
                    is_open = False
                    if p.color == WHITE: is_semi = False
            score += 40 if is_open else (20 if is_semi else 0)

        for rook_sq in self.board.pieces(ROOK, BLACK):
            file = rook_sq % 8
            is_open = is_semi = True
            for rank in range(8):
                p = self.board.piece_at(rank * 8 + file)
                if p and p.piece_type == PAWN:
                    is_open = False
                    if p.color == BLACK: is_semi = False
            score -= 40 if is_open else (20 if is_semi else 0)

        return score

    def _evaluate_pawn_structure(self):
        score = 0
        white_pawns  = list(self.board.pieces(PAWN, WHITE))
        black_pawns  = list(self.board.pieces(PAWN, BLACK))
        white_set    = set(white_pawns)
        black_set    = set(black_pawns)
        white_pawn_bb = int(self.board.pieces(PAWN, WHITE))
        black_pawn_bb = int(self.board.pieces(PAWN, BLACK))

        # Pions doublés
        wfiles = [sq % 8 for sq in white_pawns]
        bfiles = [sq % 8 for sq in black_pawns]
        for file in range(8):
            wc, bc = wfiles.count(file), bfiles.count(file)
            if wc > 1: score -= 15 * (wc - 1)
            if bc > 1: score += 15 * (bc - 1)

        # Pions isolés
        for sq in white_pawns:
            file = sq % 8
            if not any(p % 8 in (file - 1, file + 1)
                       for p in white_pawns if 0 <= p % 8 < 8):
                score -= 20
        for sq in black_pawns:
            file = sq % 8
            if not any(p % 8 in (file - 1, file + 1)
                       for p in black_pawns if 0 <= p % 8 < 8):
                score += 20

        # Pions passés (via bitboards rapides)
        for sq in white_pawns:
            if not (PASSED_MASK_WHITE[sq] & black_pawn_bb):
                rank = sq // 8
                bonus = 30 + rank * 10

                # ── NOUVEAU : pion passé soutenu ─────────────────────
                file = sq % 8
                for f in [file - 1, file + 1]:
                    if 0 <= f < 8 and (rank - 1) * 8 + f in white_set:
                        bonus += 40
                        break

                score += bonus

        for sq in black_pawns:
            if not (PASSED_MASK_BLACK[sq] & white_pawn_bb):
                rank = sq // 8
                bonus = 30 + (7 - rank) * 10

                # ── NOUVEAU : pion passé soutenu ─────────────────────
                file = sq % 8
                for f in [file - 1, file + 1]:
                    if 0 <= f < 8 and (rank + 1) * 8 + f in black_set:
                        bonus += 40
                        break

                score -= bonus

        # Pions arriérés – Blancs
        for sq in white_pawns:
            file, rank = sq % 8, sq // 8
            blocked = False
            for r in range(rank + 1, 8):
                p = self.board.piece_at(r * 8 + file)
                if p and p.piece_type == PAWN:
                    if p.color == BLACK: blocked = True
                    break
            if not blocked: continue
            supported = False
            for nf in [file - 1, file + 1]:
                if 0 <= nf < 8:
                    for r in range(rank - 1, 0, -1):
                        p = self.board.piece_at(r * 8 + nf)
                        if p and p.piece_type == PAWN and p.color == WHITE:
                            supported = True
                            break
                if supported: break
            if not supported: score -= 15

        # Pions arriérés – Noirs
        for sq in black_pawns:
            file, rank = sq % 8, sq // 8
            blocked = False
            for r in range(rank - 1, -1, -1):
                p = self.board.piece_at(r * 8 + file)
                if p and p.piece_type == PAWN:
                    if p.color == WHITE: blocked = True
                    break
            if not blocked: continue
            supported = False
            for nf in [file - 1, file + 1]:
                if 0 <= nf < 8:
                    for r in range(rank + 1, 7):
                        p = self.board.piece_at(r * 8 + nf)
                        if p and p.piece_type == PAWN and p.color == BLACK:
                            supported = True
                            break
                if supported: break
            if not supported: score += 15

        # Pions connectés – Blancs
        for sq in white_pawns:
            rank, file = sq // 8, sq % 8
            if rank > 0:
                for f in [file - 1, file + 1]:
                    if 0 <= f < 8 and (rank - 1) * 8 + f in white_set:
                        score += 10 + rank * 2
                        break

        # Pions connectés – Noirs
        for sq in black_pawns:
            rank, file = sq // 8, sq % 8
            if rank < 7:
                for f in [file - 1, file + 1]:
                    if 0 <= f < 8 and (rank + 1) * 8 + f in black_set:
                        score -= 10 + (7 - rank) * 2
                        break

        # Pions passés candidats – Blancs
        for sq in white_pawns:
            file, rank = sq % 8, sq // 8
            if not (PASSED_MASK_WHITE[sq] & black_pawn_bb):
                continue  # déjà passé
            is_candidate = True
            has_blocker  = False
            for f in [file - 1, file, file + 1]:
                if 0 <= f < 8:
                    w_ahead = sum(1 for p_sq in white_pawns
                                  if p_sq % 8 == f and p_sq // 8 > rank)
                    b_ahead = sum(1 for p_sq in black_pawns
                                  if p_sq % 8 == f and p_sq // 8 > rank)
                    if b_ahead > 0:
                        has_blocker = True
                    if b_ahead > w_ahead:
                        is_candidate = False
                        break
            if is_candidate and has_blocker:
                score += 10 + rank * 3

        # Pions passés candidats – Noirs
        for sq in black_pawns:
            file, rank = sq % 8, sq // 8
            if not (PASSED_MASK_BLACK[sq] & white_pawn_bb):
                continue
            is_candidate = True
            has_blocker  = False
            for f in [file - 1, file, file + 1]:
                if 0 <= f < 8:
                    w_behind = sum(1 for p_sq in white_pawns
                                   if p_sq % 8 == f and p_sq // 8 < rank)
                    b_behind = sum(1 for p_sq in black_pawns
                                   if p_sq % 8 == f and p_sq // 8 < rank)
                    if w_behind > 0:
                        has_blocker = True
                    if w_behind > b_behind:
                        is_candidate = False
                        break
            if is_candidate and has_blocker:
                score -= 10 + (7 - rank) * 3

        return score

    def _evaluate_center_control(self):
        score  = 0
        center   = [27, 28, 35, 36]
        extended = [18, 19, 20, 21, 26, 29, 34, 37, 42, 43, 44, 45]
        for sq in center:
            if self.board.is_attacked_by(WHITE, sq): score += 10
            if self.board.is_attacked_by(BLACK, sq): score -= 10
            p = self.board.piece_at(sq)
            if p: score += 20 if p.color == WHITE else -20
        for sq in extended:
            if self.board.is_attacked_by(WHITE, sq): score += 3
            if self.board.is_attacked_by(BLACK, sq): score -= 3
        return score

    def _evaluate_mobility_fast(self):
        wa = sum(
            len(list(self.board.attacks(sq)))
            for pt in [QUEEN, ROOK, BISHOP, KNIGHT]
            for sq in self.board.pieces(pt, WHITE)
        )
        ba = sum(
            len(list(self.board.attacks(sq)))
            for pt in [QUEEN, ROOK, BISHOP, KNIGHT]
            for sq in self.board.pieces(pt, BLACK)
        )
        return (wa - ba) * 3

    def _evaluate_king_safety(self):
        score = 0

        wk = self.board.king(WHITE)
        if wk is not None:
            attackers = 0
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[wk]):
                p = self.board.piece_at(sq)
                if p and p.color == BLACK: attackers += 1
            file, rank = wk % 8, wk // 8
            shield_ranks = [rank + 1, rank + 2] if rank < 6 else [rank - 1]
            pawn_shield = 0
            for r in shield_ranks:
                if 0 <= r < 8:
                    for f in [file - 1, file, file + 1]:
                        if 0 <= f < 8:
                            p = self.board.piece_at(r * 8 + f)
                            if p and p.piece_type == PAWN and p.color == WHITE:
                                pawn_shield += 1
            weak = sum(
                1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[wk])
                if not self.board.is_attacked_by(WHITE, sq)
            )
            score += -25 * attackers + 12 * pawn_shield - 8 * weak

        bk = self.board.king(BLACK)
        if bk is not None:
            attackers = 0
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[bk]):
                p = self.board.piece_at(sq)
                if p and p.color == WHITE: attackers += 1
            file, rank = bk % 8, bk // 8
            shield_ranks = [rank - 1, rank - 2] if rank > 1 else [rank + 1]
            pawn_shield = 0
            for r in shield_ranks:
                if 0 <= r < 8:
                    for f in [file - 1, file, file + 1]:
                        if 0 <= f < 8:
                            p = self.board.piece_at(r * 8 + f)
                            if p and p.piece_type == PAWN and p.color == BLACK:
                                pawn_shield += 1
            weak = sum(
                1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[bk])
                if not self.board.is_attacked_by(BLACK, sq)
            )
            score -= -25 * attackers + 12 * pawn_shield - 8 * weak

        return score

    def _evaluate_tactics(self):
        score = 0

        for color, sign in [(WHITE, 1), (BLACK, -1)]:
            opponent = not color
            for pt in [QUEEN, ROOK, BISHOP, KNIGHT]:
                for sq in self.board.pieces(pt, color):
                    if not self.board.is_attacked_by(color, sq):
                        val = PIECE_VALUES[pt]
                        if self.board.is_attacked_by(opponent, sq):
                            score -= sign * (val // 2)
                        else:
                            score -= sign * (val // 8)

        for sq in self.board.pieces(ROOK, WHITE):
            if sq // 8 == 6: score += 50
        for sq in self.board.pieces(ROOK, BLACK):
            if sq // 8 == 1: score -= 50

        for sq in self.board.pieces(KNIGHT, WHITE):
            rank, file = sq // 8, sq % 8
            if rank >= 4:
                threatened = False
                if rank + 1 < 8:
                    for f in [file - 1, file + 1]:
                        if 0 <= f < 8:
                            p = self.board.piece_at((rank + 1) * 8 + f)
                            if p and p.piece_type == PAWN and p.color == BLACK:
                                threatened = True
                                break
                if not threatened: score += 20

        for sq in self.board.pieces(KNIGHT, BLACK):
            rank, file = sq // 8, sq % 8
            if rank <= 3:
                threatened = False
                if rank - 1 >= 0:
                    for f in [file - 1, file + 1]:
                        if 0 <= f < 8:
                            p = self.board.piece_at((rank - 1) * 8 + f)
                            if p and p.piece_type == PAWN and p.color == WHITE:
                                threatened = True
                                break
                if not threatened: score -= 20

        for sq in self.board.pieces(BISHOP, WHITE):
            if len(list(self.board.attacks(sq))) >= 6: score += 15
        for sq in self.board.pieces(BISHOP, BLACK):
            if len(list(self.board.attacks(sq))) >= 6: score -= 15

        return score

    def _evaluate_king_exposure(self):
        score = 0
        white_pawns = list(self.board.pieces(PAWN, WHITE))
        black_pawns = list(self.board.pieces(PAWN, BLACK))

        bk = self.board.king(BLACK)
        if bk is not None:
            bk_file = bk % 8
            if not any(sq % 8 == bk_file for sq in black_pawns):
                heavy = sum(
                    1 for pt in [ROOK, QUEEN]
                    for sq in self.board.pieces(pt, WHITE)
                    if sq % 8 == bk_file
                )
                if heavy > 0:
                    score += 25 * heavy

        wk = self.board.king(WHITE)
        if wk is not None:
            wk_file = wk % 8
            if not any(sq % 8 == wk_file for sq in white_pawns):
                heavy = sum(
                    1 for pt in [ROOK, QUEEN]
                    for sq in self.board.pieces(pt, BLACK)
                    if sq % 8 == wk_file
                )
                if heavy > 0:
                    score -= 25 * heavy

        return score

    def _evaluate_pins(self):
        score = 0
        for pt in [KNIGHT, BISHOP, ROOK, QUEEN]:
            for sq in self.board.pieces(pt, WHITE):
                if self.board.is_pinned(WHITE, sq):
                    score -= PIECE_VALUES[pt] // 8
            for sq in self.board.pieces(pt, BLACK):
                if self.board.is_pinned(BLACK, sq):
                    score += PIECE_VALUES[pt] // 8
        return score

    # ==================================================================
    #   SEE récursif complet
    # ==================================================================

    def see(self, to_sq, attacker_color):
        target = self.board.piece_at(to_sq)
        if target is None:
            return 0

        def collect_sorted(color):
            vals = []
            for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
                for sq in self.board.pieces(pt, color):
                    if to_sq in self.board.attacks(sq):
                        vals.append(PIECE_VALUES.get(pt, 20000))
            vals.sort()
            return vals

        atk  = collect_sorted(attacker_color)
        if not atk:
            return 0

        def_  = collect_sorted(not attacker_color)
        gain  = []
        cap   = PIECE_VALUES[target.piece_type]
        sides = [atk, def_]
        side  = 0

        while sides[side]:
            pv = sides[side].pop(0)
            gain.append(cap)
            cap   = pv
            side ^= 1

        if not gain:
            return 0

        for d in range(len(gain) - 1, 0, -1):
            gain[d - 1] = gain[d - 1] - max(0, gain[d])

        return gain[0]

    # ==================================================================
    #   Ordonnancement des coups
    # ==================================================================

    def _order_moves(self, moves, depth, prev_move=None):
        scored  = []
        zkey    = self._zobrist(self.board)
        tt_move = None
        if zkey in self.transposition_table:
            try:
                tt_move = chess.Move.from_uci(
                    self.transposition_table[zkey]["best_move"])
            except Exception:
                pass

        # Counter-move du coup précédent adversaire
        counter_move = None
        if prev_move is not None:
            counter_move = self.counter_moves[prev_move.from_square][prev_move.to_square]

        for move in moves:
            score = 0

            if tt_move and move == tt_move:
                scored.append((1_000_000, move))
                continue

            if self.board.is_capture(move):
                target   = self.board.piece_at(move.to_square)
                attacker = self.board.piece_at(move.from_square)
                if target and attacker:
                    ss = self.see(move.to_square, self.board.turn)
                    score += (500_000 + ss) if ss >= 0 else ss

            if move.promotion:
                score += 900 if move.promotion == QUEEN else 300

            if depth < 64:
                if move == self.killer_moves[depth][0]:   score += 1_000
                elif move == self.killer_moves[depth][1]: score += 800

            # ── Counter-move heuristic ──────────────────────────────
            if counter_move is not None and move == counter_move:
                score += 600

            mk = move.uci()
            if mk in self.history:
                score += self.history[mk]

            self.board.push(move)
            if self.board.is_check(): score += 50
            self.board.pop()

            if move.to_square in (27, 28, 35, 36): score += 10

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _update_killers(self, move, depth):
        if depth >= 64:
            return
        if self.killer_moves[depth][0] != move:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move

    def _update_history(self, move, depth):
        key = move.uci()
        self.history[key] = self.history.get(key, 0) + depth * depth

    def _update_counter_move(self, prev_move, move):
        """Mémorise 'move' comme meilleure réponse à 'prev_move'."""
        if prev_move is not None:
            self.counter_moves[prev_move.from_square][prev_move.to_square] = move

    # ==================================================================
    #   Quiescence Search + Delta Pruning
    # ==================================================================

    def quiescence(self, alpha, beta):
        stand_pat = self.evaluate()
        if self.board.turn == BLACK:
            stand_pat = -stand_pat

        if stand_pat >= beta: return beta
        if stand_pat > alpha: alpha = stand_pat

        capture_moves = [m for m in self.board.legal_moves
                         if self.board.is_capture(m)]
        capture_moves.sort(
            key=lambda m: PIECE_VALUES.get(
                self.board.piece_at(m.to_square).piece_type, 0
            ) if self.board.piece_at(m.to_square) else 0,
            reverse=True
        )

        for move in capture_moves:
            target = self.board.piece_at(move.to_square)
            if target:
                gain = PIECE_VALUES.get(target.piece_type, 0)
                if move.promotion == QUEEN:
                    gain += PIECE_VALUES[QUEEN] - PIECE_VALUES[PAWN]
                if stand_pat + gain + DELTA_MARGIN <= alpha:
                    continue

            self.board.push(move)
            score = -self.quiescence(-beta, -alpha)
            self.board.pop()

            if score >= beta: return beta
            if score > alpha: alpha = score

        return alpha

    # ==================================================================
    #   Null Move Pruning
    # ==================================================================

    def _try_null_move(self, depth, beta, ply):
        if depth < 3 or self.board.is_check():
            return None
        non_pawn = sum(
            len(self.board.pieces(pt, self.board.turn))
            for pt in [KNIGHT, BISHOP, ROOK, QUEEN]
        )
        if non_pawn == 0:
            return None
        self.board.push(chess.Move.null())
        score = -self.negamax(depth - 1 - 2, -beta, -beta + 1, ply + 1, None)
        self.board.pop()
        return beta if score >= beta else None

    # ==================================================================
    #   Negamax principal
    # ==================================================================

    def negamax(self, depth, alpha, beta, ply, prev_move=None):
        self.nodes_searched += 1

        if self.board.is_repetition(2):
            # ── Contempt factor ──────────────────────────────────────
            # Si on est en avantage, une nulle vaut moins que 0
            contempt_sign = 1 if self.board.turn == WHITE else -1
            return -CONTEMPT * contempt_sign

        # Règle des 50 coups
        if self.board.halfmove_clock >= 100:
            return 0

        # Table de transposition
        zkey       = self._zobrist(self.board)
        alpha_orig = alpha

        if zkey in self.transposition_table:
            entry = self.transposition_table[zkey]
            if entry["depth"] >= depth:
                flag  = entry.get("flag", EXACT)
                score = entry["score"]
                if flag == EXACT:
                    return score
                elif flag == LOWERBOUND:
                    alpha = max(alpha, score)
                elif flag == UPPERBOUND:
                    beta  = min(beta,  score)
                if alpha >= beta:
                    return score

        in_check = self.board.is_check()

        if in_check and ply < 2 * self.depth + 4:
            depth += 1

        if depth <= 0:
            if self.board.is_game_over():
                s = self.evaluate()
                return s if self.board.turn == WHITE else -s
            return self.quiescence(alpha, beta)

        if self.board.is_game_over():
            if self.board.is_checkmate():
                return -100_000 + ply
            return 0

        # Razoring
        if (depth == 1
                and not in_check
                and abs(alpha) < 90_000):
            se = self.evaluate()
            if self.board.turn == BLACK:
                se = -se
            if se < alpha - RAZORING_MARGIN:
                return self.quiescence(alpha, beta)

        # Null Move Pruning
        if not in_check and depth >= 3:
            nr = self._try_null_move(depth, beta, ply)
            if nr is not None:
                return nr

        moves = list(self.board.legal_moves)
        if not moves:
            return 0
        moves = self._order_moves(moves, ply, prev_move)

        # Internal Iterative Deepening (IID)
        if (depth >= 4
                and not in_check
                and zkey not in self.transposition_table):
            self.negamax(depth - 2, alpha, beta, ply, prev_move)
            moves = self._order_moves(moves, ply, prev_move)

        # Probcut
        if (depth >= 5
                and not in_check
                and abs(beta) < 90_000):
            pc_beta      = beta + PROBCUT_MARGIN
            pc_static    = self.evaluate()
            if self.board.turn == BLACK:
                pc_static = -pc_static
            pc_threshold = pc_beta - pc_static
            pc_moves     = [m for m in moves
                            if self.board.is_capture(m)
                            and self.see(m.to_square, self.board.turn) >= pc_threshold]
            for m in pc_moves[:3]:
                self.board.push(m)
                pc_score = -self.negamax(max(1, depth - 4), -pc_beta, -pc_beta + 1, ply + 1, m)
                self.board.pop()
                if pc_score >= pc_beta:
                    return pc_beta

        # Singular Extension
        singular_move = None
        if (depth >= 4
                and not in_check
                and zkey in self.transposition_table):
            tt_e = self.transposition_table[zkey]
            if (tt_e.get("depth", 0) >= depth - 3
                    and tt_e.get("flag", EXACT) != UPPERBOUND
                    and abs(tt_e.get("score", 0)) < 90_000):
                try:
                    cand = chess.Move.from_uci(tt_e["best_move"])
                    if cand in moves:
                        s_beta  = tt_e["score"] - SINGULAR_MARGIN
                        s_depth = min(depth // 2, 3)
                        s_fails = False
                        checked = 0
                        for m in moves:
                            if m == cand or checked >= 6:
                                continue
                            checked += 1
                            self.board.push(m)
                            s_val = -self.negamax(s_depth, -s_beta, -(s_beta - 1), ply + 1, cand)
                            self.board.pop()
                            if s_val >= s_beta:
                                s_fails = True
                                break
                        if not s_fails:
                            singular_move = cand
                except Exception:
                    pass

        # Futility Pruning
        futility_pruning = False
        if (depth in FUTILITY_MARGINS
                and not in_check
                and abs(alpha) < 90_000
                and abs(beta)  < 90_000):
            fe = self.evaluate()
            if self.board.turn == BLACK:
                fe = -fe
            if fe + FUTILITY_MARGINS[depth] <= alpha:
                futility_pruning = True

        # Boucle principale
        best_move      = moves[0]
        best_score     = -10**9
        moves_searched = 0
        quiet_count    = 0   # Pour LMP

        for move in moves:
            is_capture   = self.board.is_capture(move)
            is_promotion = bool(move.promotion)
            is_quiet     = not is_capture and not is_promotion

            # Futility : ignorer les coups calmes sauf le premier
            if futility_pruning and is_quiet and moves_searched > 0:
                continue

            # ── Late Move Pruning (LMP) ──────────────────────────────
            # À faible profondeur, on coupe les coups calmes tardifs
            if (depth in LMP_COUNTS
                    and is_quiet
                    and not in_check
                    and moves_searched > 0):
                if quiet_count >= LMP_COUNTS[depth]:
                    continue

            if is_quiet:
                quiet_count += 1

            self.board.push(move)

            if moves_searched == 0:
                ext = 1 if (singular_move is not None and move == singular_move) else 0
                score = -self.negamax(depth - 1 + ext, -beta, -alpha, ply + 1, move)
            else:
                reduction = 0
                if (depth >= 3 and moves_searched >= 4
                        and not in_check
                        and not is_capture
                        and not self.board.is_check()):
                    reduction = max(1, int(
                        math.log(depth) * math.log(moves_searched) / 1.5))

                score = -self.negamax(
                    depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, move)

                if score > alpha and score < beta:
                    if reduction > 0:
                        score = -self.negamax(
                            depth - 1, -alpha - 1, -alpha, ply + 1, move)
                    if score > alpha:
                        score = -self.negamax(depth - 1, -beta, -alpha, ply + 1, move)

            self.board.pop()
            moves_searched += 1

            if score > best_score:
                best_score = score
                best_move  = move

            alpha = max(alpha, score)

            if alpha >= beta:
                if not is_capture:
                    self._update_killers(move, ply)
                    self._update_history(move, depth)
                    # ── Counter-move update ──────────────────────────
                    if prev_move is not None:
                        self._update_counter_move(prev_move, move)
                break

        # Sauvegarde TT
        flag = EXACT
        if best_score <= alpha_orig: flag = UPPERBOUND
        elif best_score >= beta:     flag = LOWERBOUND

        if len(self.transposition_table) >= TT_MAX_SIZE:
            to_delete = sorted(
                self.transposition_table.keys(),
                key=lambda k: self.transposition_table[k].get("depth", 0)
            )[:TT_MAX_SIZE // 10]
            for k in to_delete:
                del self.transposition_table[k]

        self.transposition_table[zkey] = {
            "best_move": best_move.uci(),
            "score":     best_score,
            "depth":     depth,
            "flag":      flag,
        }

        return best_score

    # ==================================================================
    #   Livre d'ouvertures
    # ==================================================================

    def get_opening_move(self, board):
        fen = board.fen()
        if fen in OPENING_BOOK:
            return random.choice(OPENING_BOOK[fen])
        return None

    # ==================================================================
    #   Promotion intelligente (évite le pat)
    # ==================================================================

    def _smart_promotion(self, board, move):
        if move.promotion != QUEEN:
            return move
        board.push(move)
        is_pat = board.is_stalemate()
        board.pop()
        if not is_pat:
            return move
        for piece in [ROOK, KNIGHT, BISHOP]:
            alt = chess.Move(move.from_square, move.to_square, promotion=piece)
            if alt not in board.legal_moves: continue
            board.push(alt)
            still_pat = board.is_stalemate()
            board.pop()
            if not still_pat: return alt
        return move

    # ==================================================================
    #   Point d'entrée principal
    # ==================================================================

    def coup(self, board):
        # Livre d'ouvertures
        color = board.turn
        if self.opening_moves_played[color] < 12:
            mv = self.get_opening_move(board)
            if mv:
                self.opening_moves_played[color] += 1
                if isinstance(mv, str):
                    mv = board.parse_san(mv)
                return mv

        # ── Détection de mat en 1 ─────────────────────────────────────
        # CORRECTION : retourne chess.Move (pas une SAN string)
        for mv in board.legal_moves:
            board.push(mv)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return mv  # ← chess.Move cohérent avec le reste

        # Initialisation
        self.board              = board
        self.killer_moves       = [[None, None] for _ in range(64)]
        self.nodes_searched     = 0
        self._search_start_time = time.time()
        self._time_exceeded     = False

        # History Aging
        for key in self.history:
            self.history[key] //= 2

        best_move  = None
        prev_score = 0

        # Iterative Deepening + Aspiration Windows
        for d in range(1, self.depth + 1):
            self._time_exceeded = False

            if d >= 2 and best_move is not None:
                delta = 50
                a = prev_score - delta
                b = prev_score + delta

                while True:
                    self._time_exceeded = False
                    score, move = self.negamax_root(d, a, b)

                    if self._time_exceeded:
                        break

                    if score <= a:
                        a      = max(-10**9, a - delta)
                        delta  = min(delta * 2, 500)   # ← progressif, pas *3
                    elif score >= b:
                        b      = min(10**9, b + delta)
                        delta  = min(delta * 2, 500)
                    else:
                        break

                    if delta > 2_000:
                        self._time_exceeded = False
                        score, move = self.negamax_root(d)
                        break
            else:
                score, move = self.negamax_root(d)

            if not self._time_exceeded:
                if move is not None:
                    best_move = move
                prev_score = score
                if abs(score) > 90_000:
                    break
            else:
                break

            if time.time() - self._search_start_time > self.time_limit * 0.85:
                break

        # Fallback
        if best_move is None:
            moves = list(board.legal_moves)
            if moves:
                ms = []
                for mv in moves:
                    sc = 0
                    if board.is_capture(mv):
                        t = board.piece_at(mv.to_square)
                        if t: sc = PIECE_VALUES[t.piece_type]
                    ms.append((sc, mv))
                ms.sort(key=lambda x: x[0], reverse=True)
                bv = ms[0][0]
                best_move = random.choice([m for s, m in ms if s == bv])
            else:
                raise ValueError("Aucun coup légal trouvé !")

        best_move = self._smart_promotion(board, best_move)
        return best_move  # ← toujours chess.Move

    # ==================================================================
    #   Negamax racine
    # ==================================================================

    def negamax_root(self, depth, alpha=-10**9, beta=10**9):
        best_score = -10**9
        best_move  = None

        moves = list(self.board.legal_moves)
        if not moves:
            return 0, None

        moves = self._order_moves(moves, 0, self._last_move)

        for move in moves:
            self.board.push(move)
            score = -self.negamax(depth - 1, -beta, -alpha, 1, move)
            self.board.pop()

            if score > best_score:
                best_score = score
                best_move  = move

            alpha = max(alpha, score)

            if time.time() - self._search_start_time > self.time_limit:
                self._time_exceeded = True
                break

        return best_score, best_move