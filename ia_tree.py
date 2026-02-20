import json
import math
import os
import random
import time
import chess

import chess.polyglot

from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK

from PST import (
    PAWN_TABLE_WHITE, PAWN_TABLE_BLACK,
    KNIGHT_TABLE,
    BISHOP_TABLE_WHITE, BISHOP_TABLE_BLACK,
    ROOK_TABLE_WHITE, ROOK_TABLE_BLACK,
    QUEEN_TABLE_WHITE, QUEEN_TABLE_BLACK,
    KING_MG_TABLE_WHITE, KING_MG_TABLE_BLACK,
    KING_EG_TABLE_WHITE, KING_EG_TABLE_BLACK,
)

from Ouvertures import OPENING_BOOK

# ── Valeurs des pièces ───────────────────────────────────────────────
PIECE_VALUES = {PAWN: 100, KNIGHT: 320, BISHOP: 330,
                ROOK: 500,  QUEEN:  900, KING:  20000}

EXACT      = 0
LOWERBOUND = 1
UPPERBOUND = 2

TT_MAX_SIZE     = 500_000
FUTILITY_MARGINS = {1: 320, 2: 500, 3: 900}
DELTA_MARGIN    = 200
RAZORING_MARGIN = 300
PROBCUT_MARGIN  = 100
SINGULAR_MARGIN = 50
CONTEMPT        = 25
LMP_COUNTS      = {1: 5, 2: 10, 3: 18}

# ── Masques bitboard pions passés (pré-calculés au chargement) ───────
def _build_passed_masks():
    w, b = [0] * 64, [0] * 64
    for sq in range(64):
        rank, file = sq >> 3, sq & 7
        lo, hi = max(0, file - 1), min(7, file + 1)
        for r in range(rank + 1, 8):
            for f in range(lo, hi + 1):
                w[sq] |= 1 << (r * 8 + f)
        for r in range(rank):
            for f in range(lo, hi + 1):
                b[sq] |= 1 << (r * 8 + f)
    return w, b

PASSED_MASK_WHITE, PASSED_MASK_BLACK = _build_passed_masks()

# Masques de colonne (file) pour chaque colonne 0-7
FILE_MASK = [sum(1 << (r * 8 + f) for r in range(8)) for f in range(8)]


class TreeIA:
    def __init__(self, depth=2, transpo_file="coups.json", train_mode=True):
        self.depth        = depth
        self.transpo_file = transpo_file
        self.train_mode   = train_mode

        self.opening_moves_played = {True: 0, False: 0}

        if os.path.exists(transpo_file):
            try:
                with open(transpo_file) as f:
                    tt = json.load(f)
                self.transposition_table = tt if isinstance(tt, dict) else {}
            except Exception:
                self.transposition_table = {}
        else:
            self.transposition_table = {}

        self.killer_moves   = [[None, None] for _ in range(64)]
        self.history        = {}
        self.counter_moves  = [[None] * 64 for _ in range(64)]
        self.nodes_searched = 0
        self.time_limit     = 5.0
        self._search_start_time = 0.0
        self._time_exceeded     = False
        self._last_move         = None

    # ── Clé Zobrist ──────────────────────────────────────────────────

    @staticmethod
    def _zobrist(board):
        try:
            return int(chess.polyglot.zobrist_hash(board))
        except Exception:
            return board.fen()

    # ── Sauvegarde disque ────────────────────────────────────────────

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
    #                         ÉVALUATION
    # ==================================================================

    def evaluate(self):
        b = self.board
        if b.is_checkmate():
            return -100_000 if b.turn == WHITE else 100_000
        if b.is_stalemate() or b.is_insufficient_material():
            return 0

        # ── Pièces (cache local pour éviter les appels répétés) ──────
        wp = {pt: b.pieces(pt, WHITE) for pt in range(1, 7)}
        bp = {pt: b.pieces(pt, BLACK) for pt in range(1, 7)}

        # ── Matériel ─────────────────────────────────────────────────
        score = 0
        mat_w = mat_b = 0
        for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN]:
            v = PIECE_VALUES[pt]
            nw, nb = len(wp[pt]), len(bp[pt])
            score += v * (nw - nb)
            if pt != PAWN:
                mat_w += v * nw
                mat_b += v * nb

        total_material = mat_w + mat_b
        is_endgame     = total_material < 2600
        material_balance = score  # valeur avant PST

        # ── PST ──────────────────────────────────────────────────────
        for sq in wp[PAWN]:   score += PAWN_TABLE_WHITE[sq]
        for sq in bp[PAWN]:   score -= PAWN_TABLE_BLACK[sq]
        for sq in wp[KNIGHT]: score += KNIGHT_TABLE[sq]
        for sq in bp[KNIGHT]: score -= KNIGHT_TABLE[sq ^ 56]
        for sq in wp[BISHOP]: score += BISHOP_TABLE_WHITE[sq]
        for sq in bp[BISHOP]: score -= BISHOP_TABLE_BLACK[sq]
        for sq in wp[ROOK]:   score += ROOK_TABLE_WHITE[sq]
        for sq in bp[ROOK]:   score -= ROOK_TABLE_BLACK[sq]
        for sq in wp[QUEEN]:  score += QUEEN_TABLE_WHITE[sq]
        for sq in bp[QUEEN]:  score -= QUEEN_TABLE_BLACK[sq]

        if len(wp[BISHOP]) >= 2: score += 50
        if len(bp[BISHOP]) >= 2: score -= 50

        wking, bking = b.king(WHITE), b.king(BLACK)
        if is_endgame:
            score += KING_EG_TABLE_WHITE[wking] - KING_EG_TABLE_BLACK[bking]
        else:
            score += KING_MG_TABLE_WHITE[wking] - KING_MG_TABLE_BLACK[bking]

        # ── Sous-évaluations (données pré-calculées partagées) ───────
        wpbb = int(wp[PAWN])
        bpbb = int(bp[PAWN])

        score += self._eval_rook_placement(wp[ROOK], bp[ROOK], wpbb, bpbb)
        score += self._eval_pawn_structure(wp[PAWN], bp[PAWN], wpbb, bpbb)
        score += self._eval_center_control()
        score += self._eval_mobility(wp, bp)
        score += self._eval_king_safety(wp, bp, wking, bking)

        if is_endgame:
            score += self._eval_endgame_king(wking, bking, wpbb, bpbb)
        else:
            score += self._eval_castling(wking, bking)

        score += self._eval_tactics(wp, bp)
        score += self._eval_king_exposure(wp, bp, wking, bking, wpbb, bpbb)
        score += self._eval_pins(wp, bp)

        # ── Bonus de simplification ──────────────────────────────────
        if abs(material_balance) > 200:
            bonus = max(0, (6400 - total_material) // 200)
            score += bonus if material_balance > 200 else -bonus

        return score

    # ------------------------------------------------------------------

    def _eval_castling(self, wk, bk):
        score = 0
        b = self.board
        if b.has_kingside_castling_rights(WHITE):  score += 15
        if b.has_queenside_castling_rights(WHITE): score += 10
        if b.has_kingside_castling_rights(BLACK):  score -= 15
        if b.has_queenside_castling_rights(BLACK): score -= 10
        if wk in (6, 2):  score += 30
        if bk in (62, 58): score -= 30
        return score

    def _eval_endgame_king(self, wk, bk, wpbb, bpbb):
        score = 0
        b     = self.board

        # Centralisation
        center = 3.5
        wd = abs((wk & 7) - center) + abs((wk >> 3) - center)
        bd = abs((bk & 7) - center) + abs((bk >> 3) - center)
        score += int((bd - wd) * 25)

        # KK : éloigner le roi perdant du bord
        if not (b.occupied_co[WHITE] & ~b.pawns & ~b.kings) and \
           not (b.occupied_co[BLACK] & ~b.pawns & ~b.kings):
            be = min(bk & 7, 7 - (bk & 7), bk >> 3, 7 - (bk >> 3))
            we = min(wk & 7, 7 - (wk & 7), wk >> 3, 7 - (wk >> 3))
            score += (we - be) * 5

        # Opposition
        wr, wf = wk >> 3, wk & 7
        br, bf = bk >> 3, bk & 7
        if (wr == br and abs(wf - bf) == 2) or (wf == bf and abs(wr - br) == 2):
            score += 30 if b.turn == BLACK else -30
        elif abs(wf - bf) % 2 == 0 and abs(wr - br) % 2 == 0 and abs(wf-bf)+abs(wr-br) > 2:
            score += 15 if b.turn == BLACK else -15

        # Carré du pion passé
        extra_w = 0 if b.turn == WHITE else 1
        extra_b = 0 if b.turn == BLACK else 1
        for sq in b.pieces(PAWN, WHITE):
            if not (PASSED_MASK_WHITE[sq] & bpbb):
                rank, file = sq >> 3, sq & 7
                if max(abs((bk >> 3) - 7), abs((bk & 7) - file)) > (7 - rank) + extra_w:
                    score += 200
                if rank >= 4:
                    key_rank = min(7, rank + 2)
                    for kf in range(max(0, file-1), min(7, file+1)+1):
                        if wk == key_rank * 8 + kf:
                            score += 50

        for sq in b.pieces(PAWN, BLACK):
            if not (PASSED_MASK_BLACK[sq] & wpbb):
                rank, file = sq >> 3, sq & 7
                if max(abs((wk >> 3) - 0), abs((wk & 7) - file)) > rank + extra_b:
                    score -= 200
                if rank <= 3:
                    key_rank = max(0, rank - 2)
                    for kf in range(max(0, file-1), min(7, file+1)+1):
                        if bk == key_rank * 8 + kf:
                            score -= 50

        return score

    def _eval_rook_placement(self, wrooks, brooks, wpbb, bpbb):
        """Bitboards pour détecter colonnes ouvertes/semi-ouvertes."""
        score = 0
        all_pawn_bb = wpbb | bpbb
        for sq in wrooks:
            fm = FILE_MASK[sq & 7]
            if not (all_pawn_bb & fm):   score += 40
            elif not (wpbb & fm):        score += 20
        for sq in brooks:
            fm = FILE_MASK[sq & 7]
            if not (all_pawn_bb & fm):   score -= 40
            elif not (bpbb & fm):        score -= 20
        return score

    def _eval_pawn_structure(self, wpawns, bpawns, wpbb, bpbb):
        score = 0
        wp_list = list(wpawns)
        bp_list = list(bpawns)
        wp_set  = set(wp_list)
        bp_set  = set(bp_list)

        # Fichiers occupés (pour pions doublés et isolés en une passe)
        wf_set = set(sq & 7 for sq in wp_list)
        bf_set = set(sq & 7 for sq in bp_list)
        from collections import Counter
        wfc = Counter(sq & 7 for sq in wp_list)
        bfc = Counter(sq & 7 for sq in bp_list)

        # Pions doublés
        for f, c in wfc.items():
            if c > 1: score -= 15 * (c - 1)
        for f, c in bfc.items():
            if c > 1: score += 15 * (c - 1)

        # Pions isolés (pas de voisin sur colonne adjacente)
        for sq in wp_list:
            f = sq & 7
            if (f - 1 not in wf_set) and (f + 1 not in wf_set):
                score -= 20
        for sq in bp_list:
            f = sq & 7
            if (f - 1 not in bf_set) and (f + 1 not in bf_set):
                score += 20

        # Pions passés
        for sq in wp_list:
            if not (PASSED_MASK_WHITE[sq] & bpbb):
                rank, file = sq >> 3, sq & 7
                bonus = 30 + rank * 10
                bk = self.board.king(BLACK)    # Distance du roi noir
                if bk is not None:
                    dist = abs((bk & 7) - file) + abs((bk >> 3) - rank)
                    bonus += dist * 3
                for f in (file - 1, file + 1):
                    if 0 <= f < 8 and (rank - 1) * 8 + f in wp_set:
                        bonus += 40; break
                score += bonus

        for sq in bp_list:
            if not (PASSED_MASK_BLACK[sq] & wpbb):
                rank, file = sq >> 3, sq & 7
                bonus = 30 + (7 - rank) * 10
                wk = self.board.king(WHITE)
                if wk is not None:
                    dist = abs((wk & 7) - file) + abs((wk >> 3) - rank)
                    bonus += dist * 3
                for f in (file - 1, file + 1):
                    if 0 <= f < 8 and (rank + 1) * 8 + f in bp_set:
                        bonus += 40; break
                score -= bonus

        # Pions arriérés
        b = self.board
        for sq in wp_list:
            file, rank = sq & 7, sq >> 3
            blocked = False
            for r in range(rank + 1, 8):
                p = b.piece_at(r * 8 + file)
                if p and p.piece_type == PAWN:
                    if p.color == BLACK: blocked = True
                    break
            if not blocked: continue
            supported = False
            for nf in (file - 1, file + 1):
                if 0 <= nf < 8:
                    for r in range(rank - 1, 0, -1):
                        p = b.piece_at(r * 8 + nf)
                        if p and p.piece_type == PAWN and p.color == WHITE:
                            supported = True; break
                if supported: break
            if not supported: score -= 15

        for sq in bp_list:
            file, rank = sq & 7, sq >> 3
            blocked = False
            for r in range(rank - 1, -1, -1):
                p = b.piece_at(r * 8 + file)
                if p and p.piece_type == PAWN:
                    if p.color == WHITE: blocked = True
                    break
            if not blocked: continue
            supported = False
            for nf in (file - 1, file + 1):
                if 0 <= nf < 8:
                    for r in range(rank + 1, 7):
                        p = b.piece_at(r * 8 + nf)
                        if p and p.piece_type == PAWN and p.color == BLACK:
                            supported = True; break
                if supported: break
            if not supported: score += 15

        # Pions connectés
        for sq in wp_list:
            rank, file = sq >> 3, sq & 7
            if rank > 0 and any(0 <= f < 8 and (rank-1)*8+f in wp_set
                                for f in (file-1, file+1)):
                score += 10 + rank * 2

        for sq in bp_list:
            rank, file = sq >> 3, sq & 7
            if rank < 7 and any(0 <= f < 8 and (rank+1)*8+f in bp_set
                                for f in (file-1, file+1)):
                score -= 10 + (7 - rank) * 2

        # Pions passés candidats
        for sq in wp_list:
            file, rank = sq & 7, sq >> 3
            if not (PASSED_MASK_WHITE[sq] & bpbb):
                continue
            if all(
                sum(1 for p in bp_list if p & 7 == f and p >> 3 > rank) <=
                sum(1 for p in wp_list if p & 7 == f and p >> 3 > rank)
                for f in range(max(0,file-1), min(7,file+1)+1)
            ) and any(p & 7 == f and p >> 3 > rank
                      for f in range(max(0,file-1), min(7,file+1)+1)
                      for p in bp_list):
                score += 10 + rank * 3

        for sq in bp_list:
            file, rank = sq & 7, sq >> 3
            if not (PASSED_MASK_BLACK[sq] & wpbb):
                continue
            if all(
                sum(1 for p in wp_list if p & 7 == f and p >> 3 < rank) <=
                sum(1 for p in bp_list if p & 7 == f and p >> 3 < rank)
                for f in range(max(0,file-1), min(7,file+1)+1)
            ) and any(p & 7 == f and p >> 3 < rank
                      for f in range(max(0,file-1), min(7,file+1)+1)
                      for p in wp_list):
                score -= 10 + (7 - rank) * 3

        return score

    def _eval_center_control(self):
        score = 0
        b = self.board
        for sq in (27, 28, 35, 36):
            if b.is_attacked_by(WHITE, sq): score += 10
            if b.is_attacked_by(BLACK, sq): score -= 10
            p = b.piece_at(sq)
            if p: score += 20 if p.color == WHITE else -20
        for sq in (18, 19, 20, 21, 26, 29, 34, 37, 42, 43, 44, 45):
            if b.is_attacked_by(WHITE, sq): score += 3
            if b.is_attacked_by(BLACK, sq): score -= 3
        return score

    def _eval_mobility(self, wp, bp):
        """popcount via bin() — pas de list() intermédiaire."""
        b  = self.board
        wa = sum(bin(int(b.attacks(sq))).count('1')
                 for pt in (QUEEN, ROOK, BISHOP, KNIGHT)
                 for sq in wp[pt])
        ba = sum(bin(int(b.attacks(sq))).count('1')
                 for pt in (QUEEN, ROOK, BISHOP, KNIGHT)
                 for sq in bp[pt])
        return (wa - ba) * 3

    def _eval_king_safety(self, wp, bp, wk, bk):
        score = 0
        b = self.board

        for king, color, sign, shield_dir in (
            (wk, WHITE,  1,  1),
            (bk, BLACK, -1, -1),
        ):
            if king is None: continue
            opp = not color
            attackers = sum(1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king])
                            if (p := b.piece_at(sq)) and p.color == opp)
            file, rank = king & 7, king >> 3
            sr1 = rank + shield_dir
            sr2 = rank + 2 * shield_dir
            pawn_shield = sum(
                1 for r in (sr1, sr2) if 0 <= r < 8
                for f in range(max(0, file-1), min(7, file+1)+1)
                if (p := b.piece_at(r*8+f)) and p.piece_type == PAWN and p.color == color
            )
            weak = sum(1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king])
                       if not b.is_attacked_by(color, sq))
            score += sign * (-25 * attackers + 12 * pawn_shield - 8 * weak)

        return score

    def _eval_tactics(self, wp, bp):
        score = 0
        b = self.board

        for color, sign in ((WHITE, 1), (BLACK, -1)):
            opp = not color
            pieces = wp if color == WHITE else bp
            for pt in (QUEEN, ROOK, BISHOP, KNIGHT):
                for sq in pieces[pt]:
                    if not b.is_attacked_by(color, sq):
                        v = PIECE_VALUES[pt]
                        score -= sign * (v // 2 if b.is_attacked_by(opp, sq) else v // 8)

        for sq in wp[ROOK]:
            if sq >> 3 == 6: score += 50
        for sq in bp[ROOK]:
            if sq >> 3 == 1: score -= 50

        for sq in wp[KNIGHT]:
            rank, file = sq >> 3, sq & 7
            if rank >= 4 and rank + 1 < 8 and not any(
                b.piece_at((rank+1)*8+f) and
                b.piece_at((rank+1)*8+f).piece_type == PAWN and
                b.piece_at((rank+1)*8+f).color == BLACK
                for f in (file-1, file+1) if 0 <= f < 8
            ): score += 20

        for sq in bp[KNIGHT]:
            rank, file = sq >> 3, sq & 7
            if rank <= 3 and rank - 1 >= 0 and not any(
                b.piece_at((rank-1)*8+f) and
                b.piece_at((rank-1)*8+f).piece_type == PAWN and
                b.piece_at((rank-1)*8+f).color == WHITE
                for f in (file-1, file+1) if 0 <= f < 8
            ): score -= 20

        for sq in wp[BISHOP]:
            if bin(int(b.attacks(sq))).count('1') >= 6: score += 15
        for sq in bp[BISHOP]:
            if bin(int(b.attacks(sq))).count('1') >= 6: score -= 15

        return score

    def _eval_king_exposure(self, wp, bp, wk, bk, wpbb, bpbb):
        score = 0
        b = self.board
        if bk is not None:
            fm = FILE_MASK[bk & 7]
            if not (bpbb & fm):
                heavy = sum(1 for pt in (ROOK, QUEEN) for sq in wp[pt] if sq & 7 == bk & 7)
                if heavy: score += 25 * heavy
        if wk is not None:
            fm = FILE_MASK[wk & 7]
            if not (wpbb & fm):
                heavy = sum(1 for pt in (ROOK, QUEEN) for sq in bp[pt] if sq & 7 == wk & 7)
                if heavy: score -= 25 * heavy
        return score

    def _eval_pins(self, wp, bp):
        score = 0
        b = self.board
        for pt in (KNIGHT, BISHOP, ROOK, QUEEN):
            for sq in wp[pt]:
                if b.is_pinned(WHITE, sq): score -= PIECE_VALUES[pt] // 8
            for sq in bp[pt]:
                if b.is_pinned(BLACK, sq): score += PIECE_VALUES[pt] // 8
        return score

    # ==================================================================
    #   SEE
    # ==================================================================

    def _solve_kpk(self):
        b = self.board

        pieces = b.piece_map()
        if len(pieces) != 3:
            return None

        pawns = list(b.pieces(PAWN, WHITE)) + list(b.pieces(PAWN, BLACK))
        if len(pawns) != 1:
            return None

        pawn_sq = pawns[0]
        pawn_color = b.color_at(pawn_sq)

        wk = b.king(WHITE)
        bk = b.king(BLACK)

        # Règle du carré simplifiée
        rank = pawn_sq >> 3
        file = pawn_sq & 7

        if pawn_color == WHITE:
            steps = 7 - rank
            dist = max(abs((bk >> 3) - 7), abs((bk & 7) - file))
            if dist > steps:
                return 100000
        else:
            steps = rank
            dist = max(abs((wk >> 3) - 0), abs((wk & 7) - file))
            if dist > steps:
                return -100000

        return None
    

    def see(self, to_sq, attacker_color):
        target = self.board.piece_at(to_sq)
        if target is None:
            return 0
        b = self.board

        def collect(color):
            vals = sorted(
                PIECE_VALUES.get(pt, 20000)
                for pt in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
                for sq in b.pieces(pt, color)
                if to_sq in b.attacks(sq)
            )
            return vals

        atk = collect(attacker_color)
        if not atk:
            return 0
        def_ = collect(not attacker_color)

        gain, cap = [], PIECE_VALUES[target.piece_type]
        sides, side = [atk, def_], 0
        while sides[side]:
            pv = sides[side].pop(0)
            gain.append(cap)
            cap = pv
            side ^= 1

        for d in range(len(gain) - 1, 0, -1):
            gain[d-1] -= max(0, gain[d])
        return gain[0] if gain else 0

    # ==================================================================
    #   Ordonnancement des coups
    # ==================================================================

    def _order_moves(self, moves, depth, prev_move=None):
        b       = self.board
        zkey    = self._zobrist(b)
        tt_move = None
        if zkey in self.transposition_table:
            try:
                tt_move = chess.Move.from_uci(self.transposition_table[zkey]["best_move"])
            except Exception:
                pass

        counter_move = (self.counter_moves[prev_move.from_square][prev_move.to_square]
                        if prev_move is not None else None)
        killers = self.killer_moves[depth] if depth < 64 else [None, None]

        def _score(move):
            if tt_move and move == tt_move:
                return 1_000_000

            sc = 0
            if b.is_capture(move):
                target   = b.piece_at(move.to_square)
                attacker = b.piece_at(move.from_square)
                if target and attacker:
                    ss = self.see(move.to_square, b.turn)
                    sc += (500_000 + ss) if ss >= 0 else ss

            if move.promotion:
                sc += 900 if move.promotion == QUEEN else 300

            if move == killers[0]:   sc += 1_000
            elif move == killers[1]: sc += 800

            if counter_move and move == counter_move:
                sc += 600

            mk = move.uci()
            if mk in self.history:
                sc += self.history[mk]

            if move.to_square in (27, 28, 35, 36):
                sc += 10

            return sc

        return sorted(moves, key=_score, reverse=True)

    def _update_killers(self, move, depth):
        if depth < 64 and self.killer_moves[depth][0] != move:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move

    def _update_history(self, move, depth):
        k = move.uci()
        self.history[k] = self.history.get(k, 0) + depth * depth

    def _update_counter_move(self, prev_move, move):
        if prev_move is not None:
            self.counter_moves[prev_move.from_square][prev_move.to_square] = move

    # ==================================================================
    #   Quiescence
    # ==================================================================

    def quiescence(self, alpha, beta):
        b         = self.board
        stand_pat = self.evaluate()
        if b.turn == BLACK:
            stand_pat = -stand_pat

        if stand_pat >= beta: return beta
        if stand_pat > alpha: alpha = stand_pat

        captures = sorted(
            (m for m in b.legal_moves if b.is_capture(m)),
            key=lambda m: PIECE_VALUES.get(
                b.piece_at(m.to_square).piece_type, 0) if b.piece_at(m.to_square) else 0,
            reverse=True,
        )

        for move in captures:
            target = b.piece_at(move.to_square)
            if target:
                gain = PIECE_VALUES.get(target.piece_type, 0)
                if move.promotion == QUEEN:
                    gain += PIECE_VALUES[QUEEN] - PIECE_VALUES[PAWN]
                if stand_pat + gain + DELTA_MARGIN <= alpha:
                    continue

            b.push(move)
            score = -self.quiescence(-beta, -alpha)
            b.pop()

            if score >= beta: return beta
            if score > alpha: alpha = score

        return alpha

    # ==================================================================
    #   Null Move
    # ==================================================================

    def _try_null_move(self, depth, beta, ply):
        b = self.board
        if depth < 3 or b.is_check() or len(b.piece_map()) <= 8:
            return None
        if not (b.occupied_co[b.turn] & ~b.pawns & ~b.kings):
            return None
        b.push(chess.Move.null())
        score = -self.negamax(depth - 3, -beta, -beta + 1, ply + 1, None)
        b.pop()
        return beta if score >= beta else None

    # ==================================================================
    #   Negamax
    # ==================================================================

    def negamax(self, depth, alpha, beta, ply, prev_move=None):
        if self._time_exceeded:
            return 0
        if time.time() - self._search_start_time > self.time_limit:
            self._time_exceeded = True
            return 0
        self.nodes_searched += 1
        b = self.board

        # ── Solveur KPK ────────────────────────────
        kpk_score = self._solve_kpk()
        if kpk_score is not None:
            return kpk_score - ply

        if b.is_repetition(2):
            return -CONTEMPT

        if b.halfmove_clock >= 100:
            return 0

        zkey       = self._zobrist(b)
        alpha_orig = alpha

        if zkey in self.transposition_table:
            entry = self.transposition_table[zkey]
            if entry["depth"] >= depth:
                flag, score = entry.get("flag", EXACT), entry["score"]
                if flag == EXACT:                  return score
                elif flag == LOWERBOUND: alpha = max(alpha, score)
                elif flag == UPPERBOUND: beta  = min(beta,  score)
                if alpha >= beta:                  return score

        in_check = b.is_check()
        if in_check and ply < 2 * self.depth + 4:
            depth += 1

        if depth <= 0:
            if b.is_game_over():
                s = self.evaluate()
                return s if b.turn == WHITE else -s
            return self.quiescence(alpha, beta)

        if b.is_game_over():
            return (-100_000 + ply) if b.is_checkmate() else 0

        # Razoring
        if depth == 1 and not in_check and abs(alpha) < 90_000:
            se = self.evaluate()
            if b.turn == BLACK: se = -se
            if se < alpha - RAZORING_MARGIN:
                return self.quiescence(alpha, beta)

        # Null Move
        if not in_check and depth >= 3:
            nr = self._try_null_move(depth, beta, ply)
            if nr is not None:
                return nr

        moves = self._order_moves(list(b.legal_moves), ply, prev_move)
        if not moves:
            return 0

        # IID
        if depth >= 4 and not in_check and zkey not in self.transposition_table:
            self.negamax(depth - 2, alpha, beta, ply, prev_move)
            moves = self._order_moves(moves, ply, prev_move)

        # Probcut
        if depth >= 5 and not in_check and abs(beta) < 90_000:
            pc_beta   = beta + PROBCUT_MARGIN
            pc_static = self.evaluate()
            if b.turn == BLACK: pc_static = -pc_static
            threshold = pc_beta - pc_static
            for m in [m for m in moves if b.is_capture(m) and
                      self.see(m.to_square, b.turn) >= threshold][:3]:
                b.push(m)
                pc_score = -self.negamax(max(1, depth-4), -pc_beta, -pc_beta+1, ply+1, m)
                b.pop()
                if pc_score >= pc_beta:
                    return pc_beta

        # Singular Extension
        singular_move = None
        if depth >= 4 and not in_check and zkey in self.transposition_table:
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
                        for i, m in enumerate(moves):
                            if m == cand or i >= 6: continue
                            b.push(m)
                            s_val = -self.negamax(s_depth, -s_beta, -(s_beta-1), ply+1, cand)
                            b.pop()
                            if s_val >= s_beta:
                                s_fails = True; break
                        if not s_fails:
                            singular_move = cand
                except Exception:
                    pass

        # Futility
        futility_pruning = False
        if (depth in FUTILITY_MARGINS and not in_check
                and abs(alpha) < 90_000 and abs(beta) < 90_000):
            fe = self.evaluate()
            if b.turn == BLACK: fe = -fe
            futility_pruning = fe + FUTILITY_MARGINS[depth] <= alpha

        # Boucle principale
        best_move, best_score = moves[0], -10**9
        moves_searched = quiet_count = 0

        for move in moves:
            is_capture   = b.is_capture(move)
            is_promotion = bool(move.promotion)
            is_quiet     = not is_capture and not is_promotion

            if futility_pruning and is_quiet and moves_searched > 0:
                continue
            if (depth in LMP_COUNTS and is_quiet and not in_check
                    and moves_searched > 0 and quiet_count >= LMP_COUNTS[depth]):
                continue
            if is_quiet:
                quiet_count += 1

            b.push(move)
            if moves_searched == 0:
                ext   = 1 if singular_move is not None and move == singular_move else 0
                score = -self.negamax(depth - 1 + ext, -beta, -alpha, ply + 1, move)
            else:
                reduction = 0
                if (depth >= 3 and moves_searched >= 4
                        and not in_check and not is_capture and not b.is_check()):
                    reduction = max(1, int(math.log(depth) * math.log(moves_searched) / 1.5))

                score = -self.negamax(depth-1-reduction, -alpha-1, -alpha, ply+1, move)
                if alpha < score < beta:
                    if reduction > 0:
                        score = -self.negamax(depth-1, -alpha-1, -alpha, ply+1, move)
                    if score > alpha:
                        score = -self.negamax(depth-1, -beta, -alpha, ply+1, move)

            b.pop()
            moves_searched += 1

            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)

            if alpha >= beta:
                if not is_capture:
                    self._update_killers(move, ply)
                    self._update_history(move, depth)
                    self._update_counter_move(prev_move, move)
                break

        # TT eviction : supprime aléatoirement 10% des entrées les moins profondes
        if len(self.transposition_table) >= TT_MAX_SIZE:
            shallow = [k for k, v in self.transposition_table.items()
                       if v.get("depth", 0) <= 2]
            for k in random.sample(shallow, min(TT_MAX_SIZE // 10, len(shallow))):
                del self.transposition_table[k]

        flag = EXACT
        if best_score <= alpha_orig: flag = UPPERBOUND
        elif best_score >= beta:     flag = LOWERBOUND

        self.transposition_table[zkey] = {
            "best_move": best_move.uci(),
            "score":     best_score,
            "depth":     depth,
            "flag":      flag,
        }
        return best_score

    # ==================================================================
    #   Ouvertures / Promotion / Coup principal
    # ==================================================================

    def get_opening_move(self, board):
        moves = OPENING_BOOK.get(board.fen())
        return random.choice(moves) if moves else None

    def _smart_promotion(self, board, move):
        if move.promotion != QUEEN:
            return move
        board.push(move)
        is_stale = board.is_stalemate()
        board.pop()
        if not is_stale:
            return move
        for piece in (ROOK, KNIGHT, BISHOP):
            alt = chess.Move(move.from_square, move.to_square, promotion=piece)
            if alt not in board.legal_moves: continue
            board.push(alt)
            still = board.is_stalemate()
            board.pop()
            if not still: return alt
        return move

    def coup(self, board):
        # Ouvertures
        color = board.turn
        if self.opening_moves_played[color] < 12:
            mv = self.get_opening_move(board)
            if mv:
                self.opening_moves_played[color] += 1
                if isinstance(mv, str):
                    mv = board.parse_san(mv)
                return mv

        # Mat en 1
        for mv in board.legal_moves:
            board.push(mv)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return mv

        # Initialisation
        self.board              = board
        self.killer_moves       = [[None, None] for _ in range(64)]
        self.nodes_searched     = 0
        self._search_start_time = time.time()
        self._time_exceeded     = False

        for key in self.history:
            self.history[key] //= 2

        best_move = prev_score = None

        # ── Depth adaptatif en finale ─────────────────────────
        piece_count = len(board.piece_map())
        if piece_count <= 8 :
            effective_depth = self.depth + 1 
        elif piece_count <= 5 :
            effective_depth = self.depth + 2 
        else :                 
            effective_depth = self.depth

        for d in range(1, effective_depth + 1):
            self._time_exceeded = False

            if d >= 2 and best_move is not None:
                delta, a, b_ = 50, prev_score - 50, prev_score + 50
                while True:
                    self._time_exceeded = False
                    score, move = self.negamax_root(d, a, b_)
                    if self._time_exceeded: break
                    if score <= a:
                        a -= delta; delta = min(delta * 2, 500)
                    elif score >= b_:
                        b_ += delta; delta = min(delta * 2, 500)
                    else:
                        break
                    if delta > 2_000:
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
            if not moves:
                raise ValueError("Aucun coup légal trouvé !")
            scored = sorted(
                ((PIECE_VALUES[board.piece_at(m.to_square).piece_type]
                  if board.is_capture(m) and board.piece_at(m.to_square) else 0), m)
                for m in moves
            )
            best_val = scored[-1][0]
            best_move = random.choice([m for v, m in scored if v == best_val])

        best_move = self._smart_promotion(board, best_move)
        self._last_move = best_move
        return best_move

    # ==================================================================
    #   Negamax racine
    # ==================================================================

    def negamax_root(self, depth, alpha=-10**9, beta=10**9):
        best_score, best_move = -10**9, None
        moves = self._order_moves(list(self.board.legal_moves), 0, self._last_move)
        if not moves:
            return 0, None

        for move in moves:
            self.board.push(move)
            score = -self.negamax(depth - 1, -beta, -alpha, 1, move)
            self.board.pop()

            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)

            if time.time() - self._search_start_time > self.time_limit:
                self._time_exceeded = True
                break

        return best_score, best_move