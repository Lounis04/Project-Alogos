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
    PAWN: 100,
    KNIGHT: 320,
    BISHOP: 330,
    ROOK: 500,
    QUEEN: 900,
    KING: 20000,
}

# Types de nœuds pour la table de transposition
EXACT      = 0
LOWERBOUND = 1
UPPERBOUND = 2

# Taille maximale de la table de transposition (en nombre d'entrées)
TT_MAX_SIZE = 500_000


class TreeIA:
    def __init__(self, depth=2, transpo_file="coups.json", train_mode=True):
        self.depth       = depth
        self.transpo_file = transpo_file
        self.train_mode  = train_mode

        # Compteur d'ouverture séparé par couleur pour l'auto-jeu (entraînement).
        # True = Blancs, False = Noirs  (correspond à chess.WHITE / chess.BLACK)
        self.opening_moves_played = {True: 0, False: 0}

        # ------------------------------------------------------------------
        #   Table de transposition (en RAM)
        #   Clés : entiers Zobrist  (int 64-bit)
        #   Ancien format FEN (chaînes avec '/' ou ' ') → ignoré, repart à zéro.
        # ------------------------------------------------------------------
        if os.path.exists(transpo_file):
            try:
                with open(transpo_file, "r") as f:
                    self.transposition_table = json.load(f)
                # Sanity check : si les valeurs ne sont pas des dicts, on repart à zéro
                if not isinstance(self.transposition_table, dict):
                    self.transposition_table = {}
            except Exception:
                self.transposition_table = {}
        else:
            self.transposition_table = {}

        # Killer moves (2 par profondeur, jusqu'à 64 niveaux)
        self.killer_moves = [[None, None] for _ in range(64)]

        # History heuristic
        self.history = {}

        # Compteur de nœuds (debug)
        self.nodes_searched = 0

        # Gestion du temps
        self.time_limit         = 5.0
        self._search_start_time = 0.0
        self._time_exceeded     = False

    # ------------------------------------------------------------------
    #   Clé de transposition (Zobrist ou fallback FEN)
    # ------------------------------------------------------------------

    @staticmethod
    def _zobrist(board):
        """
        Retourne une STRING unique représentant la position, toujours
        sérialisable en JSON quelle que soit la version de python-chess.
        Ordre de préférence :
          1. board._transposition_key()  → str() (peut être int ou tuple selon version)
          2. chess.polyglot.zobrist_hash() → str()
          3. board.fen()  (fallback universel)
        """
        if hasattr(board, '_transposition_key'):
            return str(board._transposition_key())
        try:
            import chess.polyglot
            return str(chess.polyglot.zobrist_hash(board))
        except Exception:
            return board.fen()

    # ------------------------------------------------------------------
    #   Sauvegarde (uniquement en mode entraînement)
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
            print(f"[save_transpo] Erreur lors de la sauvegarde : {e}")

    # ==================================================================
    #                           ÉVALUATION
    # ==================================================================

    def evaluate(self):
        """
        Évaluation statique TOUJOURS du point de vue des BLANCS.
        Score positif  = bon pour les Blancs.
        Score négatif  = bon pour les Noirs.
        Negamax gère l'inversion avec  -negamax().
        """
        if self.board.is_checkmate():
            return -100_000 if self.board.turn == WHITE else 100_000

        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0

        score = 0

        # ── Matériel ─────────────────────────────────────────────────
        for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN]:
            score += PIECE_VALUES[pt] * (
                len(self.board.pieces(pt, WHITE))
                - len(self.board.pieces(pt, BLACK))
            )

        # ── PST Pions ────────────────────────────────────────────────
        for sq in self.board.pieces(PAWN, WHITE):
            score += PAWN_TABLE_WHITE[sq]
        for sq in self.board.pieces(PAWN, BLACK):
            score -= PAWN_TABLE_BLACK[sq]

        # ── PST Cavaliers ────────────────────────────────────────────
        for sq in self.board.pieces(KNIGHT, WHITE):
            score += KNIGHT_TABLE[sq]
        for sq in self.board.pieces(KNIGHT, BLACK):
            score -= KNIGHT_TABLE[sq ^ 56]

        # ── PST Fous ─────────────────────────────────────────────────
        for sq in self.board.pieces(BISHOP, WHITE):
            score += BISHOP_TABLE_WHITE[sq]
        for sq in self.board.pieces(BISHOP, BLACK):
            score -= BISHOP_TABLE_BLACK[sq]

        if len(self.board.pieces(BISHOP, WHITE)) >= 2:
            score += 50
        if len(self.board.pieces(BISHOP, BLACK)) >= 2:
            score -= 50

        # ── PST Tours ────────────────────────────────────────────────
        for sq in self.board.pieces(ROOK, WHITE):
            score += ROOK_TABLE_WHITE[sq]
        for sq in self.board.pieces(ROOK, BLACK):
            score -= ROOK_TABLE_BLACK[sq]

        # ── PST Reines ───────────────────────────────────────────────
        for sq in self.board.pieces(QUEEN, WHITE):
            score += QUEEN_TABLE_WHITE[sq]
        for sq in self.board.pieces(QUEEN, BLACK):
            score -= QUEEN_TABLE_BLACK[sq]

        # ── Roi (milieu / finale) ────────────────────────────────────
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

        # ── Évaluations thématiques ──────────────────────────────────
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

        # ── Bonus de simplification ──────────────────────────────────
        # Le camp avantagé (> 200 points) préfère échanger pour simplifier.
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
        """Bonus pour droits de roque – point de vue BLANCS."""
        score = 0

        if self.board.has_kingside_castling_rights(WHITE):
            score += 15
        if self.board.has_queenside_castling_rights(WHITE):
            score += 10
        if self.board.has_kingside_castling_rights(BLACK):
            score -= 15
        if self.board.has_queenside_castling_rights(BLACK):
            score -= 10

        white_king_sq = self.board.king(WHITE)
        black_king_sq = self.board.king(BLACK)

        if white_king_sq in [6, 2]:    # g1 ou c1
            score += 30
        if black_king_sq in [62, 58]:  # g8 ou c8
            score -= 30

        return score

    def _evaluate_endgame_king_activity(self):
        """Roi actif en finale – point de vue BLANCS."""
        score = 0

        white_king = self.board.king(WHITE)
        black_king = self.board.king(BLACK)

        center = 3.5
        white_dist = abs(white_king % 8 - center) + abs(white_king // 8 - center)
        black_dist = abs(black_king % 8 - center) + abs(black_king // 8 - center)
        score += int((black_dist - white_dist) * 15)

        white_pieces = sum(len(self.board.pieces(pt, WHITE)) for pt in [KNIGHT, BISHOP, ROOK, QUEEN])
        black_pieces = sum(len(self.board.pieces(pt, BLACK)) for pt in [KNIGHT, BISHOP, ROOK, QUEEN])

        if white_pieces == 0 and black_pieces == 0:
            black_edge = min(black_king % 8, 7 - black_king % 8,
                             black_king // 8, 7 - black_king // 8)
            white_edge = min(white_king % 8, 7 - white_king % 8,
                             white_king // 8, 7 - white_king // 8)
            score -= black_edge * 5
            score += white_edge * 5

        return score

    def _evaluate_rook_placement(self):
        """Tours sur colonnes ouvertes – point de vue BLANCS."""
        score = 0

        for rook_sq in self.board.pieces(ROOK, WHITE):
            file = rook_sq % 8
            is_open = is_semi_open = True
            for rank in range(8):
                piece = self.board.piece_at(rank * 8 + file)
                if piece and piece.piece_type == PAWN:
                    is_open = False
                    if piece.color == WHITE:
                        is_semi_open = False
            score += 40 if is_open else (20 if is_semi_open else 0)

        for rook_sq in self.board.pieces(ROOK, BLACK):
            file = rook_sq % 8
            is_open = is_semi_open = True
            for rank in range(8):
                piece = self.board.piece_at(rank * 8 + file)
                if piece and piece.piece_type == PAWN:
                    is_open = False
                    if piece.color == BLACK:
                        is_semi_open = False
            score -= 40 if is_open else (20 if is_semi_open else 0)

        return score

    def _evaluate_pawn_structure(self):
        """Structure des pions – point de vue BLANCS."""
        score = 0

        white_pawns = list(self.board.pieces(PAWN, WHITE))
        black_pawns = list(self.board.pieces(PAWN, BLACK))

        # ── Pions doublés ─────────────────────────────────────────────
        wfiles = [sq % 8 for sq in white_pawns]
        bfiles = [sq % 8 for sq in black_pawns]
        for file in range(8):
            wc = wfiles.count(file)
            bc = bfiles.count(file)
            if wc > 1:
                score -= 15 * (wc - 1)
            if bc > 1:
                score += 15 * (bc - 1)

        # ── Pions isolés ──────────────────────────────────────────────
        for sq in white_pawns:
            file = sq % 8
            if not any(p % 8 in (file - 1, file + 1) for p in white_pawns if 0 <= p % 8 < 8):
                score -= 20
        for sq in black_pawns:
            file = sq % 8
            if not any(p % 8 in (file - 1, file + 1) for p in black_pawns if 0 <= p % 8 < 8):
                score += 20

        # ── Pions passés ─────────────────────────────────────────────
        for sq in white_pawns:
            file, rank = sq % 8, sq // 8
            passed = True
            for r in range(rank + 1, 8):
                for f in [file - 1, file, file + 1]:
                    if 0 <= f < 8:
                        p = self.board.piece_at(r * 8 + f)
                        if p and p.piece_type == PAWN and p.color == BLACK:
                            passed = False
                            break
                if not passed:
                    break
            if passed:
                score += 30 + rank * 10

        for sq in black_pawns:
            file, rank = sq % 8, sq // 8
            passed = True
            for r in range(0, rank):
                for f in [file - 1, file, file + 1]:
                    if 0 <= f < 8:
                        p = self.board.piece_at(r * 8 + f)
                        if p and p.piece_type == PAWN and p.color == WHITE:
                            passed = False
                            break
                if not passed:
                    break
            if passed:
                score -= 30 + (7 - rank) * 10

        # ── Pions arriérés – Blancs ───────────────────────────────────
        for sq in white_pawns:
            file, rank = sq % 8, sq // 8
            # Bloqué par un pion adverse devant ?
            blocked = False
            for r in range(rank + 1, 8):
                p = self.board.piece_at(r * 8 + file)
                if p and p.piece_type == PAWN:
                    if p.color == BLACK:
                        blocked = True
                    break
            if not blocked:
                continue
            # Aucun pion ami en soutien derrière ?
            supported = False
            for nf in [file - 1, file + 1]:
                if 0 <= nf < 8:
                    for r in range(rank - 1, 0, -1):
                        p = self.board.piece_at(r * 8 + nf)
                        if p and p.piece_type == PAWN and p.color == WHITE:
                            supported = True
                            break
                if supported:
                    break
            if not supported:
                score -= 15

        # ── Pions arriérés – Noirs ────────────────────────────────────
        for sq in black_pawns:
            file, rank = sq % 8, sq // 8
            blocked = False
            for r in range(rank - 1, -1, -1):
                p = self.board.piece_at(r * 8 + file)
                if p and p.piece_type == PAWN:
                    if p.color == WHITE:
                        blocked = True
                    break
            if not blocked:
                continue
            supported = False
            for nf in [file - 1, file + 1]:
                if 0 <= nf < 8:
                    for r in range(rank + 1, 7):
                        p = self.board.piece_at(r * 8 + nf)
                        if p and p.piece_type == PAWN and p.color == BLACK:
                            supported = True
                            break
                if supported:
                    break
            if not supported:
                score += 15

        return score

    def _evaluate_center_control(self):
        """Contrôle du centre – point de vue BLANCS."""
        score = 0
        center   = [27, 28, 35, 36]
        extended = [18, 19, 20, 21, 26, 29, 34, 37, 42, 43, 44, 45]

        for sq in center:
            if self.board.is_attacked_by(WHITE, sq):
                score += 10
            if self.board.is_attacked_by(BLACK, sq):
                score -= 10
            piece = self.board.piece_at(sq)
            if piece:
                score += 20 if piece.color == WHITE else -20

        for sq in extended:
            if self.board.is_attacked_by(WHITE, sq):
                score += 3
            if self.board.is_attacked_by(BLACK, sq):
                score -= 3

        return score

    def _evaluate_mobility_fast(self):
        """Mobilité – point de vue BLANCS."""
        white_attacks = sum(
            len(list(self.board.attacks(sq)))
            for pt in [QUEEN, ROOK, BISHOP, KNIGHT]
            for sq in self.board.pieces(pt, WHITE)
        )
        black_attacks = sum(
            len(list(self.board.attacks(sq)))
            for pt in [QUEEN, ROOK, BISHOP, KNIGHT]
            for sq in self.board.pieces(pt, BLACK)
        )
        return (white_attacks - black_attacks) * 3

    def _evaluate_king_safety(self):
        """Sécurité du roi – point de vue BLANCS."""
        score = 0

        # Roi blanc
        wk = self.board.king(WHITE)
        if wk is not None:
            # Pièces noires adjacentes au roi blanc
            attackers = 0
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[wk]):
                p = self.board.piece_at(sq)
                if p and p.color == BLACK:
                    attackers += 1

            # Bouclier de pions
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

            # Cases faibles autour du roi
            weak = 0
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[wk]):
                if not self.board.is_attacked_by(WHITE, sq):
                    weak += 1

            score += -25 * attackers + 12 * pawn_shield - 8 * weak

        # Roi noir
        bk = self.board.king(BLACK)
        if bk is not None:
            attackers = 0
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[bk]):
                p = self.board.piece_at(sq)
                if p and p.color == WHITE:
                    attackers += 1

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

            weak = 0
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[bk]):
                if not self.board.is_attacked_by(BLACK, sq):
                    weak += 1

            score -= -25 * attackers + 12 * pawn_shield - 8 * weak

        return score

    def _evaluate_tactics(self):
        """
        Reconnaît les patterns tactiques – point de vue BLANCS.
        - Pièces non défendues (hanging)
        - Tour sur 7e rangée
        - Cavalier sur avant-poste
        - Fou actif sur grande diagonale
        """
        score = 0

        # ── Pièces non défendues (hanging) ───────────────────────────
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

        # ── Tour sur la 7e rangée ─────────────────────────────────────
        for sq in self.board.pieces(ROOK, WHITE):
            if sq // 8 == 6:
                score += 50
        for sq in self.board.pieces(ROOK, BLACK):
            if sq // 8 == 1:
                score -= 50

        # ── Cavalier sur avant-poste ──────────────────────────────────
        for sq in self.board.pieces(KNIGHT, WHITE):
            rank, file = sq // 8, sq % 8
            if rank >= 4:
                threatened = False
                if rank + 1 < 8:                       # garde indispensable (rank 7 → rank+1=8 hors limites)
                    for f in [file - 1, file + 1]:
                        if 0 <= f < 8:
                            p = self.board.piece_at((rank + 1) * 8 + f)
                            if p and p.piece_type == PAWN and p.color == BLACK:
                                threatened = True
                                break
                if not threatened:
                    score += 20

        for sq in self.board.pieces(KNIGHT, BLACK):
            rank, file = sq // 8, sq % 8
            if rank <= 3:
                threatened = False
                if rank - 1 >= 0:                      # garde indispensable (rank 0 → rank-1=-1 hors limites)
                    for f in [file - 1, file + 1]:
                        if 0 <= f < 8:
                            p = self.board.piece_at((rank - 1) * 8 + f)
                            if p and p.piece_type == PAWN and p.color == WHITE:
                                threatened = True
                                break
                if not threatened:
                    score -= 20

        # ── Fou actif sur grande diagonale ────────────────────────────
        for sq in self.board.pieces(BISHOP, WHITE):
            if len(list(self.board.attacks(sq))) >= 6:
                score += 15
        for sq in self.board.pieces(BISHOP, BLACK):
            if len(list(self.board.attacks(sq))) >= 6:
                score -= 15

        return score

    # ==================================================================
    #   SEE récursif complet (Static Exchange Evaluation)
    # ==================================================================

    def see(self, to_sq, attacker_color):
        """
        SEE récursif complet avec minimax rétrograde.
        Retourne le gain net pour attacker_color
        (négatif = mauvaise capture).
        Note : n'inclut pas les attaques en rayon X (simplification acceptable).
        """
        target = self.board.piece_at(to_sq)
        if target is None:
            return 0

        def collect_sorted(color):
            """Valeurs des attaquants de 'color' sur to_sq, triées croissant."""
            vals = []
            for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
                for sq in self.board.pieces(pt, color):
                    if to_sq in self.board.attacks(sq):
                        vals.append(PIECE_VALUES.get(pt, 20000))
            vals.sort()
            return vals

        atk = collect_sorted(attacker_color)
        if not atk:
            return 0

        def_ = collect_sorted(not attacker_color)

        # Simuler la séquence de captures (toujours avec la pièce la moins chère)
        gain = []
        captured_val = PIECE_VALUES[target.piece_type]
        side_lists   = [atk, def_]
        side         = 0

        while side_lists[side]:
            piece_val = side_lists[side].pop(0)
            gain.append(captured_val)
            captured_val = piece_val
            side ^= 1

        if not gain:
            return 0

        # Minimax rétrograde : chaque joueur peut refuser de capturer si perdant
        for d in range(len(gain) - 1, 0, -1):
            gain[d - 1] = gain[d - 1] - max(0, gain[d])

        return gain[0]

    # ==================================================================
    #   Ordonnancement des coups
    # ==================================================================

    def _order_moves(self, moves, depth):
        """
        Priorité :
          1. Coup de la table de transposition
          2. Captures (SEE + MVV-LVA)
          3. Promotions
          4. Killer moves
          5. History heuristic
          6. Coups vers le centre / échecs
        """
        scored = []

        # Coup TT via clé Zobrist
        zobrist_key = self._zobrist(self.board)
        tt_move = None
        if zobrist_key in self.transposition_table:
            try:
                tt_move = chess.Move.from_uci(self.transposition_table[zobrist_key]["best_move"])
            except Exception:
                pass

        for move in moves:
            score = 0

            # Priorité 1 : coup TT
            if tt_move and move == tt_move:
                scored.append((1_000_000, move))
                continue

            # Priorité 2 : captures (SEE)
            if self.board.is_capture(move):
                target   = self.board.piece_at(move.to_square)
                attacker = self.board.piece_at(move.from_square)
                if target and attacker:
                    see_score = self.see(move.to_square, self.board.turn)
                    score += (500_000 + see_score) if see_score >= 0 else see_score

            # Promotions
            if move.promotion:
                score += 900 if move.promotion == QUEEN else 300

            # Priorité 3 : killer moves
            if depth < 64:
                if move == self.killer_moves[depth][0]:
                    score += 1_000
                elif move == self.killer_moves[depth][1]:
                    score += 800

            # Priorité 4 : history heuristic
            move_key = move.uci()
            if move_key in self.history:
                score += self.history[move_key]

            # Bonus échec
            self.board.push(move)
            if self.board.is_check():
                score += 50
            self.board.pop()

            # Bonus centre
            if move.to_square in (27, 28, 35, 36):
                score += 10

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ------------------------------------------------------------------

    def _update_killers(self, move, depth):
        if depth >= 64:
            return
        if self.killer_moves[depth][0] != move:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move

    def _update_history(self, move, depth):
        move_key = move.uci()
        bonus = depth * depth
        self.history[move_key] = self.history.get(move_key, 0) + bonus

    # ==================================================================
    #   Quiescence Search
    # ==================================================================

    def quiescence(self, alpha, beta):
        stand_pat = self.evaluate()
        if self.board.turn == BLACK:
            stand_pat = -stand_pat

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        capture_moves = [m for m in self.board.legal_moves if self.board.is_capture(m)]
        capture_moves.sort(
            key=lambda m: PIECE_VALUES.get(
                self.board.piece_at(m.to_square).piece_type, 0
            ) if self.board.piece_at(m.to_square) else 0,
            reverse=True
        )

        for move in capture_moves:
            self.board.push(move)
            score = -self.quiescence(-beta, -alpha)
            self.board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

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
        R     = 2
        score = -self.negamax(depth - 1 - R, -beta, -beta + 1, ply + 1)
        self.board.pop()

        return beta if score >= beta else None

    # ==================================================================
    #   Negamax principal
    # ==================================================================

    def negamax(self, depth, alpha, beta, ply):
        self.nodes_searched += 1

        # Répétition de position
        if self.board.is_repetition(2):
            return 0

        # Table de transposition (clé Zobrist)
        zobrist_key = self._zobrist(self.board)
        alpha_orig  = alpha

        if zobrist_key in self.transposition_table:
            entry = self.transposition_table[zobrist_key]
            if entry["depth"] >= depth:
                flag  = entry.get("flag", EXACT)
                score = entry["score"]

                if flag == EXACT:
                    return score
                elif flag == LOWERBOUND:
                    alpha = max(alpha, score)
                elif flag == UPPERBOUND:
                    beta = min(beta, score)

                if alpha >= beta:
                    return score

        # Extension en cas d'échec
        in_check = self.board.is_check()
        if depth == 0 and in_check:
            depth = 1

        # Feuille
        if depth == 0:
            if self.board.is_game_over():
                score = self.evaluate()
                return score if self.board.turn == WHITE else -score
            return self.quiescence(alpha, beta)

        # Fin de partie
        if self.board.is_game_over():
            if self.board.is_checkmate():
                return -100_000 + ply
            return 0

        # Null move pruning
        if not in_check and depth >= 3:
            null_result = self._try_null_move(depth, beta, ply)
            if null_result is not None:
                return null_result

        # Génération et ordonnancement des coups
        moves = list(self.board.legal_moves)
        if not moves:
            return 0
        moves = self._order_moves(moves, ply)

        best_move  = moves[0]
        best_score = -10**9
        moves_searched = 0

        for move in moves:
            is_capture = self.board.is_capture(move)
            self.board.push(move)

            if moves_searched == 0:
                # Premier coup : recherche complète
                score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            else:
                # Late Move Reduction logarithmique
                reduction = 0
                if (depth >= 3 and moves_searched >= 4
                        and not in_check
                        and not is_capture
                        and not self.board.is_check()):
                    reduction = max(1, int(math.log(depth) * math.log(moves_searched) / 1.5))

                # Recherche réduite avec fenêtre nulle
                score = -self.negamax(depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)

                # Re-chercher si le coup semble intéressant
                if score > alpha and score < beta:
                    if reduction > 0:
                        score = -self.negamax(depth - 1, -alpha - 1, -alpha, ply + 1)
                    if score > alpha:
                        score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)

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
                break

        # Sauvegarde en table de transposition (mode entraînement)
        if self.train_mode:
            flag = EXACT
            if best_score <= alpha_orig:
                flag = UPPERBOUND
            elif best_score >= beta:
                flag = LOWERBOUND

            # Éviction si la table est trop grande (supprimer ~10% des moins profondes)
            if len(self.transposition_table) >= TT_MAX_SIZE:
                to_delete = sorted(
                    self.transposition_table.keys(),
                    key=lambda k: self.transposition_table[k].get("depth", 0)
                )[:TT_MAX_SIZE // 10]
                for k in to_delete:
                    del self.transposition_table[k]

            self.transposition_table[zobrist_key] = {
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
        """
        Si la promotion en dame choisie par la recherche crée un pat,
        essaie tour, cavalier, fou dans cet ordre.
        Retourne le coup inchangé dans tous les autres cas.
        """
        if move.promotion != QUEEN:
            return move

        board.push(move)
        is_stalemate = board.is_stalemate()
        board.pop()

        if not is_stalemate:
            return move

        for piece in [ROOK, KNIGHT, BISHOP]:
            alt = chess.Move(move.from_square, move.to_square, promotion=piece)
            if alt not in board.legal_moves:
                continue
            board.push(alt)
            still_pat = board.is_stalemate()
            board.pop()
            if not still_pat:
                return alt

        return move  # Pat inévitable, on garde la dame

    # ==================================================================
    #   Point d'entrée principal
    # ==================================================================

    def coup(self, board):
        # ── Livre d'ouvertures (compteur par couleur) ────────────────
        color = board.turn
        if self.opening_moves_played[color] < 12:
            mv = self.get_opening_move(board)
            if mv:
                self.opening_moves_played[color] += 1
                return mv

        # ── Initialisation de la recherche ───────────────────────────
        self.board              = board
        self.killer_moves       = [[None, None] for _ in range(64)]
        self.nodes_searched     = 0
        self._search_start_time = time.time()
        self._time_exceeded     = False

        best_move  = None
        prev_score = 0

        # ── Iterative Deepening + Aspiration Windows ─────────────────
        for d in range(1, self.depth + 1):
            self._time_exceeded = False

            if d >= 2 and best_move is not None:
                # Fenêtre d'aspiration autour du score précédent
                window = 50
                a = prev_score - window
                b = prev_score + window
                score, move = self.negamax_root(d, a, b)

                if not self._time_exceeded:
                    # Fail-low ou fail-high → re-cherche avec fenêtre complète
                    if move is None or score <= a or score >= b:
                        self._time_exceeded = False
                        score, move = self.negamax_root(d)
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

        # ── Fallback (ne devrait pas se produire) ────────────────────
        if best_move is None:
            moves = list(board.legal_moves)
            if moves:
                move_scores = []
                for mv in moves:
                    sc = 0
                    if board.is_capture(mv):
                        target = board.piece_at(mv.to_square)
                        if target:
                            sc = PIECE_VALUES[target.piece_type]
                    move_scores.append((sc, mv))
                move_scores.sort(key=lambda x: x[0], reverse=True)
                best_val  = move_scores[0][0]
                best_move = random.choice([m for s, m in move_scores if s == best_val])
            else:
                raise ValueError("Aucun coup légal trouvé !")

        # ── Promotion anti-pat ────────────────────────────────────────
        best_move = self._smart_promotion(board, best_move)
        return board.san(best_move)

    # ==================================================================
    #   Negamax racine
    # ==================================================================

    def negamax_root(self, depth, alpha=-10**9, beta=10**9):
        """Negamax à la racine – retourne (score, meilleur_coup)."""
        best_score = -10**9
        best_move  = None

        moves = list(self.board.legal_moves)
        if not moves:
            return 0, None

        moves = self._order_moves(moves, 0)

        for move in moves:
            self.board.push(move)
            score = -self.negamax(depth - 1, -beta, -alpha, 1)
            self.board.pop()

            if score > best_score:
                best_score = score
                best_move  = move

            alpha = max(alpha, score)

            if time.time() - self._search_start_time > self.time_limit:
                self._time_exceeded = True
                break

        return best_score, best_move