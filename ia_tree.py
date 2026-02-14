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
    KNIGHT: 320,
    BISHOP: 330,
    ROOK: 500,
    QUEEN: 900,
    KING: 20000,
}

# ==============================================================
#           NOUVEAUTÉ 1: Types de nœuds pour la table
# ==============================================================
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


class TreeIA:
    def __init__(self, depth=2, transpo_file="coups.json", train_mode=True):
        self.depth = depth
        self.transpo_file = transpo_file
        self.train_mode = train_mode

        # Compteur d'ouverture
        self.opening_moves_played = 0

        # ==============================================================
        #   NOUVEAUTÉ 2: Table de transposition optimisée (en RAM)
        # ==============================================================
        # Chargement du JSON uniquement au démarrage
        if os.path.exists(transpo_file):
            try:
                with open(transpo_file, "r") as f:
                    self.transposition_table = json.load(f)
            except:
                self.transposition_table = {}
        else:
            self.transposition_table = {}

        # ==============================================================
        #   NOUVEAUTÉ 3: Killer moves (2 par profondeur)
        # ==============================================================
        self.killer_moves = [[None, None] for _ in range(64)]

        # ==============================================================
        #   NOUVEAUTÉ 4: History heuristic
        # ==============================================================
        self.history = {}

        # Compteur de nœuds pour debug
        self.nodes_searched = 0

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
        """
        CORRECTION FINALE: Évaluation TOUJOURS du point de vue des BLANCS.
        Score positif = bon pour les Blancs
        Score négatif = bon pour les Noirs
        
        Negamax gère l'inversion avec -negamax().
        """

        if self.board.is_checkmate():
            # Qui est mat ? Celui dont c'est le tour
            return -100000 if self.board.turn == WHITE else 100000

        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0

        score = 0

        # Matériel - TOUJOURS Blancs - Noirs
        for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN]:
            score += PIECE_VALUES[pt] * (
                len(self.board.pieces(pt, WHITE))
                - len(self.board.pieces(pt, BLACK))
            )

        # PST Pions
        for sq in self.board.pieces(PAWN, WHITE):
            score += PAWN_TABLE_WHITE[sq]
        for sq in self.board.pieces(PAWN, BLACK):
            score -= PAWN_TABLE_BLACK[sq]

        # PST Cavaliers
        for sq in self.board.pieces(KNIGHT, WHITE):
            score += KNIGHT_TABLE[sq]
        for sq in self.board.pieces(KNIGHT, BLACK):
            score -= KNIGHT_TABLE[sq ^ 56]

        # PST Fous
        for sq in self.board.pieces(BISHOP, WHITE):
            score += BISHOP_TABLE_WHITE[sq]
        for sq in self.board.pieces(BISHOP, BLACK):
            score -= BISHOP_TABLE_BLACK[sq]

        # Paire de fous
        if len(self.board.pieces(BISHOP, WHITE)) >= 2:
            score += 50
        if len(self.board.pieces(BISHOP, BLACK)) >= 2:
            score -= 50

        # PST Tours
        for sq in self.board.pieces(ROOK, WHITE):
            score += ROOK_TABLE_WHITE[sq]
        for sq in self.board.pieces(ROOK, BLACK):
            score -= ROOK_TABLE_BLACK[sq]

        # PST Reines
        for sq in self.board.pieces(QUEEN, WHITE):
            score += QUEEN_TABLE_WHITE[sq]
        for sq in self.board.pieces(QUEEN, BLACK):
            score -= QUEEN_TABLE_BLACK[sq]

        # Roi
        wking = self.board.king(WHITE)
        bking = self.board.king(BLACK)
        
        # Détecter la finale
        total_material = sum(
            PIECE_VALUES[pt] * (len(self.board.pieces(pt, WHITE)) + len(self.board.pieces(pt, BLACK)))
            for pt in [QUEEN, ROOK, BISHOP, KNIGHT]
        )
        is_endgame = total_material < 2600
        
        if is_endgame:
            score += KING_EG_TABLE_WHITE[wking]
            score -= KING_EG_TABLE_BLACK[bking]
        else:
            score += KING_MG_TABLE_WHITE[wking]
            score -= KING_MG_TABLE_BLACK[bking]

        # ==============================================================
        #   ÉVALUATION AVANCÉE (INTÉGRÉE)
        # ==============================================================
        
        # Tours sur colonnes ouvertes
        score += self._evaluate_rook_placement()
        
        # Structure des pions
        score += self._evaluate_pawn_structure()
        
        # Contrôle du centre
        score += self._evaluate_center_control()
        
        # Mobilité
        score += self._evaluate_mobility_fast()
        
        # Sécurité du roi
        score += self._evaluate_king_safety()
        
        # Droits de roque et activité du roi
        if is_endgame:
            score += self._evaluate_endgame_king_activity()
        else:
            score += self._evaluate_castling_rights()

        return score

    def _evaluate_castling_rights(self):
        """Bonus pour droits de roque - point de vue BLANCS"""
        score = 0
        
        # Blancs
        if self.board.has_kingside_castling_rights(WHITE):
            score += 15
        if self.board.has_queenside_castling_rights(WHITE):
            score += 10
        
        # Noirs
        if self.board.has_kingside_castling_rights(BLACK):
            score -= 15
        if self.board.has_queenside_castling_rights(BLACK):
            score -= 10
        
        # Bonus si déjà roqué
        white_king_sq = self.board.king(WHITE)
        black_king_sq = self.board.king(BLACK)
        
        if white_king_sq in [6, 2]:  # g1 ou c1
            score += 30
        if black_king_sq in [62, 58]:  # g8 ou c8
            score -= 30
        
        return score

    def _evaluate_endgame_king_activity(self):
        """Roi actif en finale - point de vue BLANCS"""
        score = 0
        
        white_king = self.board.king(WHITE)
        black_king = self.board.king(BLACK)
        
        # Centralisation
        center = 3.5
        white_dist = abs(white_king % 8 - center) + abs(white_king // 8 - center)
        black_dist = abs(black_king % 8 - center) + abs(black_king // 8 - center)
        
        # Moins de distance = mieux
        score += int((black_dist - white_dist) * 10)
        
        # Roi + pion vs roi
        white_pieces = sum(len(self.board.pieces(pt, WHITE)) for pt in [KNIGHT, BISHOP, ROOK, QUEEN])
        black_pieces = sum(len(self.board.pieces(pt, BLACK)) for pt in [KNIGHT, BISHOP, ROOK, QUEEN])
        
        if white_pieces == 0 and black_pieces == 0:
            black_edge_dist = min(black_king % 8, 7 - black_king % 8, 
                                 black_king // 8, 7 - black_king // 8)
            white_edge_dist = min(white_king % 8, 7 - white_king % 8,
                                 white_king // 8, 7 - white_king // 8)
            score -= black_edge_dist * 5
            score += white_edge_dist * 5
        
        return score

    def _evaluate_rook_placement(self):
        """Tours sur colonnes ouvertes - point de vue BLANCS"""
        score = 0
        
        # Tours blanches
        for rook_sq in self.board.pieces(ROOK, WHITE):
            file = rook_sq % 8
            is_open = True
            is_semi_open = True
            
            for rank in range(8):
                sq = rank * 8 + file
                piece = self.board.piece_at(sq)
                if piece and piece.piece_type == PAWN:
                    is_open = False
                    if piece.color == WHITE:
                        is_semi_open = False
            
            if is_open:
                score += 40
            elif is_semi_open:
                score += 20
        
        # Tours noires
        for rook_sq in self.board.pieces(ROOK, BLACK):
            file = rook_sq % 8
            is_open = True
            is_semi_open = True
            
            for rank in range(8):
                sq = rank * 8 + file
                piece = self.board.piece_at(sq)
                if piece and piece.piece_type == PAWN:
                    is_open = False
                    if piece.color == BLACK:
                        is_semi_open = False
            
            if is_open:
                score -= 40
            elif is_semi_open:
                score -= 20
        
        return score

    def _evaluate_pawn_structure(self):
        """Structure des pions - point de vue BLANCS"""
        score = 0
        
        white_pawns = list(self.board.pieces(PAWN, WHITE))
        black_pawns = list(self.board.pieces(PAWN, BLACK))
        
        # Pions doublés - Blancs
        files = [sq % 8 for sq in white_pawns]
        for file in range(8):
            count = files.count(file)
            if count > 1:
                score -= 15 * (count - 1)
        
        # Pions doublés - Noirs
        files = [sq % 8 for sq in black_pawns]
        for file in range(8):
            count = files.count(file)
            if count > 1:
                score += 15 * (count - 1)
        
        # Pions isolés - Blancs
        for sq in white_pawns:
            file = sq % 8
            has_neighbor = False
            for neighbor_file in [file - 1, file + 1]:
                if 0 <= neighbor_file < 8:
                    if any(p % 8 == neighbor_file for p in white_pawns):
                        has_neighbor = True
                        break
            if not has_neighbor:
                score -= 20
        
        # Pions isolés - Noirs
        for sq in black_pawns:
            file = sq % 8
            has_neighbor = False
            for neighbor_file in [file - 1, file + 1]:
                if 0 <= neighbor_file < 8:
                    if any(p % 8 == neighbor_file for p in black_pawns):
                        has_neighbor = True
                        break
            if not has_neighbor:
                score += 20
        
        # Pions passés - Blancs
        for sq in white_pawns:
            file = sq % 8
            rank = sq // 8
            is_passed = True
            
            for r in range(rank + 1, 8):
                for f in [file - 1, file, file + 1]:
                    if 0 <= f < 8:
                        check_sq = r * 8 + f
                        p = self.board.piece_at(check_sq)
                        if p and p.piece_type == PAWN and p.color == BLACK:
                            is_passed = False
                            break
            
            if is_passed:
                advancement = rank
                score += 30 + advancement * 10
        
        # Pions passés - Noirs
        for sq in black_pawns:
            file = sq % 8
            rank = sq // 8
            is_passed = True
            
            for r in range(0, rank):
                for f in [file - 1, file, file + 1]:
                    if 0 <= f < 8:
                        check_sq = r * 8 + f
                        p = self.board.piece_at(check_sq)
                        if p and p.piece_type == PAWN and p.color == WHITE:
                            is_passed = False
                            break
            
            if is_passed:
                advancement = 7 - rank
                score -= 30 + advancement * 10
        
        return score

    def _evaluate_center_control(self):
        """Contrôle du centre - point de vue BLANCS"""
        score = 0
        center_squares = [27, 28, 35, 36]
        
        for sq in center_squares:
            # Attaques
            if self.board.is_attacked_by(WHITE, sq):
                score += 5
            if self.board.is_attacked_by(BLACK, sq):
                score -= 5
            
            # Occupation
            piece = self.board.piece_at(sq)
            if piece:
                if piece.color == WHITE:
                    score += 10
                else:
                    score -= 10
        
        return score

    def _evaluate_mobility_fast(self):
        """Mobilité - point de vue BLANCS"""
        white_attacks = 0
        black_attacks = 0
        
        for piece_type in [QUEEN, ROOK, BISHOP, KNIGHT]:
            for sq in self.board.pieces(piece_type, WHITE):
                attacks = len(list(self.board.attacks(sq)))
                white_attacks += attacks
            
            for sq in self.board.pieces(piece_type, BLACK):
                attacks = len(list(self.board.attacks(sq)))
                black_attacks += attacks
        
        return (white_attacks - black_attacks) * 2

    def _evaluate_king_safety(self):
        """Sécurité du roi - point de vue BLANCS"""
        score = 0
        
        # Roi blanc
        white_king_sq = self.board.king(WHITE)
        if white_king_sq is not None:
            attackers = 0
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[white_king_sq]):
                piece = self.board.piece_at(sq)
                if piece and piece.color == BLACK:
                    attackers += 1
            
            pawn_shield = 0
            file = white_king_sq % 8
            rank = white_king_sq // 8
            ranks = [rank+1, rank+2] if rank < 6 else [rank-1]
            
            for r in ranks:
                if 0 <= r < 8:
                    for f in (file-1, file, file+1):
                        if 0 <= f < 8:
                            sq = r*8 + f
                            p = self.board.piece_at(sq)
                            if p and p.piece_type == PAWN and p.color == WHITE:
                                pawn_shield += 1
            
            weak_squares = 0
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[white_king_sq]):
                if not self.board.is_attacked_by(WHITE, sq):
                    weak_squares += 1
            
            score += -25 * attackers + 12 * pawn_shield - 8 * weak_squares
        
        # Roi noir
        black_king_sq = self.board.king(BLACK)
        if black_king_sq is not None:
            attackers = 0
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[black_king_sq]):
                piece = self.board.piece_at(sq)
                if piece and piece.color == WHITE:
                    attackers += 1
            
            pawn_shield = 0
            file = black_king_sq % 8
            rank = black_king_sq // 8
            ranks = [rank-1, rank-2] if rank > 1 else [rank+1]
            
            for r in ranks:
                if 0 <= r < 8:
                    for f in (file-1, file, file+1):
                        if 0 <= f < 8:
                            sq = r*8 + f
                            p = self.board.piece_at(sq)
                            if p and p.piece_type == PAWN and p.color == BLACK:
                                pawn_shield += 1
            
            weak_squares = 0
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[black_king_sq]):
                if not self.board.is_attacked_by(BLACK, sq):
                    weak_squares += 1
            
            score -= (-25 * attackers + 12 * pawn_shield - 8 * weak_squares)
        
        return score

    # ==============================================================
    #   NOUVEAUTÉ 3 & 4: Ordre des coups avec Killers + History
    # ==============================================================

    def _order_moves(self, moves, depth):
        """
        Ordonne les coups pour maximiser l'élagage alpha-beta.
        Priorité:
        1. Coups de la table de transposition
        2. Captures (MVV-LVA)
        3. Killer moves
        4. History heuristic
        """
        scored = []
        
        # Récupérer le meilleur coup de la table de transposition
        fen = self.board.fen()
        tt_move = None
        if fen in self.transposition_table:
            try:
                tt_move = chess.Move.from_uci(self.transposition_table[fen]["best_move"])
            except:
                pass

        for move in moves:
            score = 0

            # ==============================================================
            #   PRIORITÉ 1: Coup de la table de transposition
            # ==============================================================
            if tt_move and move == tt_move:
                score += 1000000
                scored.append((score, move))
                continue

            # ==============================================================
            #   PRIORITÉ 2: Captures (MVV-LVA)
            # ==============================================================
            if self.board.is_capture(move):
                target = self.board.piece_at(move.to_square)
                attacker = self.board.piece_at(move.from_square)
                if target:
                    # Score TRÈS élevé pour les captures
                    score += PIECE_VALUES[target.piece_type] * 100
                    if attacker:
                        score -= PIECE_VALUES[attacker.piece_type] // 10

            # Promotions
            if move.promotion:
                if move.promotion == QUEEN:
                    score += 900
                else:
                    score += 300

            # ==============================================================
            #   PRIORITÉ 3: Killer moves
            # ==============================================================
            if depth < 64:
                if move == self.killer_moves[depth][0]:
                    score += 1000  # Bien moins que captures
                elif move == self.killer_moves[depth][1]:
                    score += 800

            # ==============================================================
            #   PRIORITÉ 4: History heuristic
            # ==============================================================
            move_key = move.uci()
            if move_key in self.history:
                score += self.history[move_key]

            # Échec
            self.board.push(move)
            if self.board.is_check():
                score += 50
            self.board.pop()

            # Bonus pour coups vers le centre
            if move.to_square in (27, 28, 35, 36):
                score += 10

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for s, m in scored]

    # ==============================================================
    #   NOUVEAUTÉ 3: Mise à jour des killer moves
    # ==============================================================
    def _update_killers(self, move, depth):
        """Ajoute un coup killer à cette profondeur"""
        if depth >= 64:
            return
        
        if self.killer_moves[depth][0] != move:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move

    # ==============================================================
    #   NOUVEAUTÉ 4: Mise à jour de l'history heuristic
    # ==============================================================
    def _update_history(self, move, depth):
        """Augmente le score history d'un coup qui cause une coupure"""
        move_key = move.uci()
        bonus = depth * depth
        if move_key not in self.history:
            self.history[move_key] = 0
        self.history[move_key] += bonus

    # ==============================================================
    #   FIX CRITIQUE: Quiescence Search corrigée
    # ==============================================================

    def quiescence(self, alpha, beta):
        """
        Recherche de quiescence - continue à analyser les captures
        jusqu'à atteindre une position calme
        """
        stand_pat = self.evaluate()
        # evaluate() est du point de vue des BLANCS, inverser si Noirs jouent
        if self.board.turn == BLACK:
            stand_pat = -stand_pat
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Ne considérer que les captures
        capture_moves = [m for m in self.board.legal_moves if self.board.is_capture(m)]
        
        # Ordonnancement simplifié pour la quiescence
        capture_moves = sorted(capture_moves, 
                              key=lambda m: PIECE_VALUES.get(
                                  self.board.piece_at(m.to_square).piece_type, 0
                              ) if self.board.piece_at(m.to_square) else 0,
                              reverse=True)
        
        for move in capture_moves:
            self.board.push(move)
            score = -self.quiescence(-beta, -alpha)
            self.board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha

    # ==============================================================
    #   NOUVEAUTÉ 6: Null Move Pruning
    # ==============================================================
    def _try_null_move(self, depth, beta, ply):
        """
        Null move pruning: si même en donnant un coup gratuit à l'adversaire
        on dépasse beta, on peut couper.
        """
        if depth < 3:
            return None
        
        # Conditions: pas en échec, pas en finale de pions
        if self.board.is_check():
            return None
        
        # Vérifier qu'on n'est pas en finale de pions
        non_pawn_material = sum(
            len(self.board.pieces(pt, self.board.turn))
            for pt in [KNIGHT, BISHOP, ROOK, QUEEN]
        )
        if non_pawn_material == 0:
            return None
        
        # Faire un "null move" (passer son tour)
        self.board.push(chess.Move.null())
        R = 2  # Réduction
        score = -self.negamax(depth - 1 - R, -beta, -beta + 1, ply + 1)
        self.board.pop()
        
        if score >= beta:
            return beta
        
        return None

    # ==============================================================
    #   NOUVEAUTÉ 7: Negamax (plus propre que minimax)
    # ==============================================================

    def negamax(self, depth, alpha, beta, ply):
        """
        Negamax avec alpha-beta, plus propre que minimax.
        ply = distance depuis la racine (pour mate distance pruning)
        """
        self.nodes_searched += 1
        
        # ==============================================================
        #   DÉTECTION DE RÉPÉTITION
        # ==============================================================
        # Pénaliser les répétitions de position
        if self.board.is_repetition(2):
            return 0  # Nulle par répétition
        
        # ==============================================================
        #   NOUVEAUTÉ 1: Table de transposition avec types de nœuds
        # ==============================================================
        fen = self.board.fen()
        alpha_orig = alpha

        if fen in self.transposition_table:
            entry = self.transposition_table[fen]
            if entry["depth"] >= depth:
                flag = entry.get("flag", EXACT)
                score = entry["score"]
                
                if flag == EXACT:
                    return score
                elif flag == LOWERBOUND:
                    alpha = max(alpha, score)
                elif flag == UPPERBOUND:
                    beta = min(beta, score)
                
                if alpha >= beta:
                    return score

        # Extension si échec (mais limitée)
        in_check = self.board.is_check()
        if depth == 0 and in_check:
            depth = 1

        # Profondeur 0: quiescence
        if depth == 0:
            if self.board.is_game_over():
                score = self.evaluate()
                # evaluate() est du point de vue des BLANCS
                # Si c'est au tour des NOIRS, inverser
                return score if self.board.turn == WHITE else -score
            else:
                return self.quiescence(alpha, beta)

        # Game over
        if self.board.is_game_over():
            if self.board.is_checkmate():
                return -100000 + ply  # Préférer les mats plus rapides
            return 0

        # ==============================================================
        #   NOUVEAUTÉ 6: Null Move Pruning
        # ==============================================================
        if not in_check and depth >= 3:
            null_result = self._try_null_move(depth, beta, ply)
            if null_result is not None:
                return null_result

        # Génération et ordonnancement des coups
        moves = list(self.board.legal_moves)
        if not moves:
            return 0
        
        moves = self._order_moves(moves, ply)

        best_move = moves[0]
        best_score = -10**9

        # ==============================================================
        #   NOUVEAUTÉ 8: Late Move Reduction (LMR)
        # ==============================================================
        moves_searched = 0

        for i, move in enumerate(moves):
            # Sauvegarder si c'est une capture (pour killers)
            is_capture = self.board.is_capture(move)
            
            self.board.push(move)

            # PVS (Principal Variation Search) + LMR
            if moves_searched == 0:
                # Premier coup: recherche complète
                score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            else:
                # ==============================================================
                #   LMR: Réduction pour les coups tard dans la liste
                # ==============================================================
                reduction = 0
                if (depth >= 3 and moves_searched >= 4 and 
                    not in_check and 
                    not is_capture and 
                    not self.board.is_check()):
                    reduction = 1
                
                # Recherche réduite avec fenêtre nulle
                score = -self.negamax(depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                
                # Si le coup semble bon, re-chercher complètement
                if score > alpha and score < beta:
                    if reduction > 0:
                        score = -self.negamax(depth - 1, -alpha - 1, -alpha, ply + 1)
                    if score > alpha:
                        score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)

            self.board.pop()
            moves_searched += 1

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            
            # Coupure beta
            if alpha >= beta:
                # ==============================================================
                #   NOUVEAUTÉ 3 & 4: Mise à jour killers et history
                # ==============================================================
                if not is_capture:
                    self._update_killers(move, ply)
                    self._update_history(move, depth)
                break

        # ==============================================================
        #   NOUVEAUTÉ 1: Sauvegarde avec type de nœud
        # ==============================================================
        if self.train_mode:
            flag = EXACT
            if best_score <= alpha_orig:
                flag = UPPERBOUND
            elif best_score >= beta:
                flag = LOWERBOUND
            
            self.transposition_table[fen] = {
                "best_move": best_move.uci(),
                "score": best_score,
                "depth": depth,
                "flag": flag
            }

        return best_score

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
        if self.opening_moves_played < 12:
            mv = self.get_opening_move(board)
            if mv:
                self.opening_moves_played += 1
                return mv

        # Recherche principale
        self.board = board
        
        # Réinitialiser les killer moves
        self.killer_moves = [[None, None] for _ in range(64)]

        # ==============================================================
        #   RECHERCHE DIRECTE (sans iterative deepening bugué)
        # ==============================================================
        score, best_move = self.negamax_root(self.depth)

        if best_move is None:
            moves = list(board.legal_moves)
            if moves:
                move_scores = []
                for move in moves:
                    sc = 0
                    if self.board.is_capture(move):
                        target = self.board.piece_at(move.to_square)
                        if target:
                            sc = PIECE_VALUES[target.piece_type]
                    move_scores.append((sc, move))
                
                move_scores.sort(key=lambda x: x[0], reverse=True)
                best_score = move_scores[0][0]
                best_moves = [m for s, m in move_scores if s == best_score]
                best_move = random.choice(best_moves)
            else:
                raise ValueError("Aucun coup trouvé !")

        return board.san(best_move)
    
    def negamax_root(self, depth):
        """Negamax à la racine - retourne (score, meilleur_coup)"""
        alpha = -10**9
        beta = 10**9
        best_score = -10**9
        best_move = None
        
        moves = list(self.board.legal_moves)
        if not moves:
            return 0, None
        
        # Ordonner les coups
        moves = self._order_moves(moves, 0)
        
        for move in moves:
            self.board.push(move)
            score = -self.negamax(depth - 1, -beta, -alpha, 1)
            self.board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
        
        return best_score, best_move