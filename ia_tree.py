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

        # Compteur d'ouverture
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

        # NOUVEAU: Bonus pour la paire de fous
        white_bishops = len(self.board.pieces(BISHOP, WHITE))
        black_bishops = len(self.board.pieces(BISHOP, BLACK))
        if white_bishops >= 2:
            score += 50
        if black_bishops >= 2:
            score -= 50

        # Tours PST
        for sq in self.board.pieces(ROOK, WHITE):
            score += ROOK_TABLE_WHITE[sq]
        for sq in self.board.pieces(ROOK, BLACK):
            score -= ROOK_TABLE_BLACK[sq]

        # NOUVEAU: Tours sur colonnes ouvertes/semi-ouvertes
        score += self._evaluate_rook_placement()

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

        is_endgame = total_material < 2000

        if is_endgame:
            score += KING_EG_TABLE_WHITE[wking]
            score -= KING_EG_TABLE_BLACK[bking]
            # NOUVEAU: En finale, activer le roi et centraliser
            score += self._evaluate_endgame_king_activity()
        else:
            score += KING_MG_TABLE_WHITE[wking]
            score -= KING_MG_TABLE_BLACK[bking]
            # NOUVEAU: Bonus pour le roque en milieu de partie
            score += self._evaluate_castling_rights()

        # NOUVEAU: Structure des pions améliorée
        score += self._evaluate_pawn_structure()

        # NOUVEAU: Contrôle du centre
        score += self._evaluate_center_control()

        # NOUVEAU: Mobilité
        score += self._evaluate_mobility()

        # NOUVEAU: Sécurité du roi
        score += self._evaluate_king_safety()

        return score

    # ============== NOUVELLES FONCTIONS D'ÉVALUATION ==============

    def _evaluate_castling_rights(self):
        """NOUVEAU: Bonus pour avoir encore les droits de roque"""
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
        
        # Bonus si déjà roqué (roi en sécurité)
        white_king_sq = self.board.king(WHITE)
        black_king_sq = self.board.king(BLACK)
        
        if white_king_sq in [6, 2]:  # g1 ou c1
            score += 30
        if black_king_sq in [62, 58]:  # g8 ou c8
            score -= 30
            
        return score

    def _evaluate_endgame_king_activity(self):
        """NOUVEAU: En finale, le roi doit être actif et centralisé"""
        score = 0
        white_king = self.board.king(WHITE)
        black_king = self.board.king(BLACK)
        
        # Centralisation (distance au centre)
        center = 3.5
        white_dist = abs(white_king % 8 - center) + abs(white_king // 8 - center)
        black_dist = abs(black_king % 8 - center) + abs(black_king // 8 - center)
        
        score += int((black_dist - white_dist) * 10)
        
        # Si roi + pion vs roi, pousser le roi adverse vers le bord
        white_pieces = sum(len(self.board.pieces(pt, WHITE)) for pt in [KNIGHT, BISHOP, ROOK, QUEEN])
        black_pieces = sum(len(self.board.pieces(pt, BLACK)) for pt in [KNIGHT, BISHOP, ROOK, QUEEN])
        
        if white_pieces == 0 and black_pieces == 0:  # Finale roi + pions
            black_edge_dist = min(black_king % 8, 7 - black_king % 8, 
                                  black_king // 8, 7 - black_king // 8)
            white_edge_dist = min(white_king % 8, 7 - white_king % 8,
                                  white_king // 8, 7 - white_king // 8)
            score -= black_edge_dist * 5
            score += white_edge_dist * 5
            
        return score

    def _evaluate_rook_placement(self):
        """NOUVEAU: Bonus pour tours sur colonnes ouvertes"""
        score = 0
        
        for color in [WHITE, BLACK]:
            for rook_sq in self.board.pieces(ROOK, color):
                file = rook_sq % 8
                is_open = True
                is_semi_open = True
                
                # Vérifier s'il y a des pions sur cette colonne
                for rank in range(8):
                    sq = rank * 8 + file
                    piece = self.board.piece_at(sq)
                    if piece and piece.piece_type == PAWN:
                        is_open = False
                        if piece.color == color:
                            is_semi_open = False
                
                bonus = 0
                if is_open:
                    bonus = 40
                elif is_semi_open:
                    bonus = 20
                
                if color == WHITE:
                    score += bonus
                else:
                    score -= bonus
                    
        return score

    def _is_passed_pawn(self, square, color):
        """NOUVEAU: Détecte si un pion est passé"""
        file = square % 8
        rank = square // 8

        for cf in [file - 1, file, file + 1]:
            if 0 <= cf < 8:
                if color == WHITE:
                    for r in range(rank + 1, 8):
                        sq = r * 8 + cf
                        piece = self.board.piece_at(sq)
                        if piece and piece.piece_type == PAWN and piece.color == BLACK:
                            return False
                else:
                    for r in range(rank - 1, -1, -1):
                        sq = r * 8 + cf
                        piece = self.board.piece_at(sq)
                        if piece and piece.piece_type == PAWN and piece.color == WHITE:
                            return False
        return True

    def _evaluate_pawn_structure(self):
        """NOUVEAU: Évalue la structure des pions"""
        score = 0

        white_pawn_files = {}
        black_pawn_files = {}

        for sq in self.board.pieces(PAWN, WHITE):
            f = sq % 8
            white_pawn_files[f] = white_pawn_files.get(f, 0) + 1

        for sq in self.board.pieces(PAWN, BLACK):
            f = sq % 8
            black_pawn_files[f] = black_pawn_files.get(f, 0) + 1

        # Pions doublés
        for f, c in white_pawn_files.items():
            if c > 1:
                score -= 20 * (c - 1)
        for f, c in black_pawn_files.items():
            if c > 1:
                score += 20 * (c - 1)

        # Pions isolés
        for f in white_pawn_files:
            if f - 1 not in white_pawn_files and f + 1 not in white_pawn_files:
                score -= 15
        for f in black_pawn_files:
            if f - 1 not in black_pawn_files and f + 1 not in black_pawn_files:
                score += 15

        # Pions passés (bonus progressif)
        for sq in self.board.pieces(PAWN, WHITE):
            if self._is_passed_pawn(sq, WHITE):
                rank = sq // 8
                score += 30 + (rank * 15)
        for sq in self.board.pieces(PAWN, BLACK):
            if self._is_passed_pawn(sq, BLACK):
                rank = sq // 8
                score -= 30 + ((7 - rank) * 15)

        return score

    def _evaluate_center_control(self):
        """NOUVEAU: Évalue le contrôle du centre"""
        score = 0
        center = [27, 28, 35, 36]  # e4,d4,e5,d5
        ext_center = [18,19,20,21,26,29,34,37,42,43,44,45]

        for sq in center:
            p = self.board.piece_at(sq)
            if p:
                if p.color == WHITE: score += 30
                else: score -= 30

        for sq in ext_center:
            p = self.board.piece_at(sq)
            if p:
                if p.color == WHITE: score += 10
                else: score -= 10

        return score

    def _evaluate_mobility(self):
        """NOUVEAU: Évalue la mobilité (nombre de coups légaux)"""
        current_mobility = len(list(self.board.legal_moves))
        
        score = 0
        if self.board.turn == WHITE:
            score += current_mobility * 2
        else:
            score -= current_mobility * 2
            
        return score

    def _evaluate_king_safety(self):
        """NOUVEAU: Évalue la sécurité du roi"""
        score = 0
        
        for color in (WHITE, BLACK):
            king_sq = self.board.king(color)
            if king_sq is None:
                continue
            
            # Attaquants autour du roi
            attackers = 0
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
                piece = self.board.piece_at(sq)
                if piece and piece.color != color:
                    attackers += 1
            
            # Bouclier de pions
            pawn_shield = 0
            file = king_sq % 8
            rank = king_sq // 8
            
            if color == WHITE:
                ranks = [rank+1, rank+2] if rank < 6 else [rank-1]
            else:
                ranks = [rank-1, rank-2] if rank > 1 else [rank+1]
            
            for r in ranks:
                if 0 <= r < 8:
                    for f in (file-1, file, file+1):
                        if 0 <= f < 8:
                            sq = r*8 + f
                            p = self.board.piece_at(sq)
                            if p and p.piece_type == PAWN and p.color == color:
                                pawn_shield += 1
            
            # Cases faibles autour du roi
            weak_squares = 0
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
                if not self.board.is_attacked_by(color, sq):
                    weak_squares += 1
            
            delta = -25 * attackers + 12 * pawn_shield - 8 * weak_squares
            
            if color == WHITE:
                score += delta
            else:
                score -= delta
                
        return score

    # ==============================================================
    #                        ORDRE DES COUPS
    # ==============================================================

    def _order_moves(self, moves, maximizing):
        scored = []

        for move in moves:
            score = 0

            # Captures (MVV-LVA amélioré)
            if self.board.is_capture(move):
                target = self.board.piece_at(move.to_square)
                attacker = self.board.piece_at(move.from_square)
                if target:
                    score += PIECE_VALUES[target.piece_type] * 10
                    # NOUVEAU: Préférence pour attaque avec pièce légère
                    if attacker:
                        score -= PIECE_VALUES[attacker.piece_type] // 100

            # Promotions
            if move.promotion:
                if move.promotion == QUEEN:
                    score += 900
                else:
                    score += 300

            # Échec
            self.board.push(move)
            if self.board.is_check():
                score += 50
            self.board.pop()

            # NOUVEAU: Bonus pour coups vers le centre
            if move.to_square in (27,28,35,36):
                score += 10

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for s, m in scored]

    # ==============================================================
    #                  NOUVEAU: QUIESCENCE SEARCH
    # ==============================================================

    def quiescence(self, alpha, beta):
        """
        NOUVEAU: Recherche de quiescence - continue à analyser les captures
        jusqu'à atteindre une position calme
        """
        stand_pat = self.evaluate()
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Ne considérer que les captures
        capture_moves = [m for m in self.board.legal_moves if self.board.is_capture(m)]
        
        for move in self._order_moves(capture_moves, self.board.turn == WHITE):
            self.board.push(move)
            score = -self.quiescence(-beta, -alpha)
            self.board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha

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

        # NOUVEAU: Extension si position critique (échec)
        if depth == 0 and self.board.is_check():
            depth = 1

        # NOUVEAU: Quiescence search au lieu d'évaluation statique à profondeur 0
        if depth == 0:
            if self.board.is_game_over():
                return self.evaluate(), None
            else:
                val = self.quiescence(alpha, beta) if maximizing else -self.quiescence(-beta, -alpha)
                return val, None

        # Feuille (game over)
        if self.board.is_game_over():
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

        # Ouvertures (AMÉLIORÉ: jusqu'à 12 coups au lieu de 8)
        if self.opening_moves_played < 12:
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
