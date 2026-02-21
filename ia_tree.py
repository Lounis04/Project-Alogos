import json
import math
import os
import random
import time
from collections import Counter
from typing import Optional
import chess
import chess.polyglot
from chess import (Board, Move, PieceType, Color,PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK,)

from PST import (PAWN_TABLE_WHITE,   PAWN_TABLE_BLACK,KNIGHT_TABLE,BISHOP_TABLE_WHITE, BISHOP_TABLE_BLACK,ROOK_TABLE_WHITE,   ROOK_TABLE_BLACK,QUEEN_TABLE_WHITE,  QUEEN_TABLE_BLACK,KING_MG_TABLE_WHITE, KING_MG_TABLE_BLACK,KING_EG_TABLE_WHITE, KING_EG_TABLE_BLACK,)
from Ouvertures import OPENING_BOOK

#Constantes globales 
PIECE_VALUES: dict[PieceType, int] = {PAWN: 100, KNIGHT: 320, BISHOP: 330,ROOK: 500, QUEEN:  900, KING:  20000,}

# Codes de flag pour la table de transpositions
EXACT:      int = 0   
LOWERBOUND: int = 1   
UPPERBOUND: int = 2   

# Paramètres de recherche
TT_MAX_SIZE:      int          = 500_000
FUTILITY_MARGINS: dict[int, int] = {1: 320, 2: 500, 3: 900}
DELTA_MARGIN:     int          = 200
RAZORING_MARGIN:  int          = 300
PROBCUT_MARGIN:   int          = 100
SINGULAR_MARGIN:  int          = 50
CONTEMPT:         int          = 25
LMP_COUNTS:       dict[int, int] = {1: 5, 2: 10, 3: 18}
MAX_CHECK_EXT:    int          = 6


#Fonctions utilitaires de module
def _normalize_fen(fen: str) -> str:
    """Supprime la composante en-passant du FEN pour normaliser les clés du livre.
    Entrée  : fen,chaîne FEN complète (6 champs séparés par des espaces).
    Sortie  : FEN avec le 4ème champ (en-passant) remplacé par '-'."""
    parts: list[str] = fen.split()
    if len(parts) >= 6:
        parts[3] = '-'
    return ' '.join(parts)

# Pré-traitement du livre d'ouvertures : clés normalisées au démarrage
OPENING_BOOK_NORMALIZED: dict[str, list[str]] = {_normalize_fen(k): v for k, v in OPENING_BOOK.items()}

def _build_passed_masks() -> tuple[list[int], list[int]]:
    """Précalcule les masques bitboard pour détecter les pions passés.Pour chaque case, le masque donne les cases devant le pion (même colonne± 1) dans la direction d'avancement, côté blanc et côté noir.
    Entrée  : aucune.
    Sortie  : tuple (PASSED_MASK_WHITE, PASSED_MASK_BLACK) deux listes de64 entiers (bitboards 64-bit)."""
    w: list[int] = [0] * 64
    b: list[int] = [0] * 64
    for sq in range(64):
        rank: int = sq >> 3
        file: int = sq & 7
        lo:   int = max(0, file - 1)
        hi:   int = min(7, file + 1)
        for r in range(rank + 1, 8):
            for f in range(lo, hi + 1):
                w[sq] |= 1 << (r * 8 + f)
        for r in range(rank):
            for f in range(lo, hi + 1):
                b[sq] |= 1 << (r * 8 + f)
    return w, b


PASSED_MASK_WHITE: list[int]
PASSED_MASK_BLACK: list[int]
PASSED_MASK_WHITE, PASSED_MASK_BLACK = _build_passed_masks()
FILE_MASK: list[int] = [sum(1 << (r * 8 + f) for r in range(8)) for f in range(8)]

#Project Alogos sous stéroides , niveau boss de Dark Souls
class TreeIA:
    """Intelligence artificielle d'échecs reposant sur Negamax + alpha-bêta.
    Entrée  : depth,transpo_file,train_mode   
    Sortie  : objet TreeIA prêt à jouer via coup(board)."""
    def __init__(self,depth:int  = 2,transpo_file: str  = "coups.json",train_mode:bool = True) -> None:
        self.depth:        int  = depth
        self.transpo_file: str  = transpo_file
        self.train_mode:   bool = train_mode
        self.opening_moves_played: dict[bool, int] = {True: 0, False: 0}

        # Chargement de la table de transpositions depuis le disque
        if os.path.exists(transpo_file):
            try:
                with open(transpo_file) as f:
                    tt: dict = json.load(f)
                # JSON convertit les clés entières en str 
                if isinstance(tt, dict):
                    converted: dict[int | str, dict] = {}
                    for k, v in tt.items():
                        try:
                            converted[int(k)] = v
                        except (ValueError, TypeError):
                            converted[k] = v  # Clé FEN conservée telle quelle
                    self.transposition_table: dict[int | str, dict] = converted
                else:
                    self.transposition_table = {}
            except Exception:
                self.transposition_table = {}
        else:
            self.transposition_table = {}

        self.killer_moves:   list[list[Optional[Move]]] = [[None, None] for _ in range(64)]
        self.history:        dict[tuple[int, int], int] = {}
        self.counter_moves:  list[list[Optional[Move]]] = [[None] * 64 for _ in range(64)]
        self.nodes_searched:      int   = 0
        self.time_limit:          float = 5.0
        self._search_start_time:  float = 0.0
        self._time_exceeded:      bool  = False
        self._last_move:          Optional[Move] = None
        self.board: Board  # initialisé dans coup()


    # Utilitaires internes
    @staticmethod
    def _zobrist(board: Board) -> int | str:
        """Calcule la clé Zobrist du plateau.
        Entrée  : board ,plateau chess.Board courant.
        Sortie  : entier (hash Zobrist polyglot) ou chaîne FEN en cas d'erreur."""
        try:
            return int(chess.polyglot.zobrist_hash(board))
        except Exception:
            return board.fen()

    def save_transpo(self) -> None:
        """Sauvegarde la table de transpositions sur disque en mode train_mode.Utilise une écriture atomique (fichier .tmp + renommage).
        Entrée  : aucune (utilise self.transpo_file et self.train_mode).
        Sortie  : aucune (effet de bord : écriture JSON sur disque)."""
        if not self.train_mode:
            return
        try:
            tmp: str = self.transpo_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.transposition_table, f, separators=(',', ':'))
            os.replace(tmp, self.transpo_file)
        except Exception as e:
            print(f"[save_transpo] Erreur : {e}")


    # Évaluation statique
    def evaluate(self) -> int:
        """Calcule le score d'évaluation statique du plateau depuis la perspective des Blancs (positif = avantage blanc, négatif = avantage noir).
        Prend en compte : matériel, PST, paire de fous, structure de pions,contrôle du centre, mobilité, sécurité du roi, fin de partie,tactiques, clouages.
        Entrée  : aucune (utilise self.board).
        Sortie  : score entier en centipawns."""
        b = self.board
        wp: dict[PieceType, chess.SquareSet] = {pt: b.pieces(pt, WHITE) for pt in range(1, 7)}
        bp: dict[PieceType, chess.SquareSet] = {pt: b.pieces(pt, BLACK) for pt in range(1, 7)}

        score: int = 0
        mat_w: int = 0
        mat_b: int = 0

        for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN]:
            v: int = PIECE_VALUES[pt]
            nw: int = len(wp[pt])
            nb: int = len(bp[pt])
            score += v * (nw - nb)
            if pt != PAWN:
                mat_w += v * nw
                mat_b += v * nb

        total_material:   int = mat_w + mat_b
        is_endgame:       bool = total_material < 2600
        material_balance: int = score

        # Tables positionnelles (PST)
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

        # Bonus paire de fous
        if len(wp[BISHOP]) >= 2: score += 50
        if len(bp[BISHOP]) >= 2: score -= 50

        # Roi (milieu ou finale)
        wking: int = b.king(WHITE)
        bking: int = b.king(BLACK)
        if is_endgame:
            score += KING_EG_TABLE_WHITE[wking] - KING_EG_TABLE_BLACK[bking]
        else:
            score += KING_MG_TABLE_WHITE[wking] - KING_MG_TABLE_BLACK[bking]

        wpbb: int = int(wp[PAWN])
        bpbb: int = int(bp[PAWN])

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

        # Bonus de conversion en finale (avantage matériel significatif)
        if abs(material_balance) > 200:
            bonus: int = max(0, (6400 - total_material) // 200)
            score += bonus if material_balance > 200 else -bonus

        return score

    def _eval_castling(self, wk: int, bk: int) -> int:
        """Évalue les droits de roque et la position du roi après roque.
        Entrée  : wk case du roi blanc (0-63) ; bk case du roi noir (0-63).
        Sortie  : score entier (positif = avantage blanc)."""
        score: int = 0
        b = self.board
        if b.has_kingside_castling_rights(WHITE):  score += 15
        if b.has_queenside_castling_rights(WHITE): score += 10
        if b.has_kingside_castling_rights(BLACK):  score -= 15
        if b.has_queenside_castling_rights(BLACK): score -= 10
        if wk in (6, 2):   score += 30   # Roi blanc déjà roqué
        if bk in (62, 58): score -= 30   # Roi noir déjà roqué
        return score

    def _eval_endgame_king(self,wk:   int,bk:   int,wpbb: int,bpbb: int,) -> int:
        """Évalue la position des rois en finale : centralisation, opposition,activité dans les finales de pions passés.
        Entrée  : wk   case du roi blanc ; bk  case du roi noir ; wpbb bitboard des pions blancs ; bpbb bitboard des pions noirs.
        Sortie  : score entier (positif = avantage blanc)."""
        score: int = 0
        b = self.board

        center: float = 3.5
        wd: float = abs((wk & 7) - center) + abs((wk >> 3) - center)
        bd: float = abs((bk & 7) - center) + abs((bk >> 3) - center)
        score += int((bd - wd) * 25)

        # Finale roi + pions uniquement : pousser le roi adverse vers le bord
        if not (b.occupied_co[WHITE] & ~b.pawns & ~b.kings) and \
           not (b.occupied_co[BLACK] & ~b.pawns & ~b.kings):
            be: int = min(bk & 7, 7 - (bk & 7), bk >> 3, 7 - (bk >> 3))
            we: int = min(wk & 7, 7 - (wk & 7), wk >> 3, 7 - (wk >> 3))
            score += (we - be) * 5

        # Opposition directe et indirecte
        wr: int = wk >> 3;  wf: int = wk & 7
        br: int = bk >> 3;  bf: int = bk & 7
        if (wr == br and abs(wf - bf) == 2) or (wf == bf and abs(wr - br) == 2):
            score += 30 if b.turn == BLACK else -30
        elif abs(wf - bf) % 2 == 0 and abs(wr - br) % 2 == 0 and abs(wf - bf) + abs(wr - br) > 2:
            score += 15 if b.turn == BLACK else -15

        extra_w: int = 0 if b.turn == WHITE else 1
        extra_b: int = 0 if b.turn == BLACK else 1

        # Pions passés blancs hors de portée du roi noir
        for sq in b.pieces(PAWN, WHITE):
            if not (PASSED_MASK_WHITE[sq] & bpbb):
                rank: int = sq >> 3;  file: int = sq & 7
                steps: int = 7 - rank - (1 if rank == 1 else 0)
                if max(abs((bk >> 3) - 7), abs((bk & 7) - file)) > steps + extra_w:
                    score += 200
                if rank >= 4:
                    key_rank: int = min(7, rank + 2)
                    for kf in range(max(0, file - 1), min(7, file + 1) + 1):
                        if wk == key_rank * 8 + kf:
                            score += 50

        # Pions passés noirs hors de portée du roi blanc
        for sq in b.pieces(PAWN, BLACK):
            if not (PASSED_MASK_BLACK[sq] & wpbb):
                rank = sq >> 3;  file = sq & 7
                steps_b: int = rank - (1 if rank == 6 else 0)
                if max(wk >> 3, abs((wk & 7) - file)) > steps_b + extra_b:
                    score -= 200
                if rank <= 3:
                    key_rank = max(0, rank - 2)
                    for kf in range(max(0, file - 1), min(7, file + 1) + 1):
                        if bk == key_rank * 8 + kf:
                            score -= 50
        return score

    def _eval_rook_placement(self,wrooks: chess.SquareSet,brooks: chess.SquareSet,wpbb:   int,bpbb:   int,) -> int:
        """Bonus pour les tours sur colonnes ouvertes ou semi-ouvertes.
        Entrée  : wrooks/brooks cases des tours blanc/noir ;wpbb/bpbb bitboards des pions blanc/noir.
        Sortie  : score entier (positif = avantage blanc)."""
        score:      int = 0
        all_pawn_bb: int = wpbb | bpbb
        for sq in wrooks:
            fm: int = FILE_MASK[sq & 7]
            if   not (all_pawn_bb & fm): score += 40   # Colonne ouverte
            elif not (wpbb & fm):        score += 20   # Colonne semi-ouverte
        for sq in brooks:
            fm = FILE_MASK[sq & 7]
            if   not (all_pawn_bb & fm): score -= 40
            elif not (bpbb & fm):        score -= 20
        return score

    def _eval_pawn_structure(self,wpawns: chess.SquareSet,bpawns: chess.SquareSet,wpbb:   int,bpbb:   int,) -> int:
        """Évalue la structure de pions : doublés, isolés, passés, bloqués, chaînes.
        Entrée  : wpawns/bpawns cases des pions blanc/noir ; wpbb/bpbb bitboards des pions blanc/noir.
        Sortie  : score entier (positif = avantage blanc)."""
        score:   int = 0
        wp_list: list[int] = list(wpawns)
        bp_list: list[int] = list(bpawns)
        wp_set:  set[int]  = set(wp_list)
        bp_set:  set[int]  = set(bp_list)

        wf_set: set[int]       = set(sq & 7 for sq in wp_list)
        bf_set: set[int]       = set(sq & 7 for sq in bp_list)
        wfc:    Counter[int]   = Counter(sq & 7 for sq in wp_list)
        bfc:    Counter[int]   = Counter(sq & 7 for sq in bp_list)

        # Pions doublés
        for f, c in wfc.items():
            if c > 1: score -= 15 * (c - 1)
        for f, c in bfc.items():
            if c > 1: score += 15 * (c - 1)

        # Pions isolés
        for sq in wp_list:
            f = sq & 7
            if (f - 1 not in wf_set) and (f + 1 not in wf_set):
                score -= 20
        for sq in bp_list:
            f = sq & 7
            if (f - 1 not in bf_set) and (f + 1 not in bf_set):
                score += 20

        bk: int = self.board.king(BLACK)
        wk: int = self.board.king(WHITE)

        # Pions passés blancs
        for sq in wp_list:
            if not (PASSED_MASK_WHITE[sq] & bpbb):
                rank: int = sq >> 3;  file: int = sq & 7
                bonus: int = 30 + rank * 10
                if bk is not None:
                    dist: int = abs((bk & 7) - file) + abs((bk >> 3) - rank)
                    bonus += dist * 3
                for f in (file - 1, file + 1):
                    if 0 <= f < 8 and rank > 0 and (rank - 1) * 8 + f in wp_set:
                        bonus += 40; break
                score += bonus

        # Pions passés noirs
        for sq in bp_list:
            if not (PASSED_MASK_BLACK[sq] & wpbb):
                rank = sq >> 3;  file = sq & 7
                bonus = 30 + (7 - rank) * 10
                if wk is not None:
                    dist = abs((wk & 7) - file) + abs((wk >> 3) - rank)
                    bonus += dist * 3
                for f in (file - 1, file + 1):
                    if 0 <= f < 8 and rank < 7 and (rank + 1) * 8 + f in bp_set:
                        bonus += 40; break
                score -= bonus

        b = self.board

        # Pions bloqués blancs (pénalité si bloqué et non soutenu)
        for sq in wp_list:
            file, rank = sq & 7, sq >> 3
            blocked: bool = False
            for r in range(rank + 1, 8):
                p = b.piece_at(r * 8 + file)
                if p and p.piece_type == PAWN:
                    if p.color == BLACK: blocked = True
                    break
            if not blocked:
                continue
            supported: bool = any(0 <= nf < 8
                and rank > 0
                and b.piece_at((rank - 1) * 8 + nf) is not None
                and b.piece_at((rank - 1) * 8 + nf).piece_type == PAWN
                and b.piece_at((rank - 1) * 8 + nf).color == WHITE
                for nf in (file - 1, file + 1)
                )
            if not supported:
                score -= 15

        # Pions bloqués noirs
        for sq in bp_list:
            file, rank = sq & 7, sq >> 3
            blocked = False
            for r in range(rank - 1, -1, -1):
                p = b.piece_at(r * 8 + file)
                if p and p.piece_type == PAWN:
                    if p.color == WHITE: blocked = True
                    break
            if not blocked:
                continue
            supported = any(
                0 <= nf < 8
                and rank < 7
                and b.piece_at((rank + 1) * 8 + nf) is not None
                and b.piece_at((rank + 1) * 8 + nf).piece_type == PAWN
                and b.piece_at((rank + 1) * 8 + nf).color == BLACK
                for nf in (file - 1, file + 1)
            )
            if not supported:
                score += 15

        # Chaînes de pions blancs
        for sq in wp_list:
            rank, file = sq >> 3, sq & 7
            if rank > 0 and any(0 <= f < 8 and (rank - 1) * 8 + f in wp_set for f in (file - 1, file + 1)):
                score += 10 + rank * 2

        # Chaînes de pions noirs
        for sq in bp_list:
            rank, file = sq >> 3, sq & 7
            if rank < 7 and any(0 <= f < 8 and (rank + 1) * 8 + f in bp_set for f in (file - 1, file + 1)):
                score -= 10 + (7 - rank) * 2

        # Avance des pions passés blancs avec soutien
        for sq in wp_list:
            file, rank = sq & 7, sq >> 3
            if not (PASSED_MASK_WHITE[sq] & bpbb):
                continue
            cols: set[int] = set(range(max(0, file - 1), min(7, file + 1) + 1))
            blockers:   int = sum(1 for p in bp_list if (p & 7) in cols and (p >> 3) > rank)
            supporters: int = sum(1 for p in wp_list if (p & 7) in cols and (p >> 3) <= rank and p != sq)
            if supporters >= blockers:
                score += 10 + rank * 3

        # Avance des pions passés noirs avec soutien
        for sq in bp_list:
            file, rank = sq & 7, sq >> 3
            if not (PASSED_MASK_BLACK[sq] & wpbb):
                continue
            cols = set(range(max(0, file - 1), min(7, file + 1) + 1))
            blockers   = sum(1 for p in wp_list if (p & 7) in cols and (p >> 3) < rank)
            supporters = sum(1 for p in bp_list if (p & 7) in cols and (p >> 3) >= rank and p != sq)
            if supporters >= blockers:
                score -= 10 + (7 - rank) * 3

        return score

    def _eval_center_control(self) -> int:
        """Évalue le contrôle du centre étendu (cases d4/d5/e4/e5 + anneau extérieur).
        Entrée  : aucune (utilise self.board).
        Sortie  : score entier (positif = avantage blanc)."""
        score: int = 0
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

    def _eval_mobility(self,wp: dict[PieceType, chess.SquareSet],bp: dict[PieceType, chess.SquareSet],) -> int:
        """Évalue la mobilité des pièces lourdes et légères (nombre de cases attaquées).
        Entrée  : wp/bp dictionnaires {type_pièce: cases} pour blanc/noir.
        Sortie  : score entier pondéré (positif = avantage blanc)."""
        b = self.board
        wa: int = sum(int(b.attacks(sq)).bit_count() for pt in (QUEEN, ROOK, BISHOP, KNIGHT) for sq in wp[pt])
        ba: int = sum(int(b.attacks(sq)).bit_count() for pt in (QUEEN, ROOK, BISHOP, KNIGHT) for sq in bp[pt])
        return (wa - ba) * 3

    def _eval_king_safety(self,wp:   dict[PieceType, chess.SquareSet],bp:   dict[PieceType, chess.SquareSet],wk:   int,bk:   int,) -> int:
        """Évalue la sécurité du roi : attaquants proches, bouclier de pions, cases faibles.
        Entrée  : wp/bp ,dictionnaires pièces blanc/noir ; wk/bk cases des rois.
        Sortie  : score entier (positif = avantage blanc)."""
        score: int = 0
        b = self.board
        for king, color, sign, shield_dir in ((wk, WHITE,  1,  1),(bk, BLACK, -1, -1),):
            if king is None: continue
            opp: Color = not color
            attackers: int = sum(1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king]) if (p := b.piece_at(sq)) and p.color == opp)
            file: int = king & 7
            rank: int = king >> 3
            sr1:  int = rank + shield_dir
            sr2:  int = rank + 2 * shield_dir
            pawn_shield: int = sum(
                1 for r in (sr1, sr2) if 0 <= r < 8
                for f in range(max(0, file - 1), min(7, file + 1) + 1)
                if (p := b.piece_at(r * 8 + f)) and p.piece_type == PAWN and p.color == color
            )
            weak: int = sum(1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king]) if not b.is_attacked_by(color, sq))
            score += sign * (-25 * attackers + 12 * pawn_shield - 8 * weak)
        return score

    def _eval_tactics(self,wp: dict[PieceType, chess.SquareSet],bp: dict[PieceType, chess.SquareSet],) -> int:
        """Évalue les éléments tactiques : pièces non protégées/attaquées, tours sur la 7e rangée, cavaliers avant-postes, fous actifs.
        Entrée  : wp/bp dictionnaires pièces blanc/noir.
        Sortie  : score entier (positif = avantage blanc)."""
        score: int = 0
        b = self.board

        # Pénalité pour pièces non défendues et/ou attaquées
        for color, sign in ((WHITE, 1), (BLACK, -1)):
            opp    = not color
            pieces = wp if color == WHITE else bp
            for pt in (QUEEN, ROOK, BISHOP, KNIGHT):
                for sq in pieces[pt]:
                    if not b.is_attacked_by(color, sq):
                        v = PIECE_VALUES[pt]
                        score -= sign * (v // 2 if b.is_attacked_by(opp, sq) else v // 8)

        # Tours sur la 7e rangée
        for sq in wp[ROOK]:
            if sq >> 3 == 6: score += 50
        for sq in bp[ROOK]:
            if sq >> 3 == 1: score -= 50

        # Cavaliers blancs en avant-poste (non chassables par pions noirs)
        for sq in wp[KNIGHT]:
            rank, file = sq >> 3, sq & 7
            if rank >= 4 and rank + 1 < 8 and not any(
                b.piece_at((rank + 1) * 8 + f) and
                b.piece_at((rank + 1) * 8 + f).piece_type == PAWN and
                b.piece_at((rank + 1) * 8 + f).color == BLACK
                for f in (file - 1, file + 1) if 0 <= f < 8
            ): score += 20

        # Cavaliers noirs en avant-poste
        for sq in bp[KNIGHT]:
            rank, file = sq >> 3, sq & 7
            if rank <= 3 and rank - 1 >= 0 and not any(
                b.piece_at((rank - 1) * 8 + f) and
                b.piece_at((rank - 1) * 8 + f).piece_type == PAWN and
                b.piece_at((rank - 1) * 8 + f).color == WHITE
                for f in (file - 1, file + 1) if 0 <= f < 8
            ): score -= 20

        # Fous actifs (≥ 6 cases attaquées)
        for sq in wp[BISHOP]:
            if int(b.attacks(sq)).bit_count() >= 6: score += 15
        for sq in bp[BISHOP]:
            if int(b.attacks(sq)).bit_count() >= 6: score -= 15

        return score

    def _eval_king_exposure(self,wp:   dict[PieceType, chess.SquareSet],bp:   dict[PieceType, chess.SquareSet],wk:   int,bk:   int,wpbb: int,bpbb: int,) -> int:
        """Pénalise le roi lorsque sa colonne est ouverte et attaquée par tours/dames ennemies.
        Entrée  : wp/bp pièces blanc/noir ; wk/bk cases des rois ; wpbb/bpbb bitboards des pions blanc/noir.
        Sortie  : score entier (positif = avantage blanc)."""
        score: int = 0
        b = self.board
        if bk is not None:
            fm: int = FILE_MASK[bk & 7]
            if not (bpbb & fm):
                heavy: int = sum(1 for pt in (ROOK, QUEEN) for sq in wp[pt] if sq & 7 == bk & 7)
                if heavy: score += 25 * heavy
        if wk is not None:
            fm = FILE_MASK[wk & 7]
            if not (wpbb & fm):
                heavy = sum(1 for pt in (ROOK, QUEEN) for sq in bp[pt] if sq & 7 == wk & 7)
                if heavy: score -= 25 * heavy
        return score

    def _eval_pins(self,wp: dict[PieceType, chess.SquareSet],bp: dict[PieceType, chess.SquareSet],) -> int:
        """Pénalise les pièces clouées (incapables de se défendre correctement).
        Entrée  : wp/bp dictionnaires pièces blanc/noir.
        Sortie  : score entier (positif = avantage blanc, i.e. moins de clouages)."""
        score: int = 0
        b = self.board
        for pt in (KNIGHT, BISHOP, ROOK, QUEEN):
            for sq in wp[pt]:
                if b.is_pinned(WHITE, sq): score -= PIECE_VALUES[pt] // 8
            for sq in bp[pt]:
                if b.is_pinned(BLACK, sq): score += PIECE_VALUES[pt] // 8
        return score

    # Résolveur KPK (Roi + Pion vs Roi)
    def _solve_kpk(self, ply: int = 0) -> Optional[int]:
        """Détermine heuristiquement le résultat d'une finale KPK par la règle du carré : si le roi adverse ne peut pas intercepter le pion avant sa promotion, retourne un score de victoire.
        Entrée  : ply profondeur courante (pour normaliser le score de mat).
        Sortie  : score entier si la finale est résolue, None sinon."""
        b = self.board
        pieces = b.piece_map()
        if len(pieces) != 3:
            return None
        pawns: list[int] = list(b.pieces(PAWN, WHITE)) + list(b.pieces(PAWN, BLACK))
        if len(pawns) != 1:
            return None

        pawn_sq:    int   = pawns[0]
        pawn_color: Color = b.color_at(pawn_sq)
        wk: int = b.king(WHITE)
        bk: int = b.king(BLACK)
        rank: int = pawn_sq >> 3
        file: int = pawn_sq & 7

        if pawn_color == WHITE:
            steps: int = 7 - rank - (1 if rank == 1 else 0)
            dist:  int = max(abs((bk >> 3) - 7), abs((bk & 7) - file))
            if dist > steps + (1 if b.turn == BLACK else 0):
                return (100_000 - ply) if b.turn == WHITE else -(100_000 - ply)
        else:
            steps = rank - (1 if rank == 6 else 0)
            dist  = max(wk >> 3, abs((wk & 7) - file))
            if dist > steps + (1 if b.turn == WHITE else 0):
                return (100_000 - ply) if b.turn == BLACK else -(100_000 - ply)
        return None
    
    # Évaluation statique des échanges (SEE)
    def see(self, to_sq: int, attacker_color: Color) -> int:
        """Évaluation statique des échanges (Static Exchange Evaluation).Simule une séquence d'échanges sur la case to_sq pour estimer le gain ou la perte matérielle.
        Entrée  : to_sq , case cible (0-63) ; attacker_color couleur de l'attaquant initial.
        Sortie  : score entier en centipawns (positif = gain pour l'attaquant)."""
        target = self.board.piece_at(to_sq)
        if target is None:
            return 0
        b = self.board

        def collect(color: Color) -> list[tuple[int, int]]:
            """Collecte et trie les attaquants d'une couleur par valeur croissante.
            Entrée  : color ,couleur des attaquants.
            Sortie  : liste triée de tuples (valeur_pièce, case)."""
            return sorted(
                (PIECE_VALUES.get(b.piece_at(sq).piece_type, 20000), sq)
                for pt in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
                for sq in b.pieces(pt, color)
                if to_sq in b.attacks(sq)
            )

        atk_w: list[tuple[int, int]] = collect(attacker_color)
        if not atk_w:
            return 0
        atk_d: list[tuple[int, int]] = collect(not attacker_color)

        gain: list[int] = []
        cap:  int       = PIECE_VALUES[target.piece_type]
        sides:   list[list[tuple[int, int]]] = [atk_w, atk_d]
        side:    int                         = 0
        indices: list[int]                   = [0, 0]

        while indices[side] < len(sides[side]):
            pv: int = sides[side][indices[side]][0]
            indices[side] += 1
            gain.append(cap)
            cap  = pv
            side ^= 1

        # Propagation arrière des gains (negamax sur la séquence d'échanges)
        for d in range(len(gain) - 1, 0, -1):
            gain[d - 1] -= max(0, gain[d])
        return gain[0] if gain else 0

    # Tri des coups et heuristiques
    _CENTER_CORE:  frozenset[int] = frozenset({27, 28, 35, 36})
    _CENTER_OUTER: frozenset[int] = frozenset({18, 19, 20, 21, 26, 29, 34, 37, 42, 43, 44, 45})

    def _order_moves(self,moves:     list[Move],ply:       int,prev_move: Optional[Move] = None,) -> list[Move]:
        """Trie la liste des coups pour optimiser la coupure alpha-bêta.Ordre de priorité : coup TT > captures SEE ≥ 0 > promotions > killer moves > counter move > history > bonus centre.
        Entrée  : moves :liste des coups légaux à trier ;ply: profondeur courante (pour accéder aux killers) ;prev_move: coup précédent (pour le counter move).
        Sortie  : liste de coups triée par score décroissant."""
        b    = self.board
        zkey = self._zobrist(b)

        # Coup de la table de transpositions (priorité maximale)
        tt_move: Optional[Move] = None
        if zkey in self.transposition_table:
            try:
                tt_move = chess.Move.from_uci(self.transposition_table[zkey]["best_move"])
            except Exception:
                pass

        counter_move: Optional[Move] = (self.counter_moves[prev_move.from_square][prev_move.to_square]if prev_move is not None else None)
        killers: list[Optional[Move]] = self.killer_moves[ply] if ply < 64 else [None, None]

        def _score(move: Move) -> int:
            """ Calcule le score de tri d'un coup.
            Entrée  : move  : coup à évaluer.
            Sortie  : score entier (plus élevé = joué en premier)."""
            if tt_move and move == tt_move:
                return 1_000_000

            sc: int = 0

            if b.is_capture(move):
                if b.is_en_passant(move):
                    sc += 500_000
                else:
                    target   = b.piece_at(move.to_square)
                    attacker = b.piece_at(move.from_square)
                    if target and attacker:
                        ss: int = self.see(move.to_square, b.turn)
                        sc += (500_000 + ss) if ss >= 0 else ss

            if move.promotion:
                sc += 900 if move.promotion == QUEEN else 300

            if move == killers[0]:   sc += 1_000
            elif move == killers[1]: sc += 800
            if counter_move and move == counter_move:
                sc += 600

            mk: tuple[int, int] = (move.from_square, move.to_square)
            if mk in self.history:
                sc += self.history[mk]

            if move.to_square in self._CENTER_CORE:
                sc += 15
            elif move.to_square in self._CENTER_OUTER:
                sc += 5

            return sc

        return sorted(moves, key=_score, reverse=True)

    def _update_killers(self, move: Move, ply: int) -> None:
        """Met à jour la table des killer moves pour le ply courant (FIFO de 2).
        Entrée  : move :coup qui a causé une coupure bêta ; ply – profondeur.
        Sortie  : aucune (modification in-place de self.killer_moves)."""
        if ply < 64 and self.killer_moves[ply][0] != move:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move

    def _update_history(self, move: Move, depth: int) -> None:
        """Incrémente la table d'historique pour un coup qui améliore alpha.La valeur est pondérée par depth au carré pour favoriser les coupures profondes.
        Entrée  : move  coup tranquille ayant causé une coupure bêta ; depth profondeur restante au nœud courant.
        Sortie  : aucune (modification in-place de self.history). """
        k: tuple[int, int] = (move.from_square, move.to_square)
        self.history[k] = self.history.get(k, 0) + depth * depth

    def _update_counter_move(self,prev_move: Optional[Move],move:Move) -> None:
        """Enregistre le coup courant comme réfutation du coup précédent.
        Entrée  : prev_move : coup adverse précédent (peut être None) ;move : coup qui lui répond.
        Sortie  : aucune (modification in-place de self.counter_moves)."""
        if prev_move is not None:
            self.counter_moves[prev_move.from_square][prev_move.to_square] = move

    # Recherche de quiescence
    def quiescence(self, alpha: int, beta: int, ply: int = 0) -> int:
        """Recherche de quiescence : continue la recherche uniquement sur les captures (et les coups sous échec) jusqu'à une position calme.Applique l'élagage delta pour éviter les captures clairement défavorables.
        Entrée  : alpha/beta :fenêtre de recherche ; ply :profondeur courante.
        Sortie  : score de la meilleure position calme atteignable."""
        b        = self.board
        in_check: bool = b.is_check()

        # Sous échec : on examine tous les coups légaux
        if in_check:
            legal = list(b.legal_moves)
            if not legal:
                return -100_000 + ply   # Mat
            for move in legal:
                b.push(move)
                score: int = -self.quiescence(-beta, -alpha, ply + 1)
                b.pop()
                if score >= beta: return beta
                if score > alpha: alpha = score
            return alpha

        # Stand-pat : évaluation statique comme borne inférieure
        stand_pat: int = self.evaluate()
        if b.turn == BLACK:
            stand_pat = -stand_pat

        if stand_pat >= beta: return beta
        if stand_pat > alpha: alpha = stand_pat

        # Génération et tri des captures
        captures: list[tuple[int, Move, int]] = []
        for m in b.legal_moves:
            if not b.is_capture(m):
                continue
            if b.is_en_passant(m):
                captures.append((0, m, PIECE_VALUES[PAWN]))
                continue
            target = b.piece_at(m.to_square)
            if target is None:
                continue
            see_val:  int = self.see(m.to_square, b.turn)
            gain_val: int = PIECE_VALUES.get(target.piece_type, 0)
            if m.promotion == QUEEN:
                gain_val += PIECE_VALUES[QUEEN] - PIECE_VALUES[PAWN]
            captures.append((see_val, m, gain_val))

        captures.sort(key=lambda x: x[0], reverse=True)

        for see_val, move, gain in captures:
            # Élagage delta
            if stand_pat + gain + DELTA_MARGIN <= alpha:
                continue
            # Rejette les captures clairement perdantes (SEE < -50)
            if see_val < -50:
                continue

            b.push(move)
            score = -self.quiescence(-beta, -alpha, ply + 1)
            b.pop()

            if score >= beta: return beta
            if score > alpha: alpha = score

        return alpha

    # Null-Move Pruning
    def _try_null_move(self, depth: int, beta: int, ply: int) -> Optional[int]:
        """Tente le null-move pruning : passe son tour et vérifie si le score dépasse beta avec une profondeur réduite de 3.Désactivé en finale légère (≤ 8 pièces) ou sous échec.
        Entrée  : depth : profondeur restante ; beta : borne supérieure ; ply : courant.
        Sortie  : beta si la coupure est confirmée, None sinon."""
        b = self.board
        if depth < 3 or b.is_check() or len(b.piece_map()) <= 8:
            return None
        if abs(beta) >= 90_000:
            return None
        # Évite le null-move dans les finales purement pionnesques (zugzwang)
        if not (b.occupied_co[b.turn] & ~b.pawns & ~b.kings):
            return None
        b.push(chess.Move.null())
        score: int = -self.negamax(depth - 3, -beta, -beta + 1, ply + 1, None)
        b.pop()
        return beta if score >= beta else None

    # Gestion de la table de transpositions
    def _maybe_evict_tt(self) -> None:
        """Évince les entrées peu profondes de la TT lorsqu'elle dépasse TT_MAX_SIZE.Supprime un échantillon aléatoire d'entrées de profondeur ≤ 2.
        Entrée  : aucune (utilise self.transposition_table).
        Sortie  : aucune (modification in-place de la TT)."""
        if len(self.transposition_table) < TT_MAX_SIZE:
            return
        keys:    list = list(self.transposition_table.keys())
        sample:  list = random.sample(keys, min(2000, len(keys)))
        shallow: list = [
            k for k in sample
            if self.transposition_table[k].get("depth", 0) <= 2
        ]
        to_del: list = shallow[:TT_MAX_SIZE // 10]
        for k in to_del:
            del self.transposition_table[k]

    # Negamax principal
    def negamax(self,depth: int,alpha: int,beta: int,ply: int,prev_move: Optional[Move] = None,) -> int:
        """Algorithme Negamax récursif avec élagage alpha-bêta et toutes lesoptimisations (TT, LMR, futilité, null-move, probcut, extensionsingulière, IID, extension d'échec).
        Entrée  : depth     :profondeur de recherche restante ;alpha :borne inférieure (meilleur score garanti) ;beta:borne supérieure (score adversaire connu) ;ply :profondeur depuis la racine ;
                  prev_move :coup précédent (pour counter move et null-move).
        Sortie  : score de la position depuis la perspective du joueur courant."""
        # Vérification du temps
        if self._time_exceeded:
            return 0
        if time.time() - self._search_start_time > self.time_limit:
            self._time_exceeded = True
            return 0

        self.nodes_searched += 1
        b = self.board

        # Résolveur KPK
        kpk_score: Optional[int] = self._solve_kpk(ply)
        if kpk_score is not None:
            return kpk_score

        # Répétition de position → score de nulle (avec contempt)
        if b.is_repetition(2):
            return -CONTEMPT

        # Règle des 50 coups
        if b.halfmove_clock >= 100:
            return 0

        zkey:       int = self._zobrist(b)
        alpha_orig: int = alpha

        # Consultation de la table de transpositions
        if zkey in self.transposition_table:
            entry = self.transposition_table[zkey]
            if entry["depth"] >= depth:
                flag:   int = entry.get("flag", EXACT)
                stored: int = entry["score"]
                # Dénormalisation des scores de mat
                if stored > 90_000:
                    score: int = stored - ply
                elif stored < -90_000:
                    score = stored + ply
                else:
                    score = stored
                if flag == EXACT:
                    return score
                elif flag == LOWERBOUND:
                    alpha = max(alpha, score)
                elif flag == UPPERBOUND:
                    beta  = min(beta,  score)
                if alpha >= beta:
                    return score

        in_check: bool = b.is_check()

        # Extension d'échec (limitée par MAX_CHECK_EXT)
        if in_check and ply < 2 * self.depth + MAX_CHECK_EXT:
            depth += 1

        # Nœud feuille  => recherche de quiescence
        if depth <= 0:
            if b.is_game_over():
                return (-100_000 + ply) if b.is_checkmate() else 0
            return self.quiescence(alpha, beta, ply)

        if b.is_game_over():
            return (-100_000 + ply) if b.is_checkmate() else 0

        # Razoring (depth == 1)
        if depth == 1 and not in_check and abs(alpha) < 90_000:
            se: int = self.evaluate()
            if b.turn == BLACK: se = -se
            if se < alpha - RAZORING_MARGIN:
                return self.quiescence(alpha, beta, ply)

        # Null-Move Pruning
        if not in_check and depth >= 3:
            nr: Optional[int] = self._try_null_move(depth, beta, ply)
            if nr is not None:
                return nr

        moves: list[Move] = self._order_moves(list(b.legal_moves), ply, prev_move)
        if not moves:
            return (-100_000 + ply) if b.is_checkmate() else 0

        # IID : recherche préliminaire peu profonde pour alimenter la TT
        if depth >= 4 and not in_check and zkey not in self.transposition_table:
            self.negamax(depth - 2, alpha, beta, ply, prev_move)
            moves = self._order_moves(moves, ply, prev_move)

        # ProbCut
        if depth >= 5 and not in_check and abs(beta) < 90_000:
            pc_beta:   int = beta + PROBCUT_MARGIN
            pc_static: int = self.evaluate()
            if b.turn == BLACK: pc_static = -pc_static
            threshold: int = pc_beta - pc_static
            for m in [m for m in moves if b.is_capture(m) and self.see(m.to_square, b.turn) >= threshold][:3]:
                b.push(m)
                pc_score: int = -self.negamax(max(1, depth - 4), -pc_beta, -pc_beta + 1, ply + 1, m)
                b.pop()
                if pc_score >= pc_beta:
                    return pc_beta

        # Extension singulière
        singular_move: Optional[Move] = None
        if depth >= 4 and not in_check and zkey in self.transposition_table:
            tt_e = self.transposition_table[zkey]
            if (tt_e.get("depth", 0) >= depth - 3 and tt_e.get("flag", EXACT) != UPPERBOUND and abs(tt_e.get("score", 0)) < 90_000):
                try:
                    cand: Move = chess.Move.from_uci(tt_e["best_move"])
                    if cand in moves:
                        s_beta:  int  = tt_e["score"] - SINGULAR_MARGIN
                        s_depth: int  = min(depth // 2, 3)
                        s_fails: bool = False
                        for i, m in enumerate(moves):
                            if m == cand or i >= 6:
                                continue
                            b.push(m)
                            s_val: int = -self.negamax(s_depth, -s_beta, -(s_beta - 1), ply + 1, m)
                            b.pop()
                            if s_val >= s_beta:
                                s_fails = True; break
                        if not s_fails:
                            singular_move = cand
                except Exception:
                    pass

        # Élagage par futilité
        futility_pruning: bool = False
        if (depth in FUTILITY_MARGINS and not in_check and abs(alpha) < 90_000 and abs(beta) < 90_000):
            fe: int = self.evaluate()
            if b.turn == BLACK: fe = -fe
            futility_pruning = fe + FUTILITY_MARGINS[depth] <= alpha

        # Boucle principale sur les coups
        best_move:     Move = moves[0]
        best_score:    int  = -10**9
        moves_searched: int = 0
        quiet_count:    int = 0

        for move in moves:
            is_capture:   bool = b.is_capture(move)
            is_promotion: bool = bool(move.promotion)
            is_quiet:     bool = not is_capture and not is_promotion

            # Élagage par futilité (coups tranquilles non premiers)
            if futility_pruning and is_quiet and moves_searched > 0:
                continue
            # LMP : Late Move Pruning
            if (depth in LMP_COUNTS and is_quiet and not in_check and moves_searched > 0 and quiet_count >= LMP_COUNTS[depth]):
                continue
            if is_quiet:
                quiet_count += 1

            b.push(move)

            if moves_searched == 0:
                # Premier coup : recherche complète (avec extension singulière éventuelle)
                ext: int = 1 if singular_move is not None and move == singular_move else 0
                score = -self.negamax(depth - 1 + ext, -beta, -alpha, ply + 1, move)
            else:
                # LMR : Late Move Reduction
                reduction: int = 0
                if (depth >= 3 and moves_searched >= 4 and not in_check and not is_capture and not b.is_check()):
                    reduction = max(1, int(math.log(depth) * math.log(moves_searched) / 1.5))

                score = -self.negamax(depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, move)
                if alpha < score < beta:
                    if reduction > 0:
                        score = -self.negamax(depth - 1, -alpha - 1, -alpha, ply + 1, move)
                    if score > alpha:
                        score = -self.negamax(depth - 1, -beta, -alpha, ply + 1, move)

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

        self._maybe_evict_tt()

        # Détermination du drapeau TT
        flag = EXACT
        if best_score <= alpha_orig: flag = UPPERBOUND
        elif best_score >= beta:     flag = LOWERBOUND

        # Normalisation des scores de mat avant stockage
        if best_score > 90_000:
            stored_score: int = best_score + ply
        elif best_score < -90_000:
            stored_score = best_score - ply
        else:
            stored_score = best_score

        self.transposition_table[zkey] = {"best_move": best_move.uci(),"score":stored_score,"depth":depth,"flag":flag}
        return best_score

    # Livre d'ouvertures
    def get_opening_move(self, board: Board) -> Optional[str | Move]:
        """Consulte le livre d'ouvertures et retourne un coup aléatoire si disponible.
        Entrée  : board :plateau courant.
        Sortie  : coup SAN (str) ou Move, ou None si position inconnue. """
        norm:  str             = _normalize_fen(board.fen())
        moves: Optional[list]  = OPENING_BOOK_NORMALIZED.get(norm)
        return random.choice(moves) if moves else None

    # Promotion intelligente (évite le pat)
    def _smart_promotion(self, board: Board, move: Move) -> Move:
        """Vérifie si la promotion en dame provoque un pat ; si oui, tente une promotion alternative (tour, cavalier, fou) qui ne pat pas.
        Entrée  : board  : plateau courant ; move :coup de promotion candidat.
        Sortie  : le coup de promotion le plus approprié (chess.Move)."""
        if move.promotion != QUEEN:
            return move
        board.push(move)
        is_stale: bool = board.is_stalemate()
        board.pop()
        if not is_stale:
            return move
        for piece in (ROOK, KNIGHT, BISHOP):
            alt: Move = chess.Move(move.from_square, move.to_square, promotion=piece)
            if alt not in board.legal_moves:
                continue
            board.push(alt)
            still: bool = board.is_stalemate()
            board.pop()
            if not still:
                return alt
        return move

    # Point d'entrée public
    def coup(self, board: Board) -> Move:
        """Calcule et retourne le meilleur coup pour la position courante.
        Pipeline :
          1. Livre d'ouvertures (12 premiers coups par camp)
          2. Détection du mat en 1
          3. Approfondissement itératif avec fenêtres d'aspiration
          4. Fallback aléatoire parmi les meilleures captures si aucun coup trouvé
          5. Vérification de la promotion (anti-pat)
        Entrée  : board :plateau chess.Board courant (partagé avec l'appelant).
        Sortie  : chess.Move :le meilleur coup calculé."""
        color: Color = board.turn

        #Ouvertures 
        if self.opening_moves_played[color] < 12:
            mv = self.get_opening_move(board)
            if mv:
                if isinstance(mv, str):
                    try:
                        mv = board.parse_san(mv)
                    except Exception:
                        mv = None
                if mv and mv in board.legal_moves:
                    self.opening_moves_played[color] += 1
                    return mv

        #Mat en 1 
        for mv in board.legal_moves:
            board.push(mv)
            mate: bool = board.is_checkmate()
            board.pop()
            if mate:
                return mv

        #Initialisation de la recherche
        self.board              = board
        self.killer_moves       = [[None, None] for _ in range(64)]
        self.nodes_searched     = 0
        self._search_start_time = time.time()
        self._time_exceeded     = False

        # Vieillissement de l'historique (division par 2, suppression des nuls)
        self.history = {k: v >> 1 for k, v in self.history.items() if v > 1}

        best_move:  Optional[Move] = None
        prev_score: Optional[int]  = None

        # Profondeur adaptative selon le nombre de pièces (finale)
        piece_count: int = len(board.piece_map())
        if piece_count <= 5:
            effective_depth: int = self.depth + 2
        elif piece_count <= 8:
            effective_depth = self.depth + 1
        else:
            effective_depth = self.depth

        #Approfondissement itératif 
        for d in range(1, effective_depth + 1):
            self._time_exceeded = False

            if d >= 2 and best_move is not None:
                # Fenêtres d'aspiration : agrandies progressivement si rate
                delta: int = 50
                a:  int = prev_score - 50
                b_: int = prev_score + 50
                while True:
                    self._time_exceeded = False
                    score, move = self.negamax_root(d, a, b_)
                    if self._time_exceeded:
                        break
                    if score <= a:
                        a   -= delta; delta = min(delta * 2, 500)
                    elif score >= b_:
                        b_  += delta; delta = min(delta * 2, 500)
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
                    break     # Mat détecté : inutile de continuer
            else:
                break

            if time.time() - self._search_start_time > self.time_limit * 0.85:
                break

        #Fallback
        if best_move is None:
            moves: list[Move] = list(board.legal_moves)
            if not moves:
                raise ValueError("Aucun coup légal trouvé !")
            scored: list[tuple[int, Move]] = sorted(((PIECE_VALUES[board.piece_at(m.to_square).piece_type]if board.is_capture(m) and board.piece_at(m.to_square) else 0),m) for m in moves)
            best_val: int  = scored[-1][0]
            best_move      = random.choice([m for v, m in scored if v == best_val])

        best_move        = self._smart_promotion(board, best_move)
        self._last_move  = best_move
        return best_move

    def negamax_root(self,depth: int,alpha: int = -10**9,beta:  int =  10**9,) -> tuple[int, Optional[Move]]:
        """Appel Negamax depuis la racine avec gestion du temps et stockage TT.
        Entrée  : depth :profondeur de recherche ; alpha/beta :fenêtre initiale (défaut : pleine fenêtre).
        Sortie  : tuple (meilleur_score, meilleur_coup)."""
        best_score: int         = -10**9
        best_move:  Optional[Move] = None
        alpha_orig: int         = alpha

        moves: list[Move] = self._order_moves(list(self.board.legal_moves), 0, self._last_move)
        if not moves:
            return 0, None

        for move in moves:
            if time.time() - self._search_start_time > self.time_limit:
                self._time_exceeded = True
                break
            self.board.push(move)
            score: int = -self.negamax(depth - 1, -beta, -alpha, 1, move)
            self.board.pop()

            if score > best_score:
                best_score, best_move = score, move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        # Stockage de la racine dans la TT (coup TT pour l'IID suivante)
        if best_move is not None and not self._time_exceeded:
            zkey: int | str = self._zobrist(self.board)
            flag: int = EXACT
            if best_score <= alpha_orig: flag = UPPERBOUND
            elif best_score >= beta:     flag = LOWERBOUND
            self.transposition_table[zkey] = {"best_move": best_move.uci(),"score": best_score,"depth": depth,"flag": flag}

        return best_score, best_move