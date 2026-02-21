from typing import Callable
import tkinter as tk
from chess import Board, Move, QUEEN, WHITE, BLACK

class HumanController:
    """Gère les interactions souris du joueur humain sur l'échiquier Tkinter.
    Entrée  : board         plateau de jeu partagé (chess.Board)
              canvas        canevas Tkinter sur lequel l'échiquier est dessiné
              root          fenêtre principale Tkinter
              human_white   True si les Blancs sont contrôlés par un humain
              human_black   True si les Noirs sont contrôlés par un humain
              update_board_cb callback sans argument appelé après chaque couppour rafraîchir l'affichage
    Sortie  : aucune (effets de bord : modification du plateau, appelsaux callbacks, rafraîchissement visuel)."""
    def __init__(self,board: Board,canvas: tk.Canvas,root: tk.Tk,human_white: bool,human_black: bool,update_board_cb: Callable[[], None]) -> None:
        self.board:            Board                   = board
        self.canvas:           tk.Canvas               = canvas
        self.root:             tk.Tk                   = root
        self.human_white:      bool                    = human_white
        self.human_black:      bool                    = human_black
        self.update_board_cb:  Callable[[], None]      = update_board_cb

        self.selected_square: int | None = None
        self.highlight_ids:  list[int] = []
        self.selection_id:   int | None = None
        self.canvas.bind("<Button-1>", self.on_click)

    def is_human_turn(self) -> bool:
        """Vérifie si c'est au tour d'un joueur humain de jouer.
        Entrée  : aucune (utilise self.board.turn).
        Sortie  : True si le joueur dont c'est le tour est humain, False sinon."""
        if self.board.turn == WHITE:
            return self.human_white
        return self.human_black

    def maybe_schedule_ai_turn(self, jouer_cb: Callable[[], None]) -> None:
        """Programme un coup IA si c'est au tour de l'IA, avec un délai de 500 ms.
        Entrée  : jouer_cb : callback sans argument déclenché pour jouer le coup IA.
        Sortie  : aucune (effet de bord : programmation via root.after)."""
        if not self.is_human_turn():
            self.root.after(500, jouer_cb)

    def clear_highlights(self) -> None:
        """Supprime tous les overlays de surbrillance du canevas(cases grises de déplacement et contour jaune de sélection).
        Entrée  : aucune.
        Sortie  : aucune (effets de bord : suppression d'éléments Tkinter)."""
        for hid in self.highlight_ids:
            self.canvas.delete(hid)
        self.highlight_ids.clear()

        if self.selection_id is not None:
            self.canvas.delete(self.selection_id)
            self.selection_id = None

    def highlight_square(self, row: int, col: int) -> None:
        """Dessine un overlay semi-transparent gris sur une case de destination légale.
        Entrée  : row rang de la case (0=rang 1, 7=rang 8),col colonne de la case (0=colonne a, 7=colonne h).
        Sortie  : aucune (effet de bord : ajout d'un rectangle Tkinter dansself.highlight_ids)."""
        cell_w: float = int(self.canvas.cget("width"))  / 8
        cell_h: float = int(self.canvas.cget("height")) / 8

        x1: float = col * cell_w
        y1: float = (7 - row) * cell_h
        x2: float = x1 + cell_w
        y2: float = y1 + cell_h

        hid: int = self.canvas.create_rectangle(x1, y1, x2, y2,fill="gray", stipple="gray50", outline="",)
        self.highlight_ids.append(hid)

    def highlight_selection(self, row: int, col: int) -> None:
        """Dessine un contour jaune autour de la pièce sélectionnée.
        Entrée  : row rang de la case sélectionnée (0-7),col colonne de la case sélectionnée (0-7).
        Sortie  : aucune (effet de bord : création du rectangle Tkinter,stocké dans self.selection_id)."""
        cell_w: float = int(self.canvas.cget("width"))  / 8
        cell_h: float = int(self.canvas.cget("height")) / 8

        x1: float = col * cell_w
        y1: float = (7 - row) * cell_h
        x2: float = x1 + cell_w
        y2: float = y1 + cell_h

        self.selection_id = self.canvas.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2,outline="yellow",width=3,)

    def on_click(self, event: tk.Event) -> None: 
        """Gère un clic gauche sur le canevas 
        Entrée  : event , événement Tkinter contenant les coordonnées (x, y).
        Sortie  : aucune (effets de bord : mise à jour du plateau, rafraîchissement visuel, programmation éventuelle du tour IA)."""
        if self.board.is_game_over() or not self.is_human_turn():
            return

        board_width:  int = int(self.canvas.cget("width"))
        board_height: int = int(self.canvas.cget("height"))

        col:        int = int(event.x / (board_width  / 8))
        row_screen: int = int(event.y / (board_height / 8))
        if not (0 <= col <= 7 and 0 <= row_screen <= 7):
            return

        row:    int = 7 - row_screen
        square: int = row * 8 + col

        #1er clic : sélection de la pièce 
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece is None:
                return

            # Vérifie que la pièce appartient bien au joueur courant
            if (self.board.turn == WHITE and not piece.color) or (self.board.turn == BLACK and piece.color):
                return

            self.selected_square = square
            self.clear_highlights()
            self.highlight_selection(row, col)

            for move in self.board.legal_moves:
                if move.from_square == square:
                    to_row: int = move.to_square // 8
                    to_col: int = move.to_square % 8
                    self.highlight_square(to_row, to_col)
            return

        #2ème clic : exécution du déplacement 
        from_sq: int = self.selected_square
        to_sq:   int = square

        self.clear_highlights()

        move:  Move = Move(from_sq, to_sq)
        legal: bool = move in self.board.legal_moves

        if not legal:
            move  = Move(from_sq, to_sq, promotion=QUEEN)
            legal = move in self.board.legal_moves

        if legal:
            self.board.push(move)
            self.selected_square = None
            self.update_board_cb()
            self.maybe_schedule_ai_turn(self._jouer_after_human)
            return

        self.selected_square = None

    def _jouer_after_human(self) -> None:
        """Placeholder appelé après le coup humain pour déclencher le tour IA.Doit être surchargé ou relié à la logique de jeu principale.
        Entrée  : aucune.
        Sortie  : aucune."""
        pass