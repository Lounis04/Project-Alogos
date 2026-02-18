import tkinter as tk
from chess import Move, QUEEN, ROOK, BISHOP, KNIGHT, PAWN, WHITE, BLACK


class HumanController:
    def __init__(self, board, canvas, root, human_white: bool, human_black: bool, update_board_cb):
        self.board = board
        self.canvas = canvas
        self.root = root
        self.human_white = human_white
        self.human_black = human_black
        self.update_board_cb = update_board_cb

        self.selected_square = None
        self.highlight_ids = []          # cases de mouvement
        self.selection_id = None         # contour jaune

        # Bind simple clic gauche
        self.canvas.bind("<Button-1>", self.on_click)

    def is_human_turn(self) -> bool:
        if self.board.turn == WHITE:
            return self.human_white
        else:
            return self.human_black

    def maybe_schedule_ai_turn(self, jouer_cb):
        if not self.is_human_turn():
            self.root.after(500, jouer_cb)

    def clear_highlights(self):
        """Efface tous les overlays de surbrillance."""
        for hid in self.highlight_ids:
            self.canvas.delete(hid)
        self.highlight_ids.clear()

        if self.selection_id is not None:
            self.canvas.delete(self.selection_id)
            self.selection_id = None

    def highlight_square(self, row, col):
        """Overlay gris sur une case de déplacement possible."""
        cell_w = int(self.canvas.cget("width")) / 8
        cell_h = int(self.canvas.cget("height")) / 8

        x1 = col * cell_w
        y1 = (7 - row) * cell_h
        x2 = x1 + cell_w
        y2 = y1 + cell_h

        hid = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill="gray", stipple="gray50", outline=""
        )
        self.highlight_ids.append(hid)

    def highlight_selection(self, row, col):
        """Contour jaune léger autour de la pièce sélectionnée."""
        cell_w = int(self.canvas.cget("width")) / 8
        cell_h = int(self.canvas.cget("height")) / 8

        x1 = col * cell_w
        y1 = (7 - row) * cell_h
        x2 = x1 + cell_w
        y2 = y1 + cell_h

        self.selection_id = self.canvas.create_rectangle(
            x1 + 2, y1 + 2, x2 - 2, y2 - 2,
            outline="yellow",
            width=3
        )

    # ------------------------------------------------------------------
    #   PROMOTION : dialogue de choix
    # ------------------------------------------------------------------

    def _is_promotion_move(self, from_sq, to_sq):
        """Retourne True si le coup est une promotion de pion."""
        piece = self.board.piece_at(from_sq)
        if piece is None or piece.piece_type != PAWN:
            return False
        to_rank = to_sq // 8
        return (piece.color == WHITE and to_rank == 7) or \
               (piece.color == BLACK and to_rank == 0)

    def ask_promotion(self):
        """
        Affiche une fenêtre modale pour choisir la pièce de promotion.
        Retourne le type de pièce choisi (QUEEN par défaut si fermé).
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Promotion")
        dialog.resizable(False, False)
        dialog.grab_set()          # modal : bloque la fenêtre principale

        chosen = [QUEEN]           # valeur par défaut

        tk.Label(dialog, text="Choisissez la pièce de promotion :",
                 font=("Arial", 12), pady=8).pack()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(padx=20, pady=(0, 15))

        pieces = [
            (QUEEN,  "♛ Dame"),
            (ROOK,   "♜ Tour"),
            (BISHOP, "♝ Fou"),
            (KNIGHT, "♞ Cavalier"),
        ]

        for piece_type, label in pieces:
            def choose(p=piece_type):
                chosen[0] = p
                dialog.destroy()

            tk.Button(
                btn_frame, text=label,
                font=("Arial", 13), width=10, pady=6,
                command=choose
            ).pack(side="left", padx=5)

        # Centre la fenêtre sur la fenêtre principale
        dialog.update_idletasks()
        rx = self.root.winfo_rootx() + (self.root.winfo_width()  - dialog.winfo_width())  // 2
        ry = self.root.winfo_rooty() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{rx}+{ry}")

        self.root.wait_window(dialog)
        return chosen[0]

    # ------------------------------------------------------------------
    #   GESTION DES CLICS
    # ------------------------------------------------------------------

    def on_click(self, event):
        """Gestion complète du système de 2 clics."""
        if self.board.is_game_over() or not self.is_human_turn():
            return

        board_width  = int(self.canvas.cget("width"))
        board_height = int(self.canvas.cget("height"))

        col        = int(event.x / (board_width  / 8))
        row_screen = int(event.y / (board_height / 8))
        if not (0 <= col <= 7 and 0 <= row_screen <= 7):
            return

        row    = 7 - row_screen
        square = row * 8 + col

        # ── Premier clic : sélection de la pièce ──────────────────────
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece is None:
                return
            if (self.board.turn == WHITE and not piece.color) or \
               (self.board.turn == BLACK and piece.color):
                return

            self.selected_square = square
            self.clear_highlights()
            self.highlight_selection(row, col)

            for move in self.board.legal_moves:
                if move.from_square == square:
                    to_row = move.to_square // 8
                    to_col = move.to_square % 8
                    self.highlight_square(to_row, to_col)
            return

        # ── Deuxième clic : application du coup ───────────────────────
        from_sq = self.selected_square
        to_sq   = square
        self.clear_highlights()

        # Vérification d'abord sans promotion
        move  = Move(from_sq, to_sq)
        legal = move in self.board.legal_moves

        if not legal:
            # Vérifier si c'est une promotion
            if self._is_promotion_move(from_sq, to_sq):
                # Demander au joueur quelle pièce il veut
                promotion_piece = self.ask_promotion()
                move  = Move(from_sq, to_sq, promotion=promotion_piece)
                legal = move in self.board.legal_moves
            else:
                # Essayer promotion Dame (cas où move de base n'est pas légal
                # mais avec promotion l'est, p.ex. prise en promotion)
                move  = Move(from_sq, to_sq, promotion=QUEEN)
                legal = move in self.board.legal_moves
                if legal:
                    # C'est bien une promotion : demander le choix
                    promotion_piece = self.ask_promotion()
                    move  = Move(from_sq, to_sq, promotion=promotion_piece)
                    legal = move in self.board.legal_moves

        if legal:
            self.board.push(move)
            self.selected_square = None
            self.update_board_cb()
            self.maybe_schedule_ai_turn(self._jouer_after_human)
            return

        # Coup illégal → annulation
        self.selected_square = None

    def _jouer_after_human(self):
        pass