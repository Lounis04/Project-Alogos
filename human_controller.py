from chess import Move, QUEEN, WHITE, BLACK


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
        self.selection_id = None         # contour jaune (nouveau)

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

        # Effacer aussi le contour jaune
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

        # Créer un rectangle bordure, sans remplissage
        self.selection_id = self.canvas.create_rectangle(
            x1 + 2, y1 + 2, x2 - 2, y2 - 2,
            outline="yellow",
            width=3
        )


    def on_click(self, event):
        """Gestion complète du système de 2 clics."""
        if self.board.is_game_over() or not self.is_human_turn():
            return

        # Détermination de la case cliquée
        board_width = int(self.canvas.cget("width"))
        board_height = int(self.canvas.cget("height"))

        col = int(event.x / (board_width / 8))
        row_screen = int(event.y / (board_height / 8))
        if not (0 <= col <= 7 and 0 <= row_screen <= 7):
            return

        row = 7 - row_screen
        square = row * 8 + col

        if self.selected_square is None:

            piece = self.board.piece_at(square)
            if piece is None:
                return

            if (self.board.turn == WHITE and not piece.color) or \
               (self.board.turn == BLACK and piece.color):
                return

            # Sélection
            self.selected_square = square

            # Nettoyer ancien highlight + contour
            self.clear_highlights()

            # Montrer contour jaune
            self.highlight_selection(row, col)

            # Montrer les coups légaux
            for move in self.board.legal_moves:
                if move.from_square == square:
                    to_row = move.to_square // 8
                    to_col = move.to_square % 8
                    self.highlight_square(to_row, to_col)

            return

        from_sq = self.selected_square
        to_sq = square

        # Nettoyer selection + highlights
        self.clear_highlights()

        move = Move(from_sq, to_sq)
        legal = move in self.board.legal_moves

        if not legal:
            move = Move(from_sq, to_sq, promotion=QUEEN)
            legal = move in self.board.legal_moves

        if legal:
            self.board.push(move)
            self.selected_square = None
            self.update_board_cb()
            self.maybe_schedule_ai_turn(self._jouer_after_human)
            return

        self.selected_square = None

    def _jouer_after_human(self):
        pass
