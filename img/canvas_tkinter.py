from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from chess import *
from human_controller import HumanController

# global vars (scaled down)
board_width = 512   # 1024 / 2
board_height = 512  # 1024 / 2


class Chess_UI:

    def __init__(self, root: Tk, board: Board, J_Blanc, J_Noir):

        # Images scaled down from 100 → 50
        self.img_dict = {
            'p': ImageTk.PhotoImage(Image.open('img/pion_noir.png').resize((50, 50))),
            'b': ImageTk.PhotoImage(Image.open('img/fou_noir.png').resize((50, 50))),
            'q': ImageTk.PhotoImage(Image.open('img/reine_noire.png').resize((50, 50))),
            'k': ImageTk.PhotoImage(Image.open('img/roi_noir.png').resize((50, 50))),
            'n': ImageTk.PhotoImage(Image.open('img/cavalier_noir.png').resize((50, 50))),
            'r': ImageTk.PhotoImage(Image.open('img/tour_noire.png').resize((50, 50))),
            'P': ImageTk.PhotoImage(Image.open('img/pion_blanc.png').resize((50, 50))),
            'B': ImageTk.PhotoImage(Image.open('img/fou_blanc.png').resize((50, 50))),
            'Q': ImageTk.PhotoImage(Image.open('img/reine_blanche.png').resize((50, 50))),
            'K': ImageTk.PhotoImage(Image.open('img/roi_blanc.png').resize((50, 50))),
            'N': ImageTk.PhotoImage(Image.open('img/cavalier_blanc.png').resize((50, 50))),
            'R': ImageTk.PhotoImage(Image.open('img/tour_blanche.png').resize((50, 50))),
        }

        self.root = root
        self.board = board
        self.Joueur_Blanc = J_Blanc
        self.Joueur_Noir = J_Noir

        self.human_white = (J_Blanc is None)
        self.human_black = (J_Noir is None)

        self.mainframe = ttk.Frame(self.root)
        self.mainframe.grid()

        # Letters / numbers unchanged (text only)
        for i in range(8):
            Label(self.mainframe, text=chr(ord('A') + i), bg='white').grid(row=0, column=i + 1, sticky=S)
            Label(self.mainframe, text=str(i + 1), bg='white').grid(row=i + 1, column=0, sticky=E)

        # move history (height kept)
        self.history_white = []
        self.history_black = []

        self.history_white_var = StringVar(value=self.history_white)
        self.history_white_listbox = Listbox(self.mainframe, listvariable=self.history_white_var,
                                             bg="white", height=48)
        self.history_white_listbox.grid(row=1, column=9, rowspan=8, sticky=N)

        self.history_black_var = StringVar(value=self.history_black)
        self.history_black_listbox = Listbox(self.mainframe, listvariable=self.history_black_var,
                                             bg="white", height=48)
        self.history_black_listbox.grid(row=1, column=10, rowspan=8, sticky=N)

        # canvas scaled from 1024 → 512
        self.canvas = Canvas(self.mainframe, bg="black", width=board_width, height=board_height)
        self.canvas.grid(row=1, column=1, columnspan=8, rowspan=8)

        self.bg_img = Image.open('img/plateau.png')
        self.bg_img = self.bg_img.resize((512, 512))  # scale background too
        self.bg_photo = ImageTk.PhotoImage(self.bg_img)
        self.canvas.create_image(256, 256, image=self.bg_photo)

        self.pieces_list = []

        self.human_controller = HumanController(
            board=self.board,
            canvas=self.canvas,
            root=self.root,
            human_white=self.human_white,
            human_black=self.human_black,
            update_board_cb=self.update_board,
        )

        self.human_controller._jouer_after_human = self.jouer

        self.update_board()

    # ---- coordinate helpers (automatically scaled because board_width is halved) ----

    def get_x_from_col(self, col: int) -> float:
        return board_width / 8 * col + board_width / 16

    def get_y_from_row(self, row: int) -> float:
        return board_height / 8 * row + board_height / 16

    # ---- piece drawing ----

    def display_piece(self, piece, col, row):
        self.pieces_list.append(
            self.canvas.create_image(
                self.get_x_from_col(col),
                self.get_y_from_row(row),
                image=self.img_dict[piece]
            )
        )

    # ---- board update ----

    def update_board(self):
        for piece in self.pieces_list:
            self.canvas.delete(piece)
        self.pieces_list.clear()

        row = col = 0
        for piece in self.board.board_fen():
            if '1' <= piece <= '8':
                col += int(piece)
            elif piece == '/':
                row += 1
                col = 0
            else:
                self.display_piece(piece, col, row)
                col += 1

        if self.board.turn == WHITE:
            self.history_white_listbox.update()
        else:
            self.history_black_listbox.update()

        self.human_controller.maybe_schedule_ai_turn(self.jouer)

    # ---- history ----

    def update_history_white(self, entry):
        self.history_white.append(entry)
        self.history_white_var.set(self.history_white)

    def update_history_black(self, entry):
        self.history_black.append(entry)
        self.history_black_var.set(self.history_black)

    # ---- game loop ----

    def jouer(self):
        if self.board.is_game_over():
            res = self.board.result()
            if res == "1-0":
                msg = "Les blancs ont gagné !"
            elif res == "0-1":
                msg = "Les noirs ont gagné !"
            else:
                msg = "Égalité !"

            # text coords scaled: 240 → 120
            self.canvas.create_text(
                120, 120,
                text=f"Partie terminée : {msg}",
                font=("Arial", 12, "bold"),   # 24 → 12
                fill="red"
            )
            return

        if self.board.turn == WHITE:
            if self.human_white:
                return
            self.board.push(self.Joueur_Blanc.coup(self.board))

        else:
            if self.human_black:
                return
            self.board.push(self.Joueur_Noir.coup(self.board))

        self.update_board()
