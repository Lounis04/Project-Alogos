from chess import *
from img.canvas_tkinter import *
from ia_tree import TreeIA

board = Board()
root = Tk()
root.title("Echecs")

#Humain
ia_blanc = None

#IA
ia_noir = TreeIA(depth=4, transpo_file="coups.json", train_mode=False)

c = Chess_UI(root, board, ia_blanc, ia_noir)

root.mainloop()
