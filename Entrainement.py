# Entrainement.py
import chess
from ia_tree import TreeIA

def train_selfplay(games=50, depth=2):
    ia = TreeIA(depth=depth, transpo_file="coups.json", train_mode=True)
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    for g in range(games):
        print(f"\n=== Partie {g+1}/{games} ===")
        board = chess.Board()

        while not board.is_game_over():
            move_san = ia.coup(board)
            board.push_san(move_san)

        result = board.result()
        results[result] += 1
        print("Résultat :", result)

        # Sauvegarde JSON uniquement en mode entraînement
        ia.save_transpo()

    print("\n=== Bilan des parties ===")
    print("Blancs gagnent :", results["1-0"])
    print("Noirs gagnent  :", results["0-1"])
    print("Nuls           :", results["1/2-1/2"])
