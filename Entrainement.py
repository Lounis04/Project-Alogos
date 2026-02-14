import chess
from ia_tree import TreeIA

def train_selfplay(games=50, depth=2):
    """Entraînement séquentiel (une partie à la fois)"""
    ia = TreeIA(depth=depth, transpo_file="coups.json", train_mode=True)
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    print(f"\n=== Entraînement: {games} parties (profondeur {depth}) ===\n")

    for g in range(games):
        print(f"Partie {g+1}/{games}...", end=" ", flush=True)
        board = chess.Board()
        moves = 0

        try:
            while not board.is_game_over():
                move_san = ia.coup(board)
                board.push_san(move_san)
                moves += 1

            result = board.result()
            results[result] += 1
            print(f"{result} ({moves} coups)")

            # Sauvegarde après chaque partie
            ia.save_transpo()
            
        except Exception as e:
            print(f"ERREUR: {e}")
            ia.save_transpo()

    print("\n=== Bilan ===")
    print(f"Blancs gagnent: {results['1-0']}")
    print(f"Noirs gagnent:  {results['0-1']}")
    print(f"Nulles:         {results['1/2-1/2']}")
    print()