# main_entrainement.py
import os
import json
import glob
import chess
import shutil
from multiprocessing import Pool, cpu_count
from ia_tree import TreeIA

TOTAL_GAMES = 10
DEPTH = 3

PER_PROCESS_PREFIX = "coups_"
FINAL_FILE = "coups.json"


def play_one_game(game_id):
    pid = os.getpid()
    transpo_file = f"{PER_PROCESS_PREFIX}{pid}.json"

    # Si coups.json existe, on le copie comme base
    if os.path.exists(FINAL_FILE):
        shutil.copy(FINAL_FILE, transpo_file)

    ia = TreeIA(depth=DEPTH, transpo_file=transpo_file, train_mode=True)

    board = chess.Board()
    moves = 0

    try:
        while not board.is_game_over():
            move_san = ia.coup(board)
            board.push_san(move_san)
            moves += 1
    except Exception as e:
        ia.save_transpo()
        return (game_id, f"ERROR: {e}", moves, pid, transpo_file)

    ia.save_transpo()
    return (game_id, board.result(), moves, pid, transpo_file)


def merge_all(pattern=f"{PER_PROCESS_PREFIX}*.json", output=FINAL_FILE):
    merged = {}

    files = glob.glob(pattern)
    for fp in files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except:
            continue

        for fen, entry in data.items():
            if fen not in merged:
                merged[fen] = entry
            else:
                if entry["depth"] > merged[fen]["depth"]:
                    merged[fen] = entry
                elif entry["depth"] == merged[fen]["depth"]:
                    if entry["score"] > merged[fen]["score"]:
                        merged[fen] = entry

    # écriture
    tmp = output + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, output)

    # on supprime les fichiers temporaires
    for fp in files:
        try:
            os.remove(fp)
        except:
            pass

    print(f"[✓] Fusion terminée : {len(merged)} positions → {output}")


def train_parallel():
    processes = min(TOTAL_GAMES, cpu_count())

    with Pool(processes=processes) as pool:
        results = pool.map(play_one_game, range(TOTAL_GAMES))

    print("\n=== Résultats des parties ===")
    for gid, result, moves, pid, file in results:
        print(f"Partie {gid}: {result} ({moves} coups) – fichier {file}")

    merge_all()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train_parallel()