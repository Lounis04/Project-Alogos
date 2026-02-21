import glob
import json
import os
import shutil
from multiprocessing import Pool, cpu_count
import chess

from ia_tree import TreeIA

TOTAL_GAMES: int = 2          
DEPTH:       int = 3      
PER_PROCESS_PREFIX: str = "coups_"
FINAL_FILE: str = "coups.json"

def play_one_game(game_id: int) -> tuple[int, str, int, int, str]:
    """Joue une partie complète IA vs IA et sauvegarde les transpositions locales.
    Entrée  : game_id ,identifiant unique de la partie (utilisé pour les logs).
    Sortie  : tuple (game_id, résultat, nombre_de_coups, pid, chemin_fichier)résultat : chaîne chess ('1-0', '0-1', '1/2-1/2') ou message d'erreur préfixé par 'ERROR:"""
    pid: int = os.getpid()
    transpo_file: str = f"{PER_PROCESS_PREFIX}{pid}.json"

    if os.path.exists(FINAL_FILE):
        shutil.copy(FINAL_FILE, transpo_file)

    ia: TreeIA = TreeIA(depth=DEPTH, transpo_file=transpo_file, train_mode=True)

    board: chess.Board = chess.Board()
    moves: int = 0

    try:
        while not board.is_game_over():
            move: chess.Move = ia.coup(board)
            board.push(move)
            moves += 1
    except Exception as e:
        ia.save_transpo()
        return (game_id, f"ERROR: {e}", moves, pid, transpo_file)

    ia.save_transpo()
    return (game_id, board.result(), moves, pid, transpo_file)


def merge_all(pattern: str = f"{PER_PROCESS_PREFIX}*.json",output:  str = FINAL_FILE) -> None:
    """Fusionne tous les fichiers de transpositions temporaires dans un fichier unique.En cas de doublon de position (FEN), conserve l'entrée de profondeur maximale ;
    à profondeur égale, conserve l'entrée au score le plus élevé.Écrit de manière atomique via un fichier temporaire (.tmp) puis renommage.Supprime les fichiers temporaires une fois la fusion effectuée.
    Entrée  : pattern , motif glob des fichiers à fusionner (défaut : 'coups_*.json').output ,  chemin du fichier de sortie fusionné (défaut : 'coups.json').
    Sortie  : aucune (effet de bord : écriture de output sur disque, suppressiondes fichiers temporaires, affichage du bilan en console)."""
    merged: dict[str, dict] = {}

    files: list[str] = glob.glob(pattern)
    for fp in files:
        try:
            with open(fp, "r") as f:
                data: dict[str, dict] = json.load(f)
        except Exception:
            continue

        for fen, entry in data.items():
            if fen not in merged:
                merged[fen] = entry
            else:
                # Conserve l'entrée la plus profonde / au meilleur score
                if entry["depth"] > merged[fen]["depth"]:
                    merged[fen] = entry
                elif entry["depth"] == merged[fen]["depth"]:
                    if entry["score"] > merged[fen]["score"]:
                        merged[fen] = entry

    # Écriture atomique
    tmp: str = output + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, output)

    # Nettoyage des fichiers temporaires
    for fp in files:
        try:
            os.remove(fp)
        except Exception:
            pass

    print(f"Fusion terminée : {len(merged)} positions → {output}")


def train_parallel() -> None:
    """Orchestre l'entraînement parallèle : lance les parties en multiprocessingpuis fusionne les résultats.
    Entrée  : aucune (utilise les constantes TOTAL_GAMES et cpu_count()).
    Sortie  : aucune (effets de bord : affichage des résultats, mise à jour de coups.json)."""
    processes: int = min(TOTAL_GAMES, cpu_count())

    with Pool(processes=processes) as pool:
        results: list[tuple[int, str, int, int, str]] = pool.map(play_one_game, range(TOTAL_GAMES))

    print("\n=== Résultats des parties ===")
    for gid, result, moves, pid, file in results:
        print(f"Partie {gid}: {result} ({moves} coups) – fichier {file}")

    merge_all()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train_parallel()