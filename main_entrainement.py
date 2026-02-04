# main_entrainement.py
import os
import json
import glob
import chess
from multiprocessing import Pool, cpu_count
from ia_tree import TreeIA

TOTAL_GAMES = 5
DEPTH = 3

PER_PROCESS_PREFIX = "coups_temp_"
FINAL_FILE = "coups.json"


def play_one_game(game_id):
    """Joue une partie et sauvegarde dans un fichier temporaire unique"""
    pid = os.getpid()
    transpo_file = f"{PER_PROCESS_PREFIX}{pid}_{game_id}.json"

    # Chaque processus part d'une base vide (fusion après)
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
    """Fusionne tous les fichiers temporaires DANS le fichier principal existant"""
    
    print("\n" + "="*60)
    print("FUSION DES FICHIERS")
    print("="*60)
    
    # 1. Charger le fichier principal existant
    if os.path.exists(output):
        try:
            with open(output, "r") as f:
                merged = json.load(f)
            print(f"[Chargement] {len(merged)} positions existantes dans {output}")
        except Exception as e:
            merged = {}
            print(f"[Attention] Fichier {output} illisible ({e}), redémarrage")
    else:
        merged = {}
        print(f"[Nouveau] Création de {output}")

    positions_avant = len(merged)

    # 2. Fusionner chaque fichier temporaire
    files = glob.glob(pattern)
    print(f"[Fichiers] {len(files)} fichiers temporaires trouvés")
    
    if len(files) == 0:
        print("[Attention] Aucun fichier temporaire trouvé!")
        print(f"[Debug] Pattern de recherche: {pattern}")
        print(f"[Debug] Répertoire actuel: {os.getcwd()}")
        all_json = glob.glob("*.json")
        print(f"[Debug] Tous les .json dans le dossier: {all_json}")
        return
    
    for i, fp in enumerate(files, 1):
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            print(f"  [{i}/{len(files)}] {fp}: {len(data)} positions")
        except Exception as e:
            print(f"  [{i}/{len(files)}] {fp}: ERREUR ({e})")
            continue

        for fen, entry in data.items():
            if fen not in merged:
                # Nouvelle position
                merged[fen] = entry
            else:
                # Position existante: on garde la meilleure
                old_entry = merged[fen]
                
                # Priorité 1: Profondeur plus grande = meilleure analyse
                if entry["depth"] > old_entry["depth"]:
                    merged[fen] = entry
                # Priorité 2: Même profondeur, on garde le meilleur score absolu
                elif entry["depth"] == old_entry["depth"]:
                    if abs(entry["score"]) > abs(old_entry["score"]):
                        merged[fen] = entry

    nouvelles = len(merged) - positions_avant

    # 3. Écriture atomique
    if len(merged) > 0:
        tmp = output + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(merged, f, indent=2)
            os.replace(tmp, output)
            print(f"\n[Succès] Fichier {output} créé/mis à jour")
        except Exception as e:
            print(f"\n[Erreur] Impossible d'écrire {output}: {e}")
            return
    else:
        print("\n[Attention] Aucune position à sauvegarder!")
        return

    # 4. Nettoyage
    print(f"\n[Nettoyage] Suppression des fichiers temporaires...")
    supprimés = 0
    for fp in files:
        try:
            os.remove(fp)
            supprimés += 1
        except Exception as e:
            print(f"  Impossible de supprimer {fp}: {e}")
    
    print(f"[Nettoyage] {supprimés}/{len(files)} fichiers supprimés")
    
    print(f"\n[Résultat] {len(merged)} positions totales (+{nouvelles} nouvelles)")
    print("="*60)


def train_parallel():
    """Lance l'entraînement parallèle"""
    print(f"\n=== Entraînement: {TOTAL_GAMES} parties (profondeur {DEPTH}) ===\n")

    processes = min(TOTAL_GAMES, cpu_count())

    with Pool(processes=processes) as pool:
        results = pool.map(play_one_game, range(TOTAL_GAMES))

    print("\n=== Résultats ===")
    for gid, result, moves, pid, file in results:
        print(f"Partie {gid}: {result} ({moves} coups)")

    print()
    merge_all()
    print("\n=== Entraînement terminé ===\n")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train_parallel()