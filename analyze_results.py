import os
import csv
import chess.pgn
from collections import defaultdict

OLD_CSV = os.path.join("tournament", "results.csv")
NEW_CSV = os.path.join("tournament", "results_corrected.csv")
TOURNAMENT_DIR = "tournament"

def parse_pgn_result(filepath):
    """Return result as 'win', 'loss', or 'draw' for RLModel based on PGN headers."""
    with open(filepath, "r", encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
        if not game:
            return None

        result = game.headers.get("Result", "").strip()
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")

        if result not in ["1-0", "0-1", "1/2-1/2"]:
            return None

        if result == "1/2-1/2":
            return "draw"

        # RLModel win/loss depends on color
        if white == "RLModel" and result == "1-0":
            return "win"
        elif black == "RLModel" and result == "0-1":
            return "win"
        else:
            return "loss"

def analyze_tournament():
    stats = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "games": 0})

    for root, _, files in os.walk(TOURNAMENT_DIR):
        folder = os.path.basename(root)
        if folder == "tournament" or not files:
            continue

        for file in files:
            if file.endswith(".pgn"):
                result = parse_pgn_result(os.path.join(root, file))
                if result:
                    stats[folder]["games"] += 1
                    if result == "win":
                        stats[folder]["wins"] += 1
                    elif result == "loss":
                        stats[folder]["losses"] += 1
                    elif result == "draw":
                        stats[folder]["draws"] += 1

    return stats

def load_old_results():
    if not os.path.exists(OLD_CSV):
        return {}
    old = {}
    with open(OLD_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old[row["Opponent"]] = row
    return old

def compare_results(new_stats, old_stats):
    comparison = []
    for opp, data in new_stats.items():
        old = old_stats.get(opp, None)
        new_score = (data["wins"] + 0.5 * data["draws"]) / data["games"] * 100 if data["games"] else 0
        old_score = float(old["Score%"]) if old else None

        diff = new_score - old_score if old_score is not None else None
        comparison.append({
            "Opponent": opp,
            "Games": data["games"],
            "Wins": data["wins"],
            "Draws": data["draws"],
            "Losses": data["losses"],
            "Score%": f"{new_score:.1f}",
            "OldScore%": f"{old_score:.1f}" if old_score is not None else "",
            "ΔScore": f"{diff:+.1f}" if diff is not None else ""
        })
    return comparison

def save_new_results(comparison):
    with open(NEW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Opponent", "Games", "Wins", "Draws", "Losses", "Score%", "OldScore%", "ΔScore"
        ])
        writer.writeheader()
        writer.writerows(comparison)
    print(f"\n✅ Saved corrected results to: {NEW_CSV}")

def main():
    print("Analyzing PGN results from:", TOURNAMENT_DIR)
    new_stats = analyze_tournament()
    old_stats = load_old_results()
    comparison = compare_results(new_stats, old_stats)
    save_new_results(comparison)

    # Display summary
    print("\n=== Corrected Results ===")
    for row in comparison:
        print(f"{row['Opponent']}: {row['Score%']}%  ({row['Wins']}W/{row['Draws']}D/{row['Losses']}L)  "
              f"Δ={row['ΔScore']}")

if __name__ == "__main__":
    main()
