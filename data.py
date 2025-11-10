import os
import random
import string
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
import chess
import chess.pgn
import chess.engine

# === CONFIGURATION ===
TOURNAMENT_DIR = "data"
PGN_OUTPUT = "engine.pgn"
GAMES_PER_PAIR = 100
THREADS = 5

# Engine paths
STOCKFISH_PATH = r"stockfish/stockfish.exe"
MAIA_ENGINE_PATH = r"C:\Users\rukia\Desktop\mini-rl\maia\lc0.exe"

# Maia weights and nominal Elo levels
MAIA_WEIGHTS = {
    "maia-1100": "maia-1100.pb.gz",
    "maia-1200": "maia-1200.pb.gz",
    "maia-1300": "maia-1300.pb.gz",
    "maia-1400": "maia-1400.pb.gz",
    "maia-1500": "maia-1500.pb.gz",
    "maia-1600": "maia-1600.pb.gz",
    "maia-1700": "maia-1700.pb.gz",
    "maia-1800": "maia-1800.pb.gz",
    "maia-1900": "maia-1900.pb.gz",
}

# Stockfish Elo levels (UCI_Elo mode)
STOCKFISH_LEVELS = [1600, 2000, 2400, 2800]


# === UTILS ===
def random_id(n=8):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


from threading import Lock
lc0_lock = Lock()

def setup_engine(engine_path, maia_weight=None, stockfish_elo=None):
    import time
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    time.sleep(0.25)

    if "stockfish" in engine_path.lower():
        if stockfish_elo:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
    else:
        with lc0_lock:
            if maia_weight:
                abs_weight = os.path.abspath(os.path.join(os.path.dirname(MAIA_ENGINE_PATH), maia_weight))
                engine.configure({"WeightsFile": abs_weight})
                time.sleep(0.25)

    return engine



def play_single_game(white_engine, black_engine, white_name, black_name):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = f"{white_name} vs {black_name}"
    game.headers["Site"] = "Localhost"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = "*"

    node = game

    while not board.is_game_over():
        engine = white_engine if board.turn == chess.WHITE else black_engine
        try:
            result = engine.play(board, chess.engine.Limit(time=0.05))
        except chess.engine.EngineTerminatedError:
            break
        except Exception:
            break

        board.push(result.move)
        node = node.add_variation(result.move)

    game.headers["Result"] = board.result()
    return game


def play_match(white_name, black_name, white_path, black_path, white_weight=None, black_weight=None, white_elo=None, black_elo=None):
    games = []
    try:
        white_engine = setup_engine(white_path, maia_weight=white_weight, stockfish_elo=white_elo)
        black_engine = setup_engine(black_path, maia_weight=black_weight, stockfish_elo=black_elo)

        for _ in range(GAMES_PER_PAIR):
            g = play_single_game(white_engine, black_engine, white_name, black_name)
            games.append(g)

        white_engine.quit()
        black_engine.quit()
    except Exception as e:
        print(f"[ERROR] {white_name} vs {black_name}: {e}")
    return games


def all_pairs():
    pairs = []

    # Maia vs Maia
    maia_items = list(MAIA_WEIGHTS.items())
    for i in range(len(maia_items)):
        for j in range(i + 1, len(maia_items)):
            wn, wf = maia_items[i]
            bn, bf = maia_items[j]
            pairs.append((wn, bn, MAIA_ENGINE_PATH, MAIA_ENGINE_PATH, wf, bf, None, None))

    # Stockfish vs Maia
    for sf_elo in STOCKFISH_LEVELS:
        for mn, mw in MAIA_WEIGHTS.items():
            pairs.append((f"stockfish-{sf_elo}", mn, STOCKFISH_PATH, MAIA_ENGINE_PATH, None, mw, sf_elo, None))

    # Maia vs Stockfish
    for mn, mw in MAIA_WEIGHTS.items():
        for sf_elo in STOCKFISH_LEVELS:
            pairs.append((mn, f"stockfish-{sf_elo}", MAIA_ENGINE_PATH, STOCKFISH_PATH, mw, None, None, sf_elo))

    return pairs


# === MAIN ===
def main():
    os.makedirs(TOURNAMENT_DIR, exist_ok=True)
    pairs = all_pairs()

    print(f"🎮 Starting engine-only tournament ({len(pairs)} matchups, {GAMES_PER_PAIR} games each)...")

    all_games = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = []
        for (wn, bn, wp, bp, ww, bw, we, be) in pairs:
            futures.append(executor.submit(play_match, wn, bn, wp, bp, ww, bw, we, be))

        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            all_games.extend(f.result())

    print(f"✅ Completed {len(all_games)} games total.")

    with open(PGN_OUTPUT, "w", encoding="utf-8") as f:
        for g in all_games:
            print(g, file=f, end="\n\n")

    print(f"📁 All games saved to: {os.path.abspath(PGN_OUTPUT)}")


if __name__ == "__main__":
    main()
