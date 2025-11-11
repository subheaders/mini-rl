import os
import torch
import chess
import chess.pgn
from datetime import datetime
from model import load_model
import torch.nn.functional as F
from utils.pgn_dataset import encode_board

# === Configuration ===
NEW_MODEL_PATH = "chess_model-large.pt"
OLD_MODEL_PATH = "old.pt"
TEMPERATURE = 0.8
USE_BFLOAT16 = True
OUTPUT_PGN_PATH = "models_game.pgn"

# === Device selection ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prefer MPS on Apple Silicon when available
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"

print(f"Using device: {DEVICE}")


# === Move selection (shared) ===
def select_model_move(board, policy_logits):
    """
    Select a move using the same 4672-index move encoding as training.
    """
    legal_moves = list(board.legal_moves)
    scores = []

    for move in legal_moves:
        idx = _move_to_index_play(move, board)
        if idx is None:
            continue
        scores.append((move, policy_logits[0, idx].item()))

    if not scores:
        # Fallback: if mapping fails for some reason, pick a random legal move
        return legal_moves[0]

    moves, raw_scores = zip(*scores)
    probs = F.softmax(torch.tensor(raw_scores) / TEMPERATURE, dim=0)
    move = moves[torch.multinomial(probs, 1).item()]
    return move


def _move_to_index_play(move: chess.Move, board: chess.Board):
    """
    Local copy of move_to_index from utils.pgn_dataset to ensure consistency
    with the 4672-policy head layout used during pretraining.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    # Castling (2 planes)
    if board.is_castling(move):
        plane = 0 if chess.square_file(to_sq) > chess.square_file(from_sq) else 1
        return from_sq * 73 + plane

    # Promotions (16 planes)
    promotion = move.promotion
    if promotion is not None:
        df = chess.square_file(to_sq) - chess.square_file(from_sq)
        dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

        dir_idx = None
        if dr == 1 and df == 0:
            dir_idx = 0
        elif dr == 1 and df == 1:
            dir_idx = 1
        elif dr == 1 and df == -1:
            dir_idx = 2
        elif dr == 2 and df == 0:
            dir_idx = 3
        if dir_idx is None:
            return None

        promo_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        if promotion not in promo_pieces:
            return None
        piece_idx = promo_pieces.index(promotion)

        plane = 2 + dir_idx * 4 + piece_idx
        return from_sq * 73 + plane

    # Normal moves: 56 planes
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
        (1, 2), (2, 1), (-1, 2), (-2, 1),
        (1, -2), (2, -1), (-1, -2), (-2, -1),
    ]

    plane_offset = 18
    plane = None

    for d_idx, (dx, dy) in enumerate(directions):
        if d_idx < 8:
            for dist in range(1, 8):
                if df == dx * dist and dr == dy * dist:
                    plane = plane_offset + d_idx * 7 + (dist - 1)
                    break
            if plane is not None:
                break
        else:
            if df == dx and dr == dy:
                plane = plane_offset + 8 * 7 + (d_idx - 8)
                break

    if plane is None:
        return None

    idx = from_sq * 73 + (plane - plane_offset)
    if 0 <= idx < 4672:
        return idx
    return None


# === Play a Single Game: new model vs old model ===
def play_game(new_model, old_model):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers.update({
        "Event": "NewModel vs OldModel Self-Play",
        "Site": "Localhost",
        "Date": datetime.now().strftime("%Y.%m.%d"),
        "Round": "1",
        "White": "NewModel",
        "Black": "OldModel",
        "Result": "*",
    })
    node = game

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            model = new_model
            color = "White"
        else:
            model = old_model
            color = "Black"

        # Autocast:
        # - CUDA: bf16/fp16 via torch.cuda.amp.autocast
        # - MPS: bf16 via torch.amp.autocast("mps")
        # - CPU: fp32 (no-op context)
        if DEVICE == "cuda":
            precision_dtype = torch.bfloat16 if USE_BFLOAT16 else torch.float16
            autocast_ctx = torch.cuda.amp.autocast(dtype=precision_dtype)
        elif DEVICE == "mps":
            autocast_ctx = torch.amp.autocast("mps", dtype=torch.bfloat16)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        with torch.no_grad(), autocast_ctx:
            inp = encode_board(board).unsqueeze(0).to(DEVICE)
            policy, value = model(inp)
            move = select_model_move(board, policy)

        board.push(move)
        node = node.add_variation(move)
        # Attach eval from the side that just moved
        node.comment = f"{color} Eval: {float(value.item()):+.3f}"

    game.headers["Result"] = board.result()
    with open(OUTPUT_PGN_PATH, "w", encoding="utf-8") as f:
        print(game, file=f)

    print(f"Game finished: {board.result()} (PGN saved to {OUTPUT_PGN_PATH})")


# === Main Entry ===
def main():
    print(f"Loading new model from: {NEW_MODEL_PATH}")
    new_model = load_model(NEW_MODEL_PATH, DEVICE)
    new_model.to(DEVICE)
    new_model.eval()

    print(f"Loading old model from: {OLD_MODEL_PATH}")
    old_model = load_model(OLD_MODEL_PATH, DEVICE)
    old_model.to(DEVICE)
    old_model.eval()

    # Print parameter counts for debugging / parity
    def count_params(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    new_total, new_trainable = count_params(new_model)
    old_total, old_trainable = count_params(old_model)

    print(f"NewModel parameters: total={new_total:,}, trainable={new_trainable:,}")
    print(f"OldModel parameters: total={old_total:,}, trainable={old_trainable:,}")

    play_game(new_model, old_model)


if __name__ == "__main__":
    main()
