import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def list_pgn_files(input_dir: Path):
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".pgn"])


def read_pgn(path: Path) -> str:
    # Read as text; let exceptions surface to fail fast if there is an issue
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def concat_pgns(
    input_dir: str = "pgns-run2-test91-20251106-1254",
    output_path: str = "data/lc0-1.pgn",
    max_workers: int = 8,
):
    input_dir_path = Path(input_dir)
    output_path = Path(output_path)

    if not input_dir_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pgn_files = list_pgn_files(input_dir_path)
    total = len(pgn_files)
    if total == 0:
        raise RuntimeError(f"No PGN files found in {input_dir_path}")

    # Read files in parallel
    contents = [None] * total

    def worker(idx_file):
        idx, file_path = idx_file
        contents[idx] = read_pgn(file_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(worker, enumerate(pgn_files)),
                total=total,
                desc="Reading PGNs",
            )
        )

    # Write concatenated PGNs with simple headers before each game
    with output_path.open("w", encoding="utf-8") as out_f:
        for i, text in enumerate(contents, start=1):
            if not text:
                continue

            # Ensure there is spacing before each game block
            if i > 1:
                out_f.write("\n")

            # Minimal valid PGN-style headers for visual/game separation
            out_f.write(f'[Event "LC0 batch game {i}"]\n')
            out_f.write(f'[Site "local"]\n')
            out_f.write(f'[Round "{i}"]\n')
            out_f.write('[White "Unknown"]\n')
            out_f.write('[Black "Unknown"]\n')
            out_f.write('[Result "*"]\n\n')

            # Write original content; ensure it ends with a blank line
            out_f.write(text)
            if not text.endswith("\n\n"):
                if not text.endswith("\n"):
                    out_f.write("\n")
                out_f.write("\n")


if __name__ == "__main__":
    concat_pgns()
