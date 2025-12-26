"""
Shrink data/sample_ecommerce.csv from ~120MB to ~99MB by random row sampling.

Run:
    python shrink_sample.py
"""

from pathlib import Path
import pandas as pd

# Paths
DATA_DIR = Path("data")
SRC_PATH = DATA_DIR / "sample_ecommerce.csv"
OUT_PATH = DATA_DIR / "sample_ecommerce_99mb.csv"

TARGET_MB = 99.0          # desired size (approx)
CHUNK_SIZE = 500_000      # rows per chunk when reading

def main():
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Source file not found: {SRC_PATH}")

    current_size_mb = SRC_PATH.stat().st_size / (1024 ** 2)
    print(f"Current file: {SRC_PATH}  ~ {current_size_mb:.2f} MB")

    if current_size_mb <= TARGET_MB:
        print("File is already <= target size. Nothing to do.")
        return

    # Fraction of rows to keep (approximate)
    frac = TARGET_MB / current_size_mb
    # add a tiny safety margin, so we end up slightly below 99MB
    frac *= 0.97

    frac = min(max(frac, 0.05), 1.0)  # keep between 5% and 100%
    print(f"Sampling approximately {frac*100:.1f}% of rows")

    first_chunk = True
    rows_kept = 0

    for i, chunk in enumerate(pd.read_csv(SRC_PATH, chunksize=CHUNK_SIZE)):
        sampled = chunk.sample(frac=frac, random_state=42)
        rows_kept += len(sampled)

        mode = "w" if first_chunk else "a"
        header = first_chunk

        sampled.to_csv(OUT_PATH, mode=mode, header=header, index=False)
        first_chunk = False

        print(f"Chunk {i+1}: sampled {len(sampled):,} rows (total kept: {rows_kept:,})")

    out_size_mb = OUT_PATH.stat().st_size / (1024 ** 2)
    print(f"\nDone! Output file: {OUT_PATH}  ~ {out_size_mb:.2f} MB")

    print(
        "\nIf the size is close enough to 99MB, you can now replace "
        "`sample_ecommerce.csv` with this file:"
        "\n    - delete old sample_ecommerce.csv"
        "\n    - rename sample_ecommerce_99mb.csv -> sample_ecommerce.csv"
    )


if __name__ == "__main__":
    main()
