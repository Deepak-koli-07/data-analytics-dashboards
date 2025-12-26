# preprocess_sample.py

import pandas as pd
from pathlib import Path

# path to your big CSV
RAW_PATH = Path("2019-Nov.csv")

# where to store the small file
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "sample_ecommerce.csv"

# how many rows you want in the sample (1M is plenty)
TARGET_ROWS = 1_000_000
CHUNKSIZE = 500_000         # rows per chunk to read
SAMPLE_FRAC = 0.10          # up to 10% per chunk (we cap with TARGET_ROWS)

USECOLS = [
    "event_time",
    "event_type",
    "product_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {RAW_PATH.resolve()}")

    print(f"Reading from: {RAW_PATH}")
    print(f"Writing sample to: {OUT_PATH}")
    print(f"Target rows: {TARGET_ROWS:,}")

    total_collected = 0
    samples = []

    reader = pd.read_csv(
        RAW_PATH,
        chunksize=CHUNKSIZE,
        usecols=USECOLS,
    )

    for i, chunk in enumerate(reader, start=1):
        if total_collected >= TARGET_ROWS:
            break

        # how many rows we can still add
        remaining = TARGET_ROWS - total_collected
        # tentative sample size from this chunk
        tentative = int(len(chunk) * SAMPLE_FRAC)
        n_take = min(remaining, tentative)

        if n_take <= 0:
            continue

        sample_chunk = chunk.sample(n=n_take, random_state=42)
        samples.append(sample_chunk)
        total_collected += len(sample_chunk)

        print(f"Chunk {i}: took {len(sample_chunk):,} rows "
              f"(total {total_collected:,})")

    if not samples:
        print("No samples collected – check file / settings.")
        return

    sample_df = pd.concat(samples, ignore_index=True)

    # optional: ensure datetimes & event_date
    sample_df["event_time"] = pd.to_datetime(sample_df["event_time"])
    sample_df["event_date"] = sample_df["event_time"].dt.date

    sample_df.to_csv(OUT_PATH, index=False)
    print(f"\nDone! Final sample shape: {sample_df.shape}")
    print(f"Saved to: {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
