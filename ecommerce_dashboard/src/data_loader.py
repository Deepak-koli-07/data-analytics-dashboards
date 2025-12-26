# src/data_loader.py
"""
Data loading utilities for the ecommerce dashboard.

Expected CSV columns (from Kaggle dataset):
- event_time
- event_type
- product_id
- category_id
- category_code
- brand
- price
- user_id
- user_session
"""

from pathlib import Path
import pandas as pd

# Path: project_root/data/sample_ecommerce.csv
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_ecommerce.csv"

# Hard cap for dashboard rows (tune this for speed vs. detail)
TARGET_ROWS = 50_000   # <- 50k rows


def load_data() -> pd.DataFrame:
    """
    Load the ecommerce clickstream data and downsample it to ~TARGET_ROWS
    to keep the Streamlit app snappy.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV file not found at: {DATA_PATH}")

    dtypes = {
        "event_type": "category",
        "product_id": "Int64",
        "category_id": "Int64",
        "category_code": "string",
        "brand": "string",
        "price": "float32",
        "user_id": "Int64",
        "user_session": "string",
    }

    df = pd.read_csv(
        DATA_PATH,
        dtype=dtypes,
        parse_dates=["event_time"],
    )

    # Downsample to at most TARGET_ROWS rows (random but reproducible)
    if len(df) > TARGET_ROWS:
        df = df.sample(n=TARGET_ROWS, random_state=42).sort_values("event_time")

    df["user_id"] = df["user_id"].astype("Int64").astype("string")
    df["user_session"] = df["user_session"].astype("string")

    df["event_date"] = df["event_time"].dt.date
    df["category_code"] = df["category_code"].astype("string")
    df["brand"] = df["brand"].astype("string")

    return df
