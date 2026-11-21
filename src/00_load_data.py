"""Stage 00: validate the raw gas-turbine data and write the modeling CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pems_common import (
    ANALYSIS,
    COLUMNS,
    EXPECTED_ROWS,
    EXPECTED_YEAR_COUNTS,
    PROCESSED_PATH,
    RAW_PATH,
    ensure_directories,
)

YEAR_SLICES = [
    (2011, 0, 7411),
    (2012, 7411, 15039),
    (2013, 15039, 22191),
    (2014, 22191, 29349),
    (2015, 29349, 36733),
]


def _add_year_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    if "year" in df.columns:
        return df
    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Cannot infer year: expected {EXPECTED_ROWS} rows, got {len(df)}"
        )
    years = np.empty(len(df), dtype=int)
    for year, start, stop in YEAR_SLICES:
        years[start:stop] = year
    df = df.copy()
    df.insert(0, "year", years)
    return df


def load_data(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Load, validate and return the canonical 36,733 x 12 modeling frame."""
    df = pd.read_csv(path)
    df = _add_year_if_missing(df)

    if list(df.columns) != COLUMNS:
        raise ValueError(f"Unexpected schema: {list(df.columns)}")
    if df.shape != (EXPECTED_ROWS, len(COLUMNS)):
        raise ValueError(f"Unexpected shape: {df.shape}")

    counts = df["year"].value_counts().sort_index().to_dict()
    if counts != EXPECTED_YEAR_COUNTS:
        raise ValueError(f"Unexpected year counts: {counts}")

    for column in ANALYSIS:
        df[column] = pd.to_numeric(df[column], errors="raise").astype(float)
    values = df[ANALYSIS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values found in the analysis columns")
    if df[ANALYSIS].isna().any().any():
        raise ValueError("Missing values found in the analysis columns")

    return df


def main() -> pd.DataFrame:
    ensure_directories()
    df = load_data()
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed shape: {df.shape}")
    print("Year counts:")
    print(df["year"].value_counts().sort_index().to_string())
    print(f"Missing analysis values: {int(df[ANALYSIS].isna().sum().sum())}")
    print(f"Wrote {PROCESSED_PATH}")
    return df


if __name__ == "__main__":
    main()
