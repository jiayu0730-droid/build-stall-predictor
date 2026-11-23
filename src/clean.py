"""
Module 1 - Data cleaning (KPI 1, delivery #1)

Takes the raw build export (data/builds_raw.csv) and produces a tidy
table (data/builds_clean.csv) the rest of the pipeline can trust:
  - consistent lowercase category labels
  - numeric columns really numeric
  - missing values filled with the column median, with a flag column
    recording that the value was missing (missingness itself is a signal:
    a build whose quote spread was never recorded is different from one
    with a recorded spread)
  - obvious data errors clipped to sane ranges
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NUMERIC_COLS = ["bom_lines", "quantity", "requested_lead_days",
                "suppliers_quoted", "quote_spread_pct", "design_revisions",
                "decision_reply_days", "days_to_outcome"]
FLAG_COLS = ["has_cad", "has_drawings", "has_bom", "has_prototype",
             "has_photos", "has_requirements_doc"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Category labels: lowercase, no stray spaces
    for col in ["sector", "program_type"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Numerics: force to numbers, bad entries become NaN
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flag columns: only 0/1 allowed
    for col in FLAG_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).astype(int)

    # Missing values: flag them, then fill with the median
    for col in ["quote_spread_pct", "decision_reply_days"]:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())

    # Sane ranges for values that cannot physically leave them
    df["quote_spread_pct"] = df["quote_spread_pct"].clip(0, 150)
    df["decision_reply_days"] = df["decision_reply_days"].clip(0, 30)
    df["requested_lead_days"] = df["requested_lead_days"].clip(1, 365)

    # Target must exist; drop rows without an outcome
    df = df.dropna(subset=["stalled"])
    df["stalled"] = df["stalled"].astype(int)

    return df


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "builds_raw.csv")
    cleaned = clean(raw)
    out = ROOT / "data" / "builds_clean.csv"
    cleaned.to_csv(out, index=False)
    print(f"clean: {len(raw)} rows in -> {len(cleaned)} rows out -> {out}")


if __name__ == "__main__":
    main()
