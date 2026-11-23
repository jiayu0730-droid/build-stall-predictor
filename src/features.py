"""
Module 2 - Feature engineering (KPI 1, delivery #2)

Turns the clean table into model-ready features. The central feature is
the completeness score: how much of the engineering package the customer
actually handed over at intake, read as a count of unresolved questions.
In KnowYi's framing, builds stall on information problems at the boundary
between engineering intent and physical production — this one number is
that framing made measurable, and it carries the business logic of the
project. A second key feature treats quote spread the way KnowYi's
quotes-diverge note does: not only a pricing signal, but an ambiguity
signal, which is why it is interacted with incompleteness below.

Features produced:
  - completeness_score   0..1, share of the 6 intake items provided
  - missing_key_items    how many of CAD / drawings / BOM are missing
  - lead_pressure        quantity per requested day (rush signal)
  - log_quantity         quantity on a log scale
  - quote_spread_x_incomplete  interaction: wide quote spread hurts more
                         when the package is incomplete
  - sector / program_type one-hot encoded
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INTAKE_ITEMS = ["has_cad", "has_drawings", "has_bom", "has_prototype",
                "has_photos", "has_requirements_doc"]
KEY_ITEMS = ["has_cad", "has_drawings", "has_bom"]

FEATURE_COLS = ["completeness_score", "missing_key_items", "lead_pressure",
                "log_quantity", "quote_spread_pct", "quote_spread_x_incomplete",
                "design_revisions", "decision_reply_days", "suppliers_quoted",
                "bom_lines", "requested_lead_days",
                "quote_spread_pct_was_missing", "decision_reply_days_was_missing"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["completeness_score"] = df[INTAKE_ITEMS].mean(axis=1)
    df["missing_key_items"] = len(KEY_ITEMS) - df[KEY_ITEMS].sum(axis=1)
    df["lead_pressure"] = df["quantity"] / df["requested_lead_days"].clip(lower=1) / 12.0
    df["log_quantity"] = np.log1p(df["quantity"])
    df["quote_spread_x_incomplete"] = df["quote_spread_pct"] * (1 - df["completeness_score"])

    cats = pd.get_dummies(df[["sector", "program_type"]], prefix=["sector", "prog"])
    X = pd.concat([df[FEATURE_COLS], cats], axis=1)
    return X.astype(float)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "builds_clean.csv")
    X = build_features(df)
    out = ROOT / "data" / "features.csv"
    X.assign(stalled=df["stalled"].values).to_csv(out, index=False)
    print(f"features: {X.shape[1]} columns, {len(X)} rows -> {out}")


if __name__ == "__main__":
    main()
