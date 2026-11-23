"""
Module 5 - Deployment (KPI 1, delivery #5; completes KPI 2's
"from data processing and training to deployment" chain)

Scores ONE new build at intake and prints its stall risk. The score is
not the decision; it tells the team which builds need their unresolved
questions closed first — chase the missing BOM, align the assumptions
behind the quotes, or reset expectations with the customer — while
changing course is still cheap.

Usage:
  python src/predict.py --has-cad 1 --has-drawings 1 --has-bom 0 \
      --quantity 100 --requested-lead-days 30 --quote-spread-pct 45

Risk bands:  <0.35 LOW  |  0.35-0.65 MEDIUM  |  >0.65 HIGH
In production this same function sits behind a small web form or API;
the scoring logic does not change.
"""

import argparse
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def score_build(fields: dict) -> float:
    bundle = joblib.load(ROOT / "models" / "stall_model.joblib")
    model, columns = bundle["model"], bundle["columns"]

    row = {c: 0.0 for c in columns}
    intake = ["has_cad", "has_drawings", "has_bom", "has_prototype",
              "has_photos", "has_requirements_doc"]
    completeness = sum(fields[k] for k in intake) / len(intake)
    row.update({
        "completeness_score": completeness,
        "missing_key_items": 3 - (fields["has_cad"] + fields["has_drawings"] + fields["has_bom"]),
        "lead_pressure": fields["quantity"] / max(fields["requested_lead_days"], 1) / 12.0,
        "log_quantity": float(pd.Series([fields["quantity"]]).pipe(lambda s: __import__("numpy").log1p(s).iloc[0])),
        "quote_spread_pct": fields["quote_spread_pct"],
        "quote_spread_x_incomplete": fields["quote_spread_pct"] * (1 - completeness),
        "design_revisions": fields["design_revisions"],
        "decision_reply_days": fields["decision_reply_days"],
        "suppliers_quoted": fields["suppliers_quoted"],
        "bom_lines": fields["bom_lines"],
        "requested_lead_days": fields["requested_lead_days"],
        "quote_spread_pct_was_missing": 0,
        "decision_reply_days_was_missing": 0,
    })
    sector_col = f"sector_{fields['sector']}"
    prog_col = f"prog_{fields['program_type']}"
    if sector_col in row:
        row[sector_col] = 1.0
    if prog_col in row:
        row[prog_col] = 1.0

    X = pd.DataFrame([row])[columns]
    return float(model.predict_proba(X)[0, 1])


def band(p: float) -> str:
    return "LOW" if p < 0.35 else ("MEDIUM" if p <= 0.65 else "HIGH")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score one build's stall risk at intake.")
    ap.add_argument("--sector", default="robotics")
    ap.add_argument("--program-type", default="pcba")
    for flag in ["has-cad", "has-drawings", "has-bom", "has-prototype",
                 "has-photos", "has-requirements-doc"]:
        ap.add_argument(f"--{flag}", type=int, default=1, choices=[0, 1])
    ap.add_argument("--bom-lines", type=int, default=60)
    ap.add_argument("--quantity", type=int, default=100)
    ap.add_argument("--requested-lead-days", type=int, default=30)
    ap.add_argument("--suppliers-quoted", type=int, default=3)
    ap.add_argument("--quote-spread-pct", type=float, default=25.0)
    ap.add_argument("--design-revisions", type=int, default=1)
    ap.add_argument("--decision-reply-days", type=float, default=3.0)
    args = ap.parse_args()

    fields = {
        "sector": args.sector, "program_type": args.program_type,
        "has_cad": args.has_cad, "has_drawings": args.has_drawings,
        "has_bom": args.has_bom, "has_prototype": args.has_prototype,
        "has_photos": args.has_photos,
        "has_requirements_doc": args.has_requirements_doc,
        "bom_lines": args.bom_lines, "quantity": args.quantity,
        "requested_lead_days": args.requested_lead_days,
        "suppliers_quoted": args.suppliers_quoted,
        "quote_spread_pct": args.quote_spread_pct,
        "design_revisions": args.design_revisions,
        "decision_reply_days": args.decision_reply_days,
    }
    p = score_build(fields)
    print(f"Stall risk: {p:.1%}  ->  {band(p)}")


if __name__ == "__main__":
    main()
