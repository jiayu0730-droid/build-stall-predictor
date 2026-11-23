"""
Synthetic dataset for the KnowYi build-stall prediction project.

WHY SYNTHETIC: KnowYi's real build/quote records are private and have not
been shared yet. This generator creates practice data shaped exactly like
the records KnowYi's own public materials describe, so the whole pipeline
(cleaning -> features -> model -> deployment) can be built and tested now.
When the real export arrives, it replaces data/builds_raw.csv and every
script runs unchanged.

Each row = one customer build that arrived at KnowYi with some mix of
engineering inputs (CAD, drawings, BOM, prototype, photos, requirements),
matching how KnowYi describes its intake.

The outcome we predict: stalled (1) = the build stalled before completion
because of unresolved blocking decisions; stalled (0) = it completed.

The stall probability is driven by realistic causes taken from KnowYi's
public knowledge library:
  - incomplete engineering inputs (missing BOM / CAD / drawings)
  - quote divergence between suppliers (a sign of an unclear package)
  - many design revisions after kickoff
  - lead-time pressure (requested lead time too short for the quantity)
  - low decision responsiveness from the customer side
plus noise, so the model has something real but imperfect to learn.
"""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

N = 1200  # past builds

SECTORS = ["robotics", "industrial_systems", "automotive_electronics",
           "diagnostics", "avionics"]
PROGRAM_TYPES = ["machined_components", "pcba", "enclosures",
                 "harnesses", "electromechanical_assemblies"]

sector = rng.choice(SECTORS, size=N, p=[0.34, 0.24, 0.18, 0.12, 0.12])
program_type = rng.choice(PROGRAM_TYPES, size=N,
                          p=[0.26, 0.26, 0.18, 0.12, 0.18])

# --- What the customer handed over at intake (1 = provided, 0 = missing) ---
has_cad = rng.binomial(1, 0.78, N)
has_drawings = rng.binomial(1, 0.72, N)
has_bom = rng.binomial(1, 0.64, N)
has_prototype = rng.binomial(1, 0.41, N)
has_photos = rng.binomial(1, 0.55, N)
has_requirements_doc = rng.binomial(1, 0.58, N)

bom_lines = np.where(has_bom == 1, rng.integers(5, 220, N), 0)
quantity = rng.choice([10, 25, 50, 100, 250, 500, 1000], size=N,
                      p=[0.18, 0.20, 0.22, 0.18, 0.12, 0.07, 0.03])
requested_lead_days = rng.integers(7, 90, N)
suppliers_quoted = rng.integers(1, 7, N)

# Quote divergence: how far apart the highest and lowest supplier quotes
# were, as a % of the lowest. Unclear packages diverge more.
base_spread = rng.normal(18, 8, N)
unclear_boost = (1 - has_bom) * 14 + (1 - has_cad) * 9 + (1 - has_drawings) * 7
quote_spread_pct = np.clip(base_spread + unclear_boost + rng.normal(0, 5, N), 2, 95)

design_revisions = rng.poisson(1.6, N) + (1 - has_requirements_doc) * rng.poisson(1.2, N)
decision_reply_days = np.clip(rng.normal(3.2, 2.0, N) + (1 - has_requirements_doc) * 1.5, 0.2, 14)

# --- Hidden stall risk the model must learn (never given to the model) ---
completeness = (has_cad + has_drawings + has_bom + has_prototype
                + has_photos + has_requirements_doc) / 6.0
lead_pressure = np.clip(quantity / np.maximum(requested_lead_days, 1) / 12.0, 0, 2)

logit = (-1.55
         - 2.10 * completeness
         + 0.028 * quote_spread_pct
         + 0.34 * design_revisions
         + 0.16 * decision_reply_days
         + 0.55 * lead_pressure
         + rng.normal(0, 0.55, N))
p_stall = 1 / (1 + np.exp(-logit))
stalled = rng.binomial(1, p_stall)

days_to_outcome = np.where(
    stalled == 1,
    rng.integers(15, 120, N),
    np.clip(requested_lead_days + rng.integers(-6, 15, N), 5, 150))

df = pd.DataFrame({
    "build_id": [f"B-{i:04d}" for i in range(1, N + 1)],
    "sector": sector,
    "program_type": program_type,
    "has_cad": has_cad,
    "has_drawings": has_drawings,
    "has_bom": has_bom,
    "has_prototype": has_prototype,
    "has_photos": has_photos,
    "has_requirements_doc": has_requirements_doc,
    "bom_lines": bom_lines,
    "quantity": quantity,
    "requested_lead_days": requested_lead_days,
    "suppliers_quoted": suppliers_quoted,
    "quote_spread_pct": np.round(quote_spread_pct, 1),
    "design_revisions": design_revisions,
    "decision_reply_days": np.round(decision_reply_days, 1),
    "stalled": stalled,
    "days_to_outcome": days_to_outcome,
})

# Sprinkle in the kind of mess real exports have: some blanks, some
# inconsistent casing, so the cleaning module has honest work to do.
mess = df.copy()
idx = rng.choice(N, size=40, replace=False)
mess.loc[idx[:20], "quote_spread_pct"] = np.nan
mess.loc[idx[20:30], "decision_reply_days"] = np.nan
mess.loc[idx[30:], "sector"] = mess.loc[idx[30:], "sector"].str.upper()

out = Path(__file__).resolve().parent / "builds_raw.csv"
mess.to_csv(out, index=False)
print(f"Wrote {out}  rows={len(mess)}  stall_rate={mess['stalled'].mean():.1%}")
