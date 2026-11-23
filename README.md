# Build Stall Predictor — KnowYi ML Internship Project

**One-line summary:** Given what a customer hands over at intake (CAD, drawings, BOM, prototype, photos, requirements) plus the quoting picture, predict whether the build will stall before completion — early enough for the team to fix the package instead of firefighting later.

This is the shape of KnowYi's core problem in its own words: builds stall on *blocking decisions* and *incomplete engineering inputs*, not on machining. The model turns that thesis into a number at intake time.

## Status (2026-11-20)

Full pipeline built and verified end to end on **synthetic practice data** modeled on KnowYi's public business description. KnowYi's real build/quote records are private and have not been shared yet; when they arrive, the real export replaces `data/builds_raw.csv` and every step below runs unchanged.

## Results on practice data

| | CV AUC (5-fold) | Test AUC | Test accuracy |
|---|---|---|---|
| Baseline (logistic regression) | 0.729 | 0.715 | 0.713 |
| Improved (random forest) | 0.717 | 0.724 | 0.704 |

Dataset: 1,200 builds, 36.6% stalled. Top drivers of stall risk, in order:
quote spread × package incompleteness, design revisions, quote spread, lead-time pressure, completeness score, customer decision reply time.

**How to read this for KPI 3:** the baseline score is the reference point. The target value in the issue should be set *after* the baseline exists on real data (baseline + an agreed margin), never fixed before the data is seen.

Deployment demo (`src/predict.py`):
- Incomplete rush build (no CAD, no BOM, 55% quote spread, 500 units in 14 days): **53.5% stall risk — MEDIUM**
- Complete package, tight quotes, relaxed lead time: **7.3% stall risk — LOW**

## How it maps to the internship KPIs

| KPI | Where it lives |
|---|---|
| 1. One usable module every two weeks, merged to main | The five modules below, one per delivery |
| 2. One complete project, data → training → deployment, owned independently | This repo, end to end |
| 3. Model hits the target value on the agreed metric, written in the issue | `reports/metrics.json`; target = real-data baseline + margin, agreed with Ella and written into the issue |

## The five modules

1. `src/clean.py` — cleaning: consistent labels, numeric types, missing values flagged then median-filled, sane ranges.
2. `src/features.py` — features: **Package Completeness Score** (share of the 6 intake items provided), missing key items, lead pressure, quote-spread × incompleteness interaction, one-hot sector/program.
3. `src/train.py` (baseline) — logistic regression, the honest reference score.
4. `src/train.py` (improved + evaluation) — random forest, 5-fold CV + held-out test, feature importance, `reports/metrics.json`, saved model.
5. `src/predict.py` — deployment: score one new build at intake from the command line; same function drops behind a web form/API unchanged.

Supporting files: `data/generate_synthetic_data.py` (practice-data generator, documents exactly which real-world causes the synthetic stalls were built from), `run_all.py` (one-command end-to-end run).

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

## Swapping in real data

1. Export KnowYi's historical builds/quotes to CSV with the same columns as `data/builds_raw.csv` (see the generator for the exact schema).
2. Save it as `data/builds_raw.csv`, skip step 1 of `run_all.py`, run steps 2–5.
3. Compare the real baseline AUC with the practice one; agree the KPI 3 target with Ella and write it into the issue.
