# Build Stall Predictor

The engineering work looks finished, but the build does not move.

A drawing can be technically complete and still leave questions that only surface in production. A BOM can be complete and still carry unresolved assumptions about substitutes, lifecycle, or assembly. KnowYi describes these as information problems at the boundary between engineering intent and physical production. This project asks whether that boundary can be read earlier — at intake, before quoting starts — and turned into a number the team can act on.

The predictor reads a build the way it arrives: what the customer handed over (CAD, drawings, BOM, prototype, photos, requirements), how far apart the supplier quotes came back, how many times the design has already changed, and how much time the schedule actually allows. In KnowYi's terms, a wide quote spread is not only a pricing signal. It can be an ambiguity signal — evidence that each supplier filled the gaps in the package differently. The model combines those signals into one stall risk, scored the day the build arrives.

Engineering intent → manufacturing decision → physical outcome → better next decision. The predictor sits between the first two steps: fewer unresolved questions reaching the quoting stage, and a shorter path from engineering files to delivered hardware.

## Status

Practice build, end to end, on synthetic data modeled on KnowYi's public description of its intake flow. KnowYi's real build records are not public and have not been shared yet. When they are, the real export replaces `data/builds_raw.csv` and every step below runs unchanged. Nothing in the results section should be read as a result on real builds.

## What the practice data showed

1,200 synthetic builds, 36.6% stalled.

| | CV AUC (5-fold) | Test AUC | Test accuracy |
|---|---|---|---|
| Baseline (logistic regression) | 0.729 | 0.715 | 0.713 |
| Improved (random forest) | 0.717 | 0.724 | 0.704 |

What drove the risk, in order: quote spread combined with an incomplete package, design revisions, quote spread on its own, lead-time pressure, package completeness, customer decision time. The pattern matches the thesis: builds rarely stall on machining. They stall on what was never resolved before production started.

Scoring one build at intake (`src/predict.py`):
- Incomplete rush build (no CAD, no BOM, 55% quote spread, 500 units in 14 days): **53.5% stall risk — MEDIUM**
- Complete package, tight quotes, relaxed lead time: **7.3% stall risk — LOW**

The score is not the decision. The score tells the team which builds need their unresolved questions closed first — chase the missing BOM, align the assumptions behind the quotes, reset the delivery expectation — while changing suppliers is still cheap.

## How it is built

Five modules, each small enough to review on its own:

1. `src/clean.py` — cleaning. Consistent labels, numeric types, missing values flagged and filled. A missing value is kept visible rather than erased: a build whose quote spread was never recorded is a different build from one with a recorded spread.
2. `src/features.py` — features. The central one is a completeness score: the share of the engineering package actually handed over, read as a count of unresolved questions. Plus missing key items, lead pressure, and the interaction between quote spread and incompleteness.
3. `src/train.py` (baseline) — logistic regression. The honest reference score, kept deliberately simple.
4. `src/train.py` (improved + evaluation) — random forest, 5-fold cross-validation and a held-out test split, feature importance, `reports/metrics.json`, saved model.
5. `src/predict.py` — deployment. Scores one new build at intake from the command line. The same function sits behind a web form or API unchanged.

Supporting files: `data/generate_synthetic_data.py` (the practice-data generator; it documents which real-world causes the synthetic stalls were built from), `run_all.py` (one-command end-to-end run).

## Why a model, in KnowYi's terms

Most systems remember transactions: purchase orders, revisions, shipments. The reasoning that connects them is what gets lost, and the next build rediscovers it. A trained model is one small form of the alternative — completed builds become reusable judgment, so a future build does not have to start from zero. KnowYi calls the larger version of this manufacturing decision memory. This project is a first, narrow instance: engineering condition → likely outcome, learned from builds that already happened.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

## Swapping in real data

1. Export historical builds to CSV with the same columns as `data/builds_raw.csv` (the generator file documents the exact schema).
2. Save it as `data/builds_raw.csv`, skip step 1 of `run_all.py`, run steps 2–5.
3. Compare the real baseline with the practice baseline. The performance target belongs after that comparison, not before it.

## Internship KPI mapping

| KPI | Where it lives |
|---|---|
| 1. One usable module every two weeks, merged to main | The five modules above |
| 2. One complete project, data → training → deployment, owned independently | This repository, end to end |
| 3. Model hits the target value on the agreed metric, written in the issue | `reports/metrics.json`; target = real-data baseline + agreed margin, set with Ella after the real baseline exists |
