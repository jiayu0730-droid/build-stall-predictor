"""
Modules 3 + 4 - Baseline model, improved model, evaluation
(KPI 1, deliveries #3 and #4; this is also where KPI 3 gets measured)

Two models, on purpose:
  1. Baseline = logistic regression. The "dumb but honest" first version.
     Its score is the reference point: KPI 3's target value should be set
     as baseline + an agreed margin, AFTER seeing this number, never
     before seeing the data.
  2. Improved = random forest. Captures interactions (e.g. wide quote
     spread only matters when the package is incomplete).

Evaluation: 5-fold cross-validated AUC and accuracy on all data, plus a
held-out test split (20%) that neither model trains on. Results are
written to reports/metrics.json and the improved model is saved to
models/stall_model.joblib for the prediction script.
"""

import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "features.csv")
    X = data.drop(columns=["stalled"])
    y = data["stalled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    improved = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                      random_state=42, n_jobs=-1)

    results = {"n_rows": int(len(data)), "stall_rate": round(float(y.mean()), 4)}

    for name, model in [("baseline_logistic", baseline), ("improved_random_forest", improved)]:
        cv_auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_auc_std": round(float(cv_auc.std()), 4),
            "test_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
        }
        print(f"{name}: CV AUC {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f}) | "
              f"test AUC {results[name]['test_auc']:.3f} | "
              f"test accuracy {results[name]['test_accuracy']:.3f}")

    # Feature importance from the improved model, in plain ranked order
    importances = (pd.Series(improved.feature_importances_, index=X.columns)
                   .sort_values(ascending=False))
    results["top_features"] = {k: round(float(v), 4) for k, v in importances.head(8).items()}
    print("\nTop features:")
    for k, v in importances.head(8).items():
        print(f"  {k:32s} {v:.3f}")

    (ROOT / "reports").mkdir(exist_ok=True)
    with open(ROOT / "reports" / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": improved, "columns": list(X.columns)},
                ROOT / "models" / "stall_model.joblib")
    print(f"\nSaved reports/metrics.json and models/stall_model.joblib")


if __name__ == "__main__":
    main()
