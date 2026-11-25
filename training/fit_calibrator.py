"""Fit a Platt calibrator so the served score means what it says (audit 3.4 / 8.3b).

    python training/fit_calibrator.py --raw training/data/churn_raw.csv --out models/calibrated

What it does
  1. Re-creates the notebook's cleaned dataset and its exact stratified 70/10/20 split (seed 42).
  2. REFUSES to continue unless the reproduction is exact: split sizes 6996/1000/1999 and the raw model's
     validation / test ROC-AUC equal the notebook's (0.8664 / 0.8605). Otherwise "validation" rows could be
     rows the model trained on, and the calibrator would be fit on leaked data.
  3. Fits Platt scaling (logistic regression on the model's log-odds) on the VALIDATION split only.
  4. Reports Brier / ECE / AUC before vs after on the untouched TEST split, plus observed churn per risk tier.
  5. Writes a complete, self-consistent model folder (manifest, hashes, calibrator) to --out.

The decision boundary is preserved: the new threshold is the calibrated value of the old raw threshold, so exactly
the same customers are flagged. Only the meaning of the number changes: it becomes a probability.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(HERE))

from churn_api.ml.bundle import ModelBundle          # noqa: E402
from churn_api.ml.features import build_feature_frame  # noqa: E402
from notebook_cleaning import notebook_clean          # noqa: E402

EXPECTED = {"sizes": (6996, 1000, 1999), "val_auc": 0.8664, "test_auc": 0.8605}
CUTOFFS = [0.15, 0.40]      # audit report section 8.3b


def ece(y, p, bins: int = 10) -> float:
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=HERE / "data" / "churn_raw.csv")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "calibrated")
    ap.add_argument("--version", default="1.1.0")
    args = ap.parse_args()

    bundle = ModelBundle.load(args.model_dir)
    df = notebook_clean(pd.read_csv(args.raw)).reset_index(drop=True)
    y = df["Exited"].to_numpy()
    idx = np.arange(len(df))
    tmp, test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    train, val = train_test_split(tmp, test_size=0.125, random_state=42, stratify=y[tmp])

    raw = bundle.predict_raw(build_feature_frame(df, bundle))
    got = {"sizes": (len(train), len(val), len(test)), "val_auc": roc_auc_score(y[val], raw[val]),
           "test_auc": roc_auc_score(y[test], raw[test])}
    print(f"split reproduction: sizes={got['sizes']} val AUC={got['val_auc']:.4f} test AUC={got['test_auc']:.4f}")
    if (got["sizes"] != EXPECTED["sizes"] or abs(got["val_auc"] - EXPECTED["val_auc"]) > 5e-4
            or abs(got["test_auc"] - EXPECTED["test_auc"]) > 5e-4):
        sys.exit(f"REFUSING: could not reproduce the notebook split/metrics {EXPECTED}. A calibrator fit on rows the "
                 "model trained on would be overconfident. Check the raw CSV and notebook_cleaning.py.")

    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(raw[val]).reshape(-1, 1), y[val])
    cal = platt.predict_proba(logit(raw).reshape(-1, 1))[:, 1]

    report = {"split": dict(zip(("train", "validation", "test"), got["sizes"])), "method": "Platt scaling on validation split",
              "test": {"roc_auc_raw": float(roc_auc_score(y[test], raw[test])), "roc_auc_calibrated": float(roc_auc_score(y[test], cal[test])),
                       "brier_raw": float(brier_score_loss(y[test], raw[test])), "brier_calibrated": float(brier_score_loss(y[test], cal[test])),
                       "ece_raw": ece(y[test], raw[test]), "ece_calibrated": ece(y[test], cal[test]),
                       "mean_predicted_raw": float(raw[test].mean()), "mean_predicted_calibrated": float(cal[test].mean()),
                       "observed_churn": float(y[test].mean())}}
    tier = np.digitize(cal[test], CUTOFFS)
    report["test_tiers"] = [{"tier": n, "customers": int((tier == i).sum()), "observed_churn": float(y[test][tier == i].mean())}
                            for i, n in enumerate(["Low", "Medium", "High"])]
    thr = float(platt.predict_proba(logit(np.array([bundle.threshold])).reshape(-1, 1))[0, 1])
    grid = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y[val], cal[val] >= t) for t in grid]
    report["threshold"] = {"raw": bundle.threshold, "calibrated_equivalent": thr,
                           "validation_f1_optimal_calibrated": float(grid[int(np.argmax(f1s))]), "validation_f1_at_chosen": float(f1_score(y[val], cal[val] >= thr))}
    print(json.dumps(report, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(platt, args.out / "calibrator.pkl")
    card = json.loads((HERE / "model_card.json").read_text())
    card["model_version"] = args.version
    card["description"] += " Platt-calibrated: the score is a probability."
    card["metrics"]["calibration_test"] = report["test"]
    card["risk_tiers"] = {"labels": ["Low", "Medium", "High"], "cutoffs": CUTOFFS, "basis": "calibrated",
                          "note": "Cut-offs on the CALIBRATED probability (audit 8.3b). Observed churn on the held-out test split: "
                                  + ", ".join(f"{t['tier']} {t['observed_churn']:.1%}" for t in report["test_tiers"]) + "."}
    card["known_limitations"] = [l for l in card["known_limitations"] if "not a calibrated" not in l] + [
        "The calibrator was fit on 1,000 validation rows; expect roughly +/-3 points of noise in each probability band.",
        "The flag threshold keeps the original decision boundary (the calibrated value of the old raw 0.5) and is not re-tuned."]
    card_path = args.out / "model_card.calibrated.json"
    card_path.write_text(json.dumps(card, indent=2) + "\n")
    subprocess.run([sys.executable, str(HERE / "export_artifacts.py"), "--source", str(args.model_dir / "source" / "churn_model.pkl"),
                    "--preprocessing", str(args.model_dir / "preprocessing.pkl"), "--out", str(args.out),
                    "--model-card", str(card_path), "--calibrator", str(args.out / "calibrator.pkl"), "--threshold", str(thr)], check=True)
    card_path.unlink()
    (args.out / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote calibrated model folder: {args.out}\nUse it with:  MODEL_DIR={args.out}")


if __name__ == "__main__":
    main()
