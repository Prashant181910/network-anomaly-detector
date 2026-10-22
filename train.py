"""Train and evaluate the network traffic anomaly detector.

Two complementary models are trained on the same engineered flow features:

  1. RandomForest (supervised)   - classifies flows as normal / anomalous and
                                   can recognise known attack families.
  2. IsolationForest (unsupervised) - fit on NORMAL traffic only; flags flows
                                   that deviate from baseline, so it can catch
                                   attack types never seen in training.

Usage:  python train.py [--csv traffic.csv] [--out artifacts]
"""
import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from data import CATEGORICAL, FEATURES, NUMERIC, generate_flows


def build_preprocessor():
    # log1p tames the heavy-tailed byte/packet counts before scaling
    return ColumnTransformer([
        ("num", Pipeline([
            ("log", __import__("sklearn.preprocessing", fromlist=["FunctionTransformer"])
             .FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", RobustScaler()),
        ]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def metrics(y_true, y_pred, y_score):
    return {
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="flow CSV; omit to generate synthetic traffic")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv) if args.csv else generate_flows()
    print(f"Loaded {len(df)} flows | anomaly rate {df['is_anomaly'].mean():.1%}")
    print(df["label"].value_counts().to_string(), "\n")

    X, y = df[FEATURES], df["is_anomaly"]
    X_tr, X_te, y_tr, y_te, lab_tr, lab_te = train_test_split(
        X, y, df["label"], test_size=0.25, stratify=df["label"], random_state=7)

    # ---- Supervised: Random Forest ------------------------------------
    rf = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       n_jobs=-1, random_state=7)),
    ]).fit(X_tr, y_tr)
    rf_prob = rf.predict_proba(X_te)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)
    rf_m = metrics(y_te, rf_pred, rf_prob)

    # ---- Unsupervised: Isolation Forest (trained on normal only) ------
    iso = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", IsolationForest(n_estimators=300, contamination=0.02, random_state=7, n_jobs=-1)),
    ]).fit(X_tr[y_tr == 0])
    iso_score = -iso.decision_function(X_te)           # higher = more anomalous
    normal_scores = -iso.decision_function(X_tr[y_tr == 0])
    thr = float(np.quantile(normal_scores, 0.98))      # 2% FPR on baseline
    iso_pred = (iso_score >= thr).astype(int)
    iso_m = metrics(y_te, iso_pred, iso_score)

    print("RandomForest      :", rf_m)
    print("IsolationForest   :", iso_m, "\n")
    print("RandomForest report\n", classification_report(
        y_te, rf_pred, target_names=["normal", "anomalous"], digits=3))

    # recall per attack family (shows what each model catches)
    per_family = {}
    for fam in sorted(lab_te.unique()):
        if fam == "normal":
            continue
        mask = (lab_te == fam).values
        per_family[fam] = {"rf_recall": round(float(rf_pred[mask].mean()), 4),
                           "iso_recall": round(float(iso_pred[mask].mean()), 4),
                           "n": int(mask.sum())}
    print("Per-attack recall:\n", json.dumps(per_family, indent=2))

    # ---- Plots --------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ConfusionMatrixDisplay(confusion_matrix(y_te, rf_pred),
                           display_labels=["normal", "anomalous"]).plot(ax=ax[0], colorbar=False)
    ax[0].set_title("Random Forest - confusion matrix")
    for name, s in [("Random Forest", rf_prob), ("Isolation Forest", iso_score)]:
        fpr, tpr, _ = roc_curve(y_te, s)
        ax[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, s):.3f})")
    ax[1].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[1].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].legend(loc="lower right")
    fams = list(per_family)
    xs = np.arange(len(fams)); w = 0.38
    ax[2].bar(xs - w/2, [per_family[f]["rf_recall"] for f in fams], w, label="Random Forest")
    ax[2].bar(xs + w/2, [per_family[f]["iso_recall"] for f in fams], w, label="Isolation Forest")
    ax[2].set_xticks(xs, fams); ax[2].set_ylim(0, 1.05)
    ax[2].set(title="Recall by attack type", ylabel="Recall"); ax[2].legend()
    plt.tight_layout(); plt.savefig(f"{args.out}/evaluation.png", dpi=130); plt.close()

    # Feature importances
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values()
    imp.tail(12).plot.barh(figsize=(7, 4.5), title="Top features (Random Forest)")
    plt.tight_layout(); plt.savefig(f"{args.out}/feature_importance.png", dpi=130); plt.close()

    # ---- Persist ------------------------------------------------------
    joblib.dump({"rf": rf, "iso": iso, "iso_threshold": thr}, f"{args.out}/model.joblib")
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump({"random_forest": rf_m, "isolation_forest": iso_m,
                   "per_family": per_family}, f, indent=2)
    print(f"\nSaved model, metrics and plots to ./{args.out}/")


if __name__ == "__main__":
    main()
