"""Train and evaluate the anomaly detector on the real NSL-KDD dataset.

Setup:
  1. Download KDDTrain+.txt and KDDTest+.txt (NSL-KDD, University of New Brunswick
     CIC, or the Kaggle mirror "NSL-KDD").
  2. Put both files in a folder, e.g. ./nsl-kdd/
  3. python train_nslkdd.py --dir nsl-kdd

Protocol: train on KDDTrain+, evaluate on KDDTest+ (the official split; the test
set contains attack types that never appear in training, which makes it a fair
test of generalisation).
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in COLUMNS if c not in CATEGORICAL + ["label", "difficulty"]]

CATEGORY = {"normal": "normal"}
for a in ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm",
          "processtable", "worm", "mailbomb"]:
    CATEGORY[a] = "dos"
for a in ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]:
    CATEGORY[a] = "probe"
for a in ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient",
          "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"]:
    CATEGORY[a] = "r2l"
for a in ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"]:
    CATEGORY[a] = "u2r"


def load(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["label"] = df["label"].str.strip().str.lower()
    df["category"] = df["label"].map(CATEGORY).fillna("unknown_attack")
    df["is_anomaly"] = (df["label"] != "normal").astype(int)
    return df


def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scale", RobustScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def metrics(y, pred, score):
    return {"precision": round(precision_score(y, pred), 4), "recall": round(recall_score(y, pred), 4),
            "f1": round(f1_score(y, pred), 4), "roc_auc": round(roc_auc_score(y, score), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="nsl-kdd")
    ap.add_argument("--out", default="artifacts_nslkdd")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tr = load(os.path.join(args.dir, "KDDTrain+.txt"))
    te = load(os.path.join(args.dir, "KDDTest+.txt"))
    print(f"Train: {len(tr)} flows ({tr['is_anomaly'].mean():.1%} attacks) | "
          f"Test: {len(te)} flows ({te['is_anomaly'].mean():.1%} attacks)")
    print("Test categories:", te["category"].value_counts().to_dict(), "\n")
    feats = NUMERIC + CATEGORICAL
    X_tr, y_tr, X_te, y_te = tr[feats], tr["is_anomaly"], te[feats], te["is_anomaly"]

    rf = Pipeline([("prep", preprocessor()),
                   ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                  n_jobs=-1, random_state=7))]).fit(X_tr, y_tr)
    rf_prob = rf.predict_proba(X_te)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)
    rf_m = metrics(y_te, rf_pred, rf_prob)

    iso = Pipeline([("prep", preprocessor()),
                    ("clf", IsolationForest(n_estimators=300, contamination=0.05,
                                            random_state=7, n_jobs=-1))]).fit(X_tr[y_tr == 0])
    iso_score = -iso.decision_function(X_te)
    thr = float(np.quantile(-iso.decision_function(X_tr[y_tr == 0]), 0.95))
    iso_pred = (iso_score >= thr).astype(int)
    iso_m = metrics(y_te, iso_pred, iso_score)

    print("RandomForest    :", rf_m)
    print("IsolationForest :", iso_m, "\n")
    print(classification_report(y_te, rf_pred, target_names=["normal", "attack"], digits=3))

    per_cat = {}
    for c in ["dos", "probe", "r2l", "u2r"]:
        m = (te["category"] == c).values
        if m.sum():
            per_cat[c] = {"n": int(m.sum()), "rf_recall": round(float(rf_pred[m].mean()), 4),
                          "iso_recall": round(float(iso_pred[m].mean()), 4)}
    print("Recall by attack category:", json.dumps(per_cat, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ConfusionMatrixDisplay(confusion_matrix(y_te, rf_pred), display_labels=["normal", "attack"]
                           ).plot(ax=ax[0], colorbar=False)
    ax[0].set_title("Random Forest - confusion matrix (KDDTest+)")
    for name, s in [("Random Forest", rf_prob), ("Isolation Forest", iso_score)]:
        fpr, tpr, _ = roc_curve(y_te, s)
        ax[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, s):.3f})")
    ax[1].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[1].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].legend(loc="lower right")
    cats = list(per_cat); xs = np.arange(len(cats)); w = .38
    ax[2].bar(xs - w/2, [per_cat[c]["rf_recall"] for c in cats], w, label="Random Forest")
    ax[2].bar(xs + w/2, [per_cat[c]["iso_recall"] for c in cats], w, label="Isolation Forest")
    ax[2].set_xticks(xs, [c.upper() for c in cats]); ax[2].set_ylim(0, 1.05)
    ax[2].set(title="Recall by attack category", ylabel="Recall"); ax[2].legend()
    plt.tight_layout(); plt.savefig(f"{args.out}/evaluation.png", dpi=130); plt.close()

    names = rf.named_steps["prep"].get_feature_names_out()
    pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values().tail(12) \
        .plot.barh(figsize=(7, 4.5), title="Top features (Random Forest)")
    plt.tight_layout(); plt.savefig(f"{args.out}/feature_importance.png", dpi=130); plt.close()

    joblib.dump({"rf": rf, "iso": iso, "iso_threshold": thr, "features": feats},
                f"{args.out}/model.joblib")
    json.dump({"dataset": "NSL-KDD (train KDDTrain+, test KDDTest+)", "random_forest": rf_m,
               "isolation_forest": iso_m, "per_category": per_cat},
              open(f"{args.out}/metrics.json", "w"), indent=2)
    print(f"\nSaved model, metrics and plots to ./{args.out}/")


if __name__ == "__main__":
    main()
