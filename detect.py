"""Score network flows with the trained detector.

Usage:
  python detect.py flows.csv              # score a CSV of flows
  python detect.py --demo                 # score a fresh batch of synthetic traffic
Outputs a table of flagged flows and writes <input>_scored.csv.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from data import FEATURES, generate_flows


def score(df, bundle):
    rf, iso, thr = bundle["rf"], bundle["iso"], bundle["iso_threshold"]
    out = df.copy()
    out["rf_prob"] = rf.predict_proba(df[FEATURES])[:, 1].round(4)
    out["iso_score"] = (-iso.decision_function(df[FEATURES])).round(4)
    out["alert"] = np.where(out["rf_prob"] >= 0.5, "ANOMALY (known pattern)",
                    np.where(out["iso_score"] >= thr, "ANOMALY (novel/outlier)", "normal"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--model", default="artifacts/model.joblib")
    args = ap.parse_args()
    if not args.csv and not args.demo:
        ap.error("provide a CSV path or --demo")

    bundle = joblib.load(args.model)
    df = generate_flows(n_normal=2000, seed=2026) if args.demo else pd.read_csv(args.csv)
    res = score(df, bundle)
    flagged = res[res["alert"] != "normal"]
    print(f"{len(flagged)} / {len(res)} flows flagged ({len(flagged)/len(res):.1%})")
    cols = ["protocol", "service", "duration", "src_pkts", "distinct_dst_ports",
            "rf_prob", "iso_score", "alert"] + (["label"] if "label" in res else [])
    print(flagged.sort_values("rf_prob", ascending=False)[cols].head(15).to_string(index=False))
    path = (args.csv or "demo").rsplit(".", 1)[0] + "_scored.csv"
    res.to_csv(path, index=False)
    print("full results ->", path)


if __name__ == "__main__":
    main()
