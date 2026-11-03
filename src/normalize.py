"""Step 1: load raw data from both domains, fit the scaler on SOURCE TRAIN
ONLY, apply it everywhere, and save everything needed to check the step.

Usage:  python src/normalize.py --config configs/baseline.yaml

Outputs:
  data/processed/<config>/source_train.csv, source_test.csv, target.csv
  results/<config>/scaler.joblib          the fitted scaler (reused as-is)
  results/<config>/normalization_report.json
      - rows kept / dropped per file, class balance
      - scaler parameters
      - per-feature mean/std of source vs target BEFORE scaling
        (this is your first, simple measure of domain shift)
"""

import argparse

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import load_config, load_domain, run_dirs, save_json, set_seed


def main(config_path):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    d = cfg["data"]
    feats, label = d["features"], d["label_column"]
    processed, results = run_dirs(cfg)

    src_tr, rep_tr = load_domain(d["source_train"], d.get("source_rename"), feats, label, d["missing"])
    src_te, rep_te = load_domain(d["source_test"], d.get("source_rename"), feats, label, d["missing"])
    tgt, rep_tg = load_domain(d["target"], d.get("target_rename"), feats, label, d["missing"])

    if cfg["normalization"]["fit_on"] != "source_train":
        raise ValueError("Scaler must be fit on source_train for zero-shot evaluation.")

    method = cfg["normalization"]["method"]
    scaler = {"standard": StandardScaler, "minmax": MinMaxScaler}[method]()
    scaler.fit(src_tr[feats])  # <-- the only place the scaler sees data
    joblib.dump(scaler, results / "scaler.joblib")

    def scale(df):
        out = df.copy()
        out[feats] = scaler.transform(df[feats])
        return out

    scale(src_tr).to_csv(processed / "source_train.csv", index=False)
    scale(src_te).to_csv(processed / "source_test.csv", index=False)
    scale(tgt).to_csv(processed / "target.csv", index=False)

    shift = pd.DataFrame({
        "source_mean": src_tr[feats].mean(),
        "source_std": src_tr[feats].std(),
        "target_mean": tgt[feats].mean(),
        "target_std": tgt[feats].std(),
    })
    shift["mean_diff_in_source_stds"] = (shift.target_mean - shift.source_mean) / shift.source_std

    if method == "standard":
        params = {"mean": dict(zip(feats, scaler.mean_.tolist())),
                  "scale": dict(zip(feats, scaler.scale_.tolist()))}
    else:
        params = {"min": dict(zip(feats, scaler.data_min_.tolist())),
                  "max": dict(zip(feats, scaler.data_max_.tolist()))}

    save_json({
        "config": cfg["_config_path"],
        "method": method,
        "fit_on": "source_train",
        "files": {"source_train": rep_tr, "source_test": rep_te, "target": rep_tg},
        "scaler_params": params,
        "raw_feature_shift": shift.round(4).to_dict(orient="index"),
    }, results / "normalization_report.json")

    print(f"Normalized with {method} scaler fit on source_train ({len(src_tr)} rows).")
    print("Raw feature shift (target mean vs source mean, in source std units):")
    print(shift["mean_diff_in_source_stds"].round(2).to_string())
    print(f"Saved -> {processed}/ and {results}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
