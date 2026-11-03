"""Combine raw HPDmobile files into one 1-minute CSV for the target domain.

Reads, per home:
  H{n}_ENVIRONMENTAL/*_ENV/*_env.csv   10-second sensor readings, one file per hub per day
  H{n}_GROUNDTRUTH/*_groundtruth.csv   10-second occupancy labels, one file per day

Each reading is matched to its home's ground truth at the same timestamp
(readings with no label are dropped), then averaged to 1-minute rows per hub,
to match UCI's 1-minute resolution.

Minute label: occupied = 1 if at least half of the matched 10-s labels are 1
(majority; a 3-of-6 tie counts as occupied). occ_fraction keeps the raw share.

Minutes where any feature is missing for every reading in that minute are dropped.

Usage:
  python scripts/prepare_hpdmobile.py                       # all homes
  python scripts/prepare_hpdmobile.py --homes H1 --out data/interim/hpdmobile_H1.csv

Outputs:
  <out>                   the combined CSV
  <out stem>_report.json  rows kept / dropped per home and in total
"""

import argparse
import json
from pathlib import Path

import pandas as pd

FEATURES = ["temp_c", "rh_percent", "light_lux", "co2eq_ppm"]
OUT_COLUMNS = ["timestamp", "home", "hub", *FEATURES,
               "occupied", "occ_fraction", "number", "n_readings"]


def read_env(path):
    df = pd.read_csv(path, usecols=lambda c: c in {"Unnamed: 0", *FEATURES, "hub", "home"})
    df = df.rename(columns={"Unnamed: 0": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def read_gt(path):
    df = pd.read_csv(path, usecols=["timestamp", "occupied", "number"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def process_home(raw, home):
    env_files = sorted((raw / f"{home}_ENVIRONMENTAL").glob("*_ENV/*_env.csv"))
    gt_files = sorted((raw / f"{home}_GROUNDTRUTH").glob("*_groundtruth.csv"))
    if not env_files:
        raise FileNotFoundError(f"No environmental files for {home} under {raw}")
    if not gt_files:
        raise FileNotFoundError(f"No ground-truth files for {home} under {raw}")

    env = pd.concat([read_env(f) for f in env_files], ignore_index=True)
    gt = pd.concat([read_gt(f) for f in gt_files], ignore_index=True)

    env_dupes = int(env.duplicated(["hub", "timestamp"]).sum())
    env = env.drop_duplicates(["hub", "timestamp"])
    gt_dupes = int(gt.duplicated("timestamp").sum())
    gt = gt.drop_duplicates("timestamp")

    merged = env.merge(gt, on="timestamp", how="left", indicator=True)
    matched = merged["_merge"] == "both"
    no_gt = int((~matched).sum())
    merged = merged[matched].drop(columns="_merge")
    na_label = int(merged["occupied"].isna().sum())
    merged = merged.dropna(subset=["occupied"])

    merged["minute"] = merged["timestamp"].dt.floor("min")
    agg = merged.groupby(["home", "hub", "minute"], as_index=False).agg(
        **{f: (f, "mean") for f in FEATURES},
        occ_fraction=("occupied", "mean"),
        number=("number", "mean"),
        n_readings=("occupied", "size"),
    )
    agg = agg.rename(columns={"minute": "timestamp"})
    agg["occupied"] = (agg["occ_fraction"] >= 0.5).astype(int)
    agg["number"] = agg["number"].round().astype("Int64")

    minutes_total = len(agg)
    missing_mask = agg[FEATURES].isna().any(axis=1)
    missing_by_feature = {f: int(agg[f].isna().sum()) for f in FEATURES}
    agg = agg[~missing_mask]

    report = {
        "env_files": len(env_files),
        "gt_files": len(gt_files),
        "hubs": sorted(env["hub"].unique().tolist()),
        "readings_raw": len(env) + env_dupes,
        "readings_duplicate_dropped": env_dupes,
        "gt_duplicate_timestamps_dropped": gt_dupes,
        "readings_without_gt_dropped": no_gt,
        "readings_with_missing_label_dropped": na_label,
        "readings_matched": len(merged),
        "minutes_produced": minutes_total,
        "minutes_missing_feature_dropped": int(missing_mask.sum()),
        "minutes_missing_by_feature": missing_by_feature,
        "rows_written": len(agg),
        "rows_per_hub": {k: int(v) for k, v in agg["hub"].value_counts().sort_index().items()},
        "class_balance": {str(k): int(v) for k, v in agg["occupied"].value_counts().sort_index().items()},
        "mixed_label_minutes": int(((agg["occ_fraction"] > 0) & (agg["occ_fraction"] < 1)).sum()),
        "date_range": [str(agg["timestamp"].min()), str(agg["timestamp"].max())],
    }
    return agg[OUT_COLUMNS], report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw/hpdmobile")
    p.add_argument("--homes", nargs="+", help="e.g. H1 H2 (default: all homes found)")
    p.add_argument("--out", default="data/interim/hpdmobile_combined.csv")
    args = p.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    homes = args.homes or sorted(d.name.split("_")[0] for d in raw.glob("H*_ENVIRONMENTAL") if d.is_dir())
    if not homes:
        raise SystemExit(f"No H*_ENVIRONMENTAL folders found under {raw}")

    frames, per_home = [], {}
    for home in homes:
        print(f"Processing {home} ...", flush=True)
        df, rep = process_home(raw, home)
        frames.append(df)
        per_home[home] = rep
        print(f"  {rep['readings_matched']:,} matched readings -> {rep['rows_written']:,} minute rows "
              f"(dropped: {rep['readings_without_gt_dropped']:,} readings without GT, "
              f"{rep['minutes_missing_feature_dropped']:,} minutes with missing features)")

    combined = pd.concat(frames, ignore_index=True).sort_values(["home", "hub", "timestamp"])
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)

    sum_keys = ["env_files", "gt_files", "readings_raw", "readings_duplicate_dropped",
                "gt_duplicate_timestamps_dropped", "readings_without_gt_dropped",
                "readings_with_missing_label_dropped", "readings_matched", "minutes_produced",
                "minutes_missing_feature_dropped", "rows_written", "mixed_label_minutes"]
    total = {k: sum(r[k] for r in per_home.values()) for k in sum_keys}
    total["class_balance"] = {str(k): int(v) for k, v in combined["occupied"].value_counts().sort_index().items()}

    report = {"homes": homes, "output": str(out), "minute_label": "majority (occ_fraction >= 0.5)",
              "total": total, "per_home": per_home}
    report_path = out.with_name(out.stem + "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Summary ===")
    for k in sum_keys:
        print(f"{k:38}{total[k]:>14,}")
    print(f"{'class_balance':38}{total['class_balance']}")
    print(f"Saved -> {out}\n         {report_path}")


if __name__ == "__main__":
    main()
