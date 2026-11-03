"""Shared helpers: config loading, seeding, and loading a domain's data
into a common column format."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_config_name"] = Path(path).stem
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_dirs(cfg):
    """Every experiment gets its own processed-data and results folder,
    named after its config file, so runs never overwrite each other."""
    name = cfg["_config_name"]
    processed = Path("data/processed") / name
    results = Path("results") / name
    processed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return processed, results


def load_domain(path, rename, features, label, missing="drop"):
    """Read a raw file, rename columns to the shared names, keep only the
    features + label, and handle missing values. Returns (df, report)."""
    df = pd.read_csv(path)
    if rename:
        df = df.rename(columns=rename)

    needed = list(features) + [label]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise KeyError(
            f"{path}: columns {absent} not found after renaming. "
            f"Columns present: {list(df.columns)}. "
            f"Fix the *_rename mapping in the config."
        )

    df = df[needed].copy()
    rows_in = len(df)
    n_missing = int(df.isna().any(axis=1).sum())
    if missing == "drop":
        df = df.dropna()
    elif missing == "ffill":
        df = df.ffill().dropna()
    else:
        raise ValueError(f"unknown missing-value strategy: {missing}")

    df[label] = df[label].astype(int)

    report = {
        "file": str(path),
        "rows_in": rows_in,
        "rows_with_missing": n_missing,
        "rows_out": len(df),
        "class_balance": {str(k): int(v) for k, v in df[label].value_counts().sort_index().items()},
    }
    return df, report


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
