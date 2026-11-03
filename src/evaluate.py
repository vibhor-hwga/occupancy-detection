"""Step 3: evaluate the trained model in-domain (source test) and
zero-shot on the target domain, and report the accuracy drop.

Usage:  python src/evaluate.py --config configs/baseline.yaml

Outputs:
  results/<config>/eval_metrics.json
  results/<config>/confusion_matrices.png
  Metrics and plot are added to the same MLflow run that train.py created.
"""

import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             balanced_accuracy_score, f1_score,
                             precision_score, recall_score)

from utils import load_config, run_dirs, save_json


def score(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "n": int(len(y)),
    }


def main(config_path):
    cfg = load_config(config_path)
    feats, label = cfg["data"]["features"], cfg["data"]["label_column"]
    processed, results = run_dirs(cfg)

    model = joblib.load(results / "model.joblib")
    with open(results / "train_metrics.json") as f:
        run_id = json.load(f)["mlflow_run_id"]

    out, preds = {}, {}
    for split in ["source_test", "target"]:
        df = pd.read_csv(processed / f"{split}.csv")
        preds[split] = (df[label], model.predict(df[feats]))
        out[split] = score(*preds[split])

    out["drop"] = {k: out["source_test"][k] - out["target"][k]
                   for k in ["accuracy", "balanced_accuracy", "f1"]}
    save_json(out, results / "eval_metrics.json")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, split in zip(axes, preds):
        ConfusionMatrixDisplay.from_predictions(*preds[split], ax=ax, colorbar=False)
        ax.set_title(f"{split} (acc {out[split]['accuracy']:.3f})")
    fig.tight_layout()
    fig.savefig(results / "confusion_matrices.png", dpi=120)

    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run(run_id=run_id):
        for split in ["source_test", "target"]:
            mlflow.log_metrics({f"{split}_{k}": v for k, v in out[split].items() if k != "n"})
        mlflow.log_metrics({f"drop_{k}": v for k, v in out["drop"].items()})
        mlflow.log_artifact(str(results / "eval_metrics.json"), "evaluation")
        mlflow.log_artifact(str(results / "confusion_matrices.png"), "evaluation")

    print(f"{'':12}{'accuracy':>10}{'bal_acc':>10}{'f1':>10}")
    for split in ["source_test", "target"]:
        s = out[split]
        print(f"{split:12}{s['accuracy']:>10.4f}{s['balanced_accuracy']:>10.4f}{s['f1']:>10.4f}")
    d = out["drop"]
    print(f"{'drop':12}{d['accuracy']:>10.4f}{d['balanced_accuracy']:>10.4f}{d['f1']:>10.4f}")
    print(f"Saved -> {results}/  (MLflow run {run_id})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
