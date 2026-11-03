"""Step 2: train a model on normalized SOURCE TRAIN data only.

Usage:  python src/train.py --config configs/baseline.yaml

Outputs:
  results/<config>/model.joblib
  An MLflow run (view with:  mlflow ui) holding the config, params,
  training metrics, the model and the scaler.
"""

import argparse
import time

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from utils import load_config, run_dirs, save_json, set_seed

MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "mlp": MLPClassifier,
}


def main(config_path):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    feats, label = cfg["data"]["features"], cfg["data"]["label_column"]
    processed, results = run_dirs(cfg)

    train = pd.read_csv(processed / "source_train.csv")
    X, y = train[feats], train[label]

    m = cfg["model"]
    params = dict(m.get("params") or {})
    params.setdefault("random_state", cfg["seed"])
    model = MODELS[m["type"]](**params)

    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run(run_name=f"{cfg['_config_name']}-{m['type']}") as run:
        t0 = time.time()
        model.fit(X, y)
        train_time = time.time() - t0

        pred = model.predict(X)
        metrics = {
            "train_accuracy": accuracy_score(y, pred),
            "train_f1": f1_score(y, pred),
            "train_time_s": train_time,
        }

        mlflow.log_params({"model_type": m["type"], "features": ",".join(feats),
                           "normalization": cfg["normalization"]["method"],
                           "seed": cfg["seed"], **params})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cfg["_config_path"], "config")
        mlflow.log_artifact(str(results / "scaler.joblib"), "preprocessing")
        mlflow.log_artifact(str(results / "normalization_report.json"), "preprocessing")

        joblib.dump(model, results / "model.joblib")
        mlflow.log_artifact(str(results / "model.joblib"), "model")
        save_json({"mlflow_run_id": run.info.run_id, **metrics}, results / "train_metrics.json")

    print(f"Trained {m['type']} on {len(train)} source rows in {train_time:.2f}s")
    print(f"Train accuracy {metrics['train_accuracy']:.4f}, F1 {metrics['train_f1']:.4f}")
    print(f"MLflow run id: {run.info.run_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
