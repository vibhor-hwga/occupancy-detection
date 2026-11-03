# Cross-domain occupancy detection

Train on UCI Occupancy Detection (source), test zero-shot on HPDmobile (target),
and measure the accuracy drop under domain shift.

## Layout

```
configs/      one YAML file per experiment (the only thing you change between runs)
data/raw/     original files, never edited       uci/   hpdmobile/
data/processed/<config>/    normalized CSVs (generated)
src/          normalize.py -> train.py -> evaluate.py
results/<config>/           scaler, model, metrics, plots (generated)
scripts/      helpers (dummy data for testing)
RESULTS.md    your log of runs worth keeping
```

## Setup (once)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
git init
git add .
git commit -m "Project skeleton"
```

If `Activate.ps1` is blocked, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Check the pipeline works (fake data)

```powershell
python scripts/make_dummy_data.py
python run_pipeline.py configs/smoke_test.yaml
```

## Real data

1. Put the UCI files (`datatraining.txt`, `datatest.txt`) in `data/raw/uci/`.
2. HPDmobile is split across homes, sensor hubs and separate ground-truth
   files. Unzip them into `data/raw/hpdmobile/` and combine them with
   ```powershell
   python scripts/prepare_hpdmobile.py
   ```
   This writes `data/interim/hpdmobile_combined.csv` (1-minute rows per hub,
   labelled by majority vote over the 10-s ground truth) and
   `hpdmobile_combined_report.json` (rows kept and dropped).
3. Open both files, check the column names, and fix `target_rename` in
   `configs/baseline.yaml`. The script stops with a clear error if any column
   is missing.
4. Run:
   ```powershell
   python run_pipeline.py configs/baseline.yaml
   ```

## Seeing what happened

- **Terminal output** of each step: rows kept, raw feature shift, accuracies.
- **`results/<config>/`**: `normalization_report.json` (rows dropped, scaler
  parameters, source-vs-target feature statistics), `train_metrics.json`,
  `eval_metrics.json`, `confusion_matrices.png`, plus the saved scaler and model.
- **MLflow dashboard**: run `mlflow ui` and open http://127.0.0.1:5000 to
  compare all runs side by side, each with its config file attached.
- **Git history**: every code or config change, with the diff.

## Rules that keep results valid

- The scaler is fit on **source train only** (`normalize.py` refuses otherwise),
  then the same saved scaler is applied to the target. Nothing about the target
  is used during training.
- Raw data is never edited. All cleaning happens in code.
- To change an experiment, make a new config file rather than editing one whose
  results you have already recorded.
- Commit before each run you plan to report, so the run matches a commit.
