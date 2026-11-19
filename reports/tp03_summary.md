# TP03 Summary — Track, Compare and Recover ML Experiments with MLflow

## 1. Environment & Tracking Setup
* **MLflow Version**: `3.18.0`
* **Tracking Server URI**: `http://127.0.0.1:5000`
* **Experiment Name**: `telco-churn-tp03`
* **Backend Store**: SQLite (`mlflow.db` — stores structured metadata: experiments, runs, parameters, metrics, tags, dataset metadata, and registered model versions/aliases)
* **Artifact Store**: Local filesystem (`mlartifacts/` — stores binary/large files: fitted scikit-learn pipelines, input examples, confusion matrix PNGs, and `provenance.json`)
* **Dataset Source**: `data/raw/telco.csv` (unchanged raw snapshot verified via SHA-256 tag)

---

## 2. Candidate Comparison & Model Selection
All candidates were evaluated on the exact same 60/20/20 stratified split (`random_state=42`) and 6 features (`tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`) with decision threshold `0.5`. The held-out test set remained untouched during selection.

| Run Name | Model Type | Validation AP (Primary) | Validation Precision | Validation Recall | Validation F1 | Fit Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline-most-frequent` | `DummyClassifier` | 0.2654 | 0.0000 | 0.0000 | 0.0000 | ~0.003 |
| `logistic-C1` | `LogisticRegression` | ~0.6480 | ~0.664 | ~0.529 | ~0.589 | ~0.022 |
| `decision-tree-depth5` | `DecisionTreeClassifier` | ~0.6030 | ~0.63 | ~0.602 | ~0.616 | ~0.019 |
| **`random-forest-n100-depth8`** | **`RandomForestClassifier`** | **0.6696** | **~0.648** | **~0.511** | **0.5710** | **~0.233** |

* **Selection Decision**: **`random-forest-n100-depth8`** was selected because it achieved the highest pre-agreed primary metric (**`val_average_precision = 0.6696`**) as well as the highest validation F1 (`0.5710`), outperforming both the majority-class baseline and linear/single-tree models without looking at the test set.

---

## 3. Selected Model Handover & Registry Details
* **Selected Run Name**: `random-forest-n100-depth8`
* **Selected Run ID**: `999e968962a841a58f6cc0b8d909cf79`
* **Logged Model URI**: `models:/m-ad65c6dbabbb4a1a1c4f605e28de751`
* **Registered Model Name**: `telco-churn`
* **Registered Version**: `Version 1`
* **Assigned Alias**: `candidate` (`models:/telco-churn@candidate`)

---

## 4. Fresh-Process Inference & Persistence Verification
* **Fresh-Process Loading (`src/telco_churn/predict.py`)**: Loaded `models:/telco-churn@candidate` in an independent Python process without calling `fit()`.
* **Prediction Agreement Check**: Verified on 5 validation customer rows; predicted classes matched 100% (`Class agreement: True`), and maximum predicted probability difference was `0.00e+00` (well within `1e-5` tolerance).
* **Server Restart Persistence**: Stopped the MLflow server (`Ctrl+C`) and restarted it with the exact same `--backend-store-uri` and `--artifacts-destination`. Confirmed via UI and `predict.py` that all runs, artifacts, registered model versions, and the `@candidate` alias persisted intact.

---

## 5. Required Screenshots
1. **Multi-Run Comparison View** (`Compare` view showing `random-forest-n100-depth8` leading on `val_average_precision`): *(Attach Screenshot 1)*
2. **Selected Run Details & Dataset Inputs** (`random-forest-n100-depth8` overview showing parameters, metrics, tags, and `telco_train` / `telco_valid` datasets): *(Attach Screenshot 2)*
3. **Model Registry & Alias** (`telco-churn` `Version 1` with `@candidate` alias): *(Attach Screenshot 3)*

---

## 6. Reflection Questions & Answers

1. **What does the tracking database store versus the artifact root?**
   * The **tracking database (`mlflow.db`)** stores lightweight, structured metadata: experiment names, run IDs, timestamps, parameters, numeric metrics, tags (`git_commit`, `data_sha256`), dataset schemas/digests, and Model Registry mappings (model names, versions, aliases).
   * The **artifact root (`mlartifacts/`)** stores large or binary files: serialized scikit-learn pipelines (`cloudpickle`), `MLmodel` metadata, input examples, confusion matrix images (`cm_*.png`), and `provenance.json`.
2. **Why is an alias (`@candidate`) better than hardcoding a run ID or version number in downstream code?**
   * An alias decouples downstream inference code from specific training runs. When a better model is validated later, the team can simply reassign the `candidate` alias to `Version 2` in the registry without modifying or redeploying consumer scripts.
3. **If a teammate checks out your Git commit on another machine, what else is needed to reproduce or load your exact run?**
   * Because `mlflow.db`, `mlartifacts/`, and `data/raw/telco.csv` are excluded from Git, the teammate needs: (a) the exact `data/raw/telco.csv` file matching the recorded `data_sha256` hash, (b) the locked environment (`uv sync --locked` using `pyproject.toml` and `uv.lock`), and (c) network/file access to the shared MLflow tracking server and artifact store.
4. **Why log `git_commit` and `data_sha256` alongside model metrics?**
   * Together they establish end-to-end **provenance**: `git_commit` pins the exact code logic that produced the pipeline, while `data_sha256` proves the exact byte-level training dataset snapshot used, preventing silent data drift or untracked code edits from invalidating comparisons.
5. **What still belongs in Git versus MLflow?**