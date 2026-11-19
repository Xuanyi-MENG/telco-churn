import hashlib
import json
import subprocess
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def get_git_info() -> tuple[str, bool]:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return commit, bool(status)
    except Exception:
        return "unknown", True


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def main():
    print("Starting TP03 Tracked Training Pipeline...")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("telco-churn-tp03")

    data_path = Path("data/raw/telco.csv")
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    git_commit, git_dirty = get_git_info()

    df = pd.read_csv(data_path)

    split_seed = 42
    threshold = 0.5
    train_df, remaining_df = train_test_split(
        df, test_size=0.4, stratify=df["Churn"], random_state=split_seed
    )
    valid_df, test_df = train_test_split(
        remaining_df, test_size=0.5, stratify=remaining_df["Churn"], random_state=split_seed
    )
    print(f"Data split done: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")

    split_ids_signature = hashlib.sha256(
        json.dumps(
            {
                "train_ids": train_df["customerID"].tolist(),
                "valid_ids": valid_df["customerID"].tolist(),
                "test_ids": test_df["customerID"].tolist(),
            }
        ).encode("utf-8")
    ).hexdigest()

    for dataset in [train_df, valid_df, test_df]:
        dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")

    features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod"]
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = ["Contract", "InternetService", "PaymentMethod"]

    X_train = train_df[features].copy()
    y_train = train_df["Churn"].map({"No": 0, "Yes": 1})
    X_valid = valid_df[features].copy()
    y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

    train_dataset_df = train_df[features + ["Churn"]].copy()
    valid_dataset_df = valid_df[features + ["Churn"]].copy()
    mlflow_train_ds = mlflow.data.from_pandas(train_dataset_df, name="telco_train", targets="Churn")
    mlflow_valid_ds = mlflow.data.from_pandas(valid_dataset_df, name="telco_valid", targets="Churn")

    candidates = [
        {
            "run_name": "baseline-most-frequent",
            "classifier": DummyClassifier(strategy="most_frequent"),
            "hyperparams": {"strategy": "most_frequent"},
        },
        {
            "run_name": "logistic-C1",
            "classifier": LogisticRegression(C=1.0, max_iter=1000, random_state=split_seed),
            "hyperparams": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        },
        {
            "run_name": "decision-tree-depth5",
            "classifier": DecisionTreeClassifier(max_depth=5, random_state=split_seed),
            "hyperparams": {"max_depth": 5, "criterion": "gini"},
        },
        {
            "run_name": "random-forest-n100-depth8",
            "classifier": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=split_seed),
            "hyperparams": {"n_estimators": 100, "max_depth": 8},
        },
    ]

    reports_dir = Path("reports/tp03_artifacts")
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = []
    best_ap = -1.0
    best_candidate_info = None

    for item in candidates:
        run_name = item["run_name"]
        classifier = item["classifier"]
        hyperparams = item["hyperparams"]

        print(f"\n--- Running MLflow Run: {run_name} ---")
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(
                {
                    "model_type": type(classifier).__name__,
                    "split_seed": split_seed,
                    "threshold": threshold,
                    **hyperparams,
                }
            )
            mlflow.set_tags(
                {
                    "git_commit": git_commit,
                    "git_dirty": str(git_dirty),
                    "data_sha256": data_sha256,
                }
            )
            mlflow.log_input(mlflow_train_ds, context="training")
            mlflow.log_input(mlflow_valid_ds, context="validation")

            preprocessor = build_preprocessor(numeric_features, categorical_features)
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

            start_time = time.time()
            pipeline.fit(X_train, y_train)
            fit_time = time.time() - start_time

            classes_list = list(pipeline.classes_)
            pos_idx = classes_list.index(1)

            if hasattr(pipeline, "predict_proba"):
                y_scores = pipeline.predict_proba(X_valid)[:, pos_idx]
            else:
                y_scores = pipeline.predict(X_valid).astype(float)

            y_pred = (y_scores >= threshold).astype(int)

            val_ap = float(average_precision_score(y_valid, y_scores))
            val_prec = float(precision_score(y_valid, y_pred, zero_division=0))
            val_rec = float(recall_score(y_valid, y_pred, zero_division=0))
            val_f1 = float(f1_score(y_valid, y_pred, zero_division=0))
            val_acc = float(accuracy_score(y_valid, y_pred))

            mlflow.log_metrics(
                {
                    "val_average_precision": round(val_ap, 4),
                    "val_precision": round(val_prec, 4),
                    "val_recall": round(val_rec, 4),
                    "val_f1": round(val_f1, 4),
                    "val_accuracy": round(val_acc, 4),
                    "fit_time_sec": round(fit_time, 4),
                }
            )

            cm = confusion_matrix(y_valid, y_pred, labels=list((0, 1)))
            fig, ax = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No (0)", "Yes (1)"])
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"Validation CM - {run_name} (labels=[0, 1])")
            cm_path = reports_dir / f"cm_{run_name}.png"
            fig.savefig(cm_path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))

            provenance = {
                "run_name": run_name,
                "source_csv": str(data_path),
                "data_sha256": data_sha256,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "features": features,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "target_mapping": {"No": 0, "Yes": 1},
                "confusion_matrix_labels": list((0, 1)),
                "split_policy": "60/20/20 stratified train/valid/test",
                "split_seed": split_seed,
                "split_allocation_sha256": split_ids_signature,
                "primary_metric": "val_average_precision",
                "decision_threshold": threshold,
            }
            mlflow.log_dict(provenance, "provenance.json")

            model_uri = None
            if run_name != "baseline-most-frequent":
                model_info = mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    name="churn_pipeline",
                    input_example=X_train.head(3),
                    serialization_format="cloudpickle",
                )
                model_uri = model_info.model_uri

            print(f"Run ID: {run.info.run_id} | val_average_precision={val_ap:.4f} | val_f1={val_f1:.4f}")
            if model_uri:
                print(f"Logged Model URI: {model_uri}")

            results.append(
                {
                    "Run ID": run.info.run_id,
                    "Candidate": run_name,
                    "Validation AP": round(val_ap, 4),
                    "Validation Precision": round(val_prec, 4),
                    "Validation Recall": round(val_rec, 4),
                    "Validation F1": round(val_f1, 4),
                    "Fit Time (s)": round(fit_time, 4),
                    "Model URI": model_uri or "N/A",
                }
            )

if __name__ == "__main__":
    main()

