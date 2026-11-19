import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split


def main():
    tracking_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    alias_uri = "models:/telco-churn@candidate"
    print(f"Loading registered model from alias URI: {alias_uri} ...")
    loaded_pipeline = mlflow.sklearn.load_model(alias_uri)

    mv = client.get_model_version_by_alias("telco-churn", "candidate")
    run_model_uri = f"runs:/{mv.run_id}/churn_pipeline"
    original_pipeline = mlflow.sklearn.load_model(run_model_uri)

    df = pd.read_csv("data/raw/telco.csv")
    _, remaining_df = train_test_split(df, test_size=0.4, stratify=df["Churn"], random_state=42)
    valid_df, _ = train_test_split(remaining_df, test_size=0.5, stratify=remaining_df["Churn"], random_state=42)

    features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod"]
    new_customers = valid_df[features].head(5).copy()
    new_customers["TotalCharges"] = pd.to_numeric(new_customers["TotalCharges"], errors="coerce")

    predictions = loaded_pipeline.predict(new_customers).tolist()
    probabilities = loaded_pipeline.predict_proba(new_customers)[:, 1].tolist()

    expected_preds = original_pipeline.predict(new_customers).tolist()
    expected_probs = original_pipeline.predict_proba(new_customers)[:, 1].tolist()

    print("\n--- Fresh-Process Inference Verification ---")
    print(f"Resolved Version : v{mv.version} (Run ID: {mv.run_id})")
    print("Input customers (5 rows):")
    print(new_customers.to_string(index=False))
    print(f"\nOriginal class predictions : {expected_preds}")
    print(f"Reloaded class predictions : {predictions}")
    print(f"Class agreement            : {predictions == expected_preds}")

    max_prob_diff = float(np.max(np.abs(np.array(probabilities) - np.array(expected_probs))))
    print(f"\nOriginal probabilities     : {[round(p, 6) for p in expected_probs]}")
    print(f"Reloaded probabilities     : {[round(p, 6) for p in probabilities]}")
    print(f"Max probability difference : {max_prob_diff:.2e} (tolerance: 1e-5)")

    assert predictions == expected_preds, "Class predictions do not match!"
    assert max_prob_diff < 1e-5, "Probability predictions exceed tolerance!"
    print("\nSUCCESS: Reloaded model predictions match original training predictions!")


if __name__ == "__main__":
    main()