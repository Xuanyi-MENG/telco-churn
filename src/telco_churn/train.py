import json
import time
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    average_precision_score)
def main():
    print("Starting TP02 Training Pipeline...")
    data_path = Path("data/raw/telco.csv")
    df = pd.read_csv(data_path)
    # 1. Split Data (60/20/20)
    train_df, remaining_df = train_test_split(
        df, test_size=0.4, stratify=df["Churn"], random_state=42)
    valid_df, test_df = train_test_split(
        remaining_df, test_size=0.5, stratify=remaining_df["Churn"], random_state=42)
    print(f"Data split done: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")
    # 2. Prepare features and target
    for dataset in [train_df, valid_df, test_df]:
        dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")

    features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod"]
    X_train = train_df[features]
    y_train = train_df["Churn"].map({"No": 0, "Yes": 1})
    X_valid = valid_df[features]
    y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

    # 3. Build Pipeline Preprocessor
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = ["Contract", "InternetService", "PaymentMethod"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)])

    # 4. Define models to compare
    models = {
        "Baseline (Majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    results = []
    best_pipeline = None

    # 5. Train and evaluate each model
    for name, classifier in models.items():
        print(f"Training: {name}...")
        start_time = time.time()
        
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        y_pred = pipeline.predict(X_valid)
        
        if hasattr(pipeline, "predict_proba"):
            y_scores = pipeline.predict_proba(X_valid)[:, 1]
        else:
            y_scores = pipeline.predict(X_valid)

        metrics = {
            "Model": name,
            "Accuracy": round(accuracy_score(y_valid, y_pred), 4),
            "Precision": round(precision_score(y_valid, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_valid, y_pred, zero_division=0), 4),
            "F1": round(f1_score(y_valid, y_pred, zero_division=0), 4),
            "Avg Precision": round(average_precision_score(y_valid, y_scores), 4),
            "Fit Time (s)": round(fit_time, 3)
        }
        results.append(metrics)
        
        # Keep Random Forest as the selected artifact
        if name == "Random Forest":
            best_pipeline = pipeline

    # 6. Save results table
    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/model_comparison.csv", index=False)
    print("\n--- Model Comparison Saved to reports/model_comparison.csv ---")
    print(results_df.to_string(index=False))

    # 7. Save the fitted artifact
    if best_pipeline:
        model_path = Path("src/telco_churn/final_pipeline.joblib")
        joblib.dump(best_pipeline, model_path)
        print(f"\nSUCCESS: Final Pipeline saved to {model_path}")

if __name__ == "__main__":
    main()