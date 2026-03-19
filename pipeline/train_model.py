import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import find_project_root

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_wrapper import SalaryPipelineWrapper, predict_pipeline

PROJECT_ROOT = find_project_root()

logging.getLogger("mlflow.pyfunc").setLevel(logging.ERROR)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)


def evaluate(y_true, y_pred):
    """
    Calculate regression metrics.
    Returns dict with rmse, mae, r2.
    """
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def fit_pipeline(df, cat_vars, num_vars):
    """Fit all transformers and model on the given dataframe. Returns bundle."""

    title_tfidf = TfidfVectorizer(
        max_features=10_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True
    )
    X_title = title_tfidf.fit_transform(df["title"].fillna(""))

    desc_tfidf = TfidfVectorizer(
        max_features=10_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True
    )
    desc_svd = TruncatedSVD(n_components=100, random_state=42)
    X_desc = desc_svd.fit_transform(
        desc_tfidf.fit_transform(df["full_description"].fillna(""))
    )

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_structured = np.hstack(
        [
            ohe.fit_transform(df[cat_vars]),
            scaler.fit_transform(imputer.fit_transform(df[num_vars])),
        ]
    )

    X = np.hstack([X_title.toarray(), X_desc, X_structured])
    y = df["avg_salary"].values

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    return (
        {
            "title_tfidf": title_tfidf,
            "desc_tfidf": desc_tfidf,
            "desc_svd": desc_svd,
            "ohe": ohe,
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
            "cat_vars": cat_vars,
            "num_vars": num_vars,
        },
        X,
        y,
    )





def get_production_val_mae(client):
    try:
        versions = client.search_model_versions("name='salary-predictor'")
        production_versions = [v for v in versions if v.current_stage == "Production"]
        production_versions = sorted(production_versions, key=lambda v: int(v.version), reverse=True)
        previous_version = production_versions[0] if production_versions else None

        if previous_version:
            run = client.get_run(previous_version.run_id)
            previous_mae = run.data.metrics.get("model/val_mae")
        else:
            print(f"Warning: could not retrieve previous MAE")
            previous_mae = None
        return(previous_mae)
    except Exception as e:
        print(f"Warning: could not retrieve previous MAE: {e}")
        return None


def train_model():

    # --- Load and merge data ---
    df = pd.read_csv(PROJECT_ROOT / "data/processed/feature_engineered_data.csv")
    descriptions = pd.read_csv(PROJECT_ROOT / "data/raw/urls_with_descriptions.csv")
    descriptions.rename(columns={"description": "full_description"}, inplace=True)
    df = df.merge(
        descriptions[["redirect_url", "full_description"]],
        on="redirect_url",
        how="left",
    )
    df["full_description"] = df["full_description"].fillna("")
    df = df[df["full_description"].str.len() > 30]
    print(f"Loaded {len(df)} records.")

    cat_vars = [
        "contract_type",
        "contract_time",
        "category_label",
        "location_area_length",
        "location_state",
        "location_region_abridged",
        "location_city_abridged",
        "missing_long_lat",
    ]

    num_vars = ["longitude", "latitude"]

    # --- Time-based split ---
    df = df.sort_values("created").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    print(f"Train: {len(train_df)} records, Val: {len(val_df)} records.")

    # --- Fit on train, evaluate on holdout ---
    print("Fitting on training set...")
    train_bundle, X_train, y_train = fit_pipeline(train_df, cat_vars, num_vars)

    y_train_pred = train_bundle["model"].predict(X_train)
    train_metrics = evaluate(y_train, y_train_pred)
    print(f"Train MAE: ${train_metrics['mae']:,.0f}")

    val_preds = predict_pipeline(train_bundle, val_df)
    val_metrics = evaluate(val_df["avg_salary"].values, val_preds)
    print(f"Val MAE: ${val_metrics['mae']:,.0f}")

    # --- Refit on full dataset for production ---
    print("Refitting on full dataset...")
    final_bundle, _, _ = fit_pipeline(df, cat_vars, num_vars)

    # --- Save ---
    output_path = PROJECT_ROOT / "models/salary_pipeline.joblib"
    output_path.parent.mkdir(exist_ok=True)
    joblib.dump(final_bundle, output_path)
    print(f"Pipeline saved to {output_path}")

    # --- MLflow logging ---
    mlflow.log_param("model_type", "Ridge")
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("title_tfidf_max_features", 10_000)
    mlflow.log_param("desc_tfidf_max_features", 10_000)
    mlflow.log_param("desc_svd_components", 100)
    mlflow.log_param("train_records", len(train_df))
    mlflow.log_param("val_records", len(val_df))

    mlflow.log_metric("model/train_mae", train_metrics["mae"])
    mlflow.log_metric("model/train_rmse", train_metrics["rmse"])
    mlflow.log_metric("model/train_r2", train_metrics["r2"])
    mlflow.log_metric("model/val_mae", val_metrics["mae"])
    mlflow.log_metric("model/val_rmse", val_metrics["rmse"])
    mlflow.log_metric("model/val_r2", val_metrics["r2"])
    mlflow.log_metric("model/feature_matrix_cols", X_train.shape[1])

    run_id = mlflow.active_run().info.run_id

    joblib.dump(final_bundle, output_path)

    # Note currently not used as it's breaking the version of mlflow we're using. 
    input_example = train_df[["title", "full_description"] + cat_vars + num_vars].iloc[
        :3
    ]

    for col in input_example.select_dtypes(include=["int64", "bool"]).columns:
        input_example[col] = input_example[col].astype("float64")
        input_example[cat_vars] = input_example[cat_vars].fillna("unknown")

    sample_predictions = predict_pipeline(final_bundle, input_example)
    signature = infer_signature(input_example, sample_predictions)
    
    # log the full bundle as a registered model
    model_info = mlflow.pyfunc.log_model(
        artifact_path="salary_pipeline",
        python_model=SalaryPipelineWrapper(),
        artifacts={"pipeline_bundle": str(output_path)},
        registered_model_name="salary-predictor",
        signature = signature,
        input_example=input_example,
    )

    version = model_info.registered_model_version

    client = mlflow.tracking.MlflowClient()

    previous_mae = get_production_val_mae(client)

    # We deploy the model to production so long as MAE is not more than 5% worse than our previous model.
    if previous_mae is None or val_metrics["mae"] < previous_mae * 1.05:
        client.transition_model_version_stage(
            name="salary-predictor",
            version=version,
            stage="Production",
            archive_existing_versions=True
        )

        mlflow.set_tag("deployment", "promoted")

        if previous_mae is None:
            print(
                f"Model promoted to Production. Val MAE: ${val_metrics['mae']:,.0f} (no previous model)"
            )
        else:
            pct_change = (val_metrics["mae"] - previous_mae) / previous_mae * 100
            direction = "improvement" if pct_change < 0 else "degradation"
            print(
                f"Model promoted to Production. Val MAE: ${val_metrics['mae']:,.0f} ({abs(pct_change):.1f}% {direction} vs previous ${previous_mae:,.0f})"
            )

    else:
        mlflow.set_tag("deployment", "blocked")
        print(
            f"Deployment blocked. New MAE ${val_metrics['mae']:,.0f} vs previous ${previous_mae:,.0f}"
        )

    return train_metrics, val_metrics


if __name__ == "__main__":
    train_model()
