import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import find_project_root

PROJECT_ROOT = find_project_root()


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


def predict_pipeline(bundle, df):
    """Transform df using fitted bundle and return predictions."""
    X_title = bundle["title_tfidf"].transform(df["title"].fillna(""))
    X_desc = bundle["desc_svd"].transform(
        bundle["desc_tfidf"].transform(df["full_description"].fillna(""))
    )
    X_structured = np.hstack(
        [
            bundle["ohe"].transform(df[bundle["cat_vars"]]),
            bundle["scaler"].transform(
                bundle["imputer"].transform(df[bundle["num_vars"]])
            ),
        ]
    )
    X = np.hstack([X_title.toarray(), X_desc, X_structured])
    return bundle["model"].predict(X)


def get_production_val_mae(client):
    try:
        production = client.get_latest_versions(
            "salary-predictor", stages=["Production"]
        )
        if not production:
            return None
        run_id = production[0].run_id
        run = client.get_run(run_id)
        return run.data.metrics.get("model/val_mae")
    except:
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
        "category.label",
        "location.area_length",
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

    # THis step actually registers the model so that it will show up on the 'Models' page of MLflow.
    class SalaryPipelineWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import joblib

            self.bundle = joblib.load(context.artifacts["pipeline_bundle"])

        def predict(self, context, model_input: pd.DataFrame) -> pd.Series:
            return predict_pipeline(self.bundle, model_input)

    joblib.dump(final_bundle, output_path)

    # Log the full bundle as a registered model
    model_info = mlflow.pyfunc.log_model(
        name="salary_pipeline",
        python_model=SalaryPipelineWrapper(),
        artifacts={"pipeline_bundle": str(output_path)},
        registered_model_name="salary-predictor",
    )

    version = model_info.registered_model_version

    client = mlflow.tracking.MlflowClient()

    previous_mae = get_production_val_mae(client)

    # We deploy the model to production so long as MAE is not more than 5% worse than our previous model.
    if previous_mae is None or val_metrics["mae"] < previous_mae * 1.05:
        client.transition_model_version_stage(
            name="salary-predictor", version=version, stage="Production"
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
