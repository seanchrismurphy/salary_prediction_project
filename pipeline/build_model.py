from pathlib import Path
import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Find the root directory, no matter where we are. 
def find_project_root(marker="README.md"):
    p = Path.cwd()
    while p != p.parent:
        if (p / marker).exists():
            return p
        p = p.parent
    raise RuntimeError("Project root not found")


def evaluate(y_true, y_pred):
    """
    Calculate regression metrics.
    Returns dict with rmse, mae, r2.
    """
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def build_model():
    PROJECT_ROOT = find_project_root()
    input_path = PROJECT_ROOT / "data/processed/feature_engineered_data.csv"
    
    df = pd.read_csv(input_path)
    
    input_path = PROJECT_ROOT / "data/raw/urls_with_descriptions.csv"
    descriptions = pd.read_csv(input_path)
    descriptions.rename(columns={'description': 'full_description'}, inplace=True)
    
    df = df.merge(descriptions[['redirect_url', 'full_description']], on='redirect_url', how='left')
    
    df['full_description']  = df['full_description'].fillna('')

    # Tried using the short descriptions where we don't have the full ones, but it degraded performance. 
    df = df[df['full_description'].str.len() > 30]
    
    X_structured = df.drop(columns=['avg_salary', 'title', 'description'])
    X_structured.columns

    cat_vars = ['contract_type', 'contract_time', 'category.label', 'location.area_length', 
                # 'location_country', 
                'location_state', 'location_region_abridged', 'location_city_abridged', 'missing_long_lat']

    num_vars = ['longitude', 'latitude']
    
    print(f"Loaded {len(df)} records.")

    print("Fitting title TF-IDF...")
    # --- Text: title (TF-IDF, no reduction) ---
    title_tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_title_full = title_tfidf.fit_transform(df['title'].fillna(''))
    
    print(f"Title matrix: {X_title_full.shape}")

    print("Fitting description TF-IDF + SVD...")
    # --- Text: description (TF-IDF + SVD) ---
    desc_tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    desc_svd = TruncatedSVD(n_components=100, random_state=42)
    X_desc_sparse = desc_tfidf.fit_transform(df['full_description'].fillna(''))
    X_desc_full = desc_svd.fit_transform(X_desc_sparse)
    
    print(f"Description matrix: {X_desc_full.shape}")

    print("Encoding structured features...")
    # --- Structured features ---
    ohe_prod = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    imputer_prod = SimpleImputer(strategy='median')
    scaler_prod = StandardScaler()

    cat_encoded_full = ohe_prod.fit_transform(df[cat_vars])
    cont_imputed_full = imputer_prod.fit_transform(df[num_vars])
    cont_scaled_full = scaler_prod.fit_transform(cont_imputed_full)

    X_structured_full = np.hstack([cat_encoded_full, cont_scaled_full])

    # --- Combine and fit final model ---
    X_full = np.hstack([
        X_title_full.toarray(),
        X_desc_full,
        X_structured_full
    ])
    y_full = df['avg_salary'].values

    print("Fitting Ridge model...")

    final_model = Ridge(alpha=1.0)
    final_model.fit(X_full, y_full)
    
    y_pred = final_model.predict(X_full)
    metrics = evaluate(y_full, y_pred)
    print(f"Training MAE: ${metrics['mae']:,.0f} (in-sample — model trained on full dataset)")
    
    print(f"Final model fitted on {len(y_full)} samples.")

    # --- Save bundle ---
    pipeline_bundle = {
        'title_tfidf': title_tfidf,
        'desc_tfidf': desc_tfidf,
        'desc_svd': desc_svd,
        'ohe': ohe_prod,
        'imputer': imputer_prod,
        'scaler': scaler_prod,
        'model': final_model,
        'cat_vars': cat_vars,
        'num_vars': num_vars,
    }

    output_path = PROJECT_ROOT / 'models/salary_pipeline.joblib'
    output_path.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline_bundle, output_path)
    print(f"Pipeline saved to {output_path}")
    
if __name__ == "__main__":
    build_model()