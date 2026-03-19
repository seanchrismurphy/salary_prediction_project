import json
import shutil

import mlflow
import pandas as pd

from utils import find_project_root, safe_save_csv


def gini(values):
    values = sorted(values)
    n = len(values)
    cumsum = sum((i + 1) * v for i, v in enumerate(values))
    return (2 * cumsum) / (n * sum(values)) - (n + 1) / n

def engineer_features(lower = 50000, upper = 300000, min_location_count = 50):
    PROJECT_ROOT = find_project_root()

    path = PROJECT_ROOT / "data/raw/api_progress_fixed.json"
    print(path) 
    
    with path.open("r") as f:
        all_data = json.load(f)
    
    df = pd.json_normalize(all_data)    
    
    df = df.drop(columns=["adref"])
    # Drop any column that contains __CLASS__
    df = df.drop(columns=[col for col in df.columns if "__CLASS__" in col])
    
    df['location.area_length'] = df['location.area'].apply(len)
    
    # create location_1, location_2, location_3 by splitting location.area lists into components, 
    # accounting for the fact that not all lists have all 5 components
    location_cols = ['location_country', 'location_state', 'location_region', 'location_city', 'location_area', 'location_suburb']
    location_df = df['location.area'].apply(lambda x: pd.Series(x + [None] * (6 - len(x))))
    location_df.columns = location_cols
    df = pd.concat([df, location_df], axis=1)
    
    # Coutn unique rows defined by title + description + salary
    df['unique_id'] = df['title'].str.lower() + '|' + df['description'] + '|' + df['salary_min'].astype(str) + '|' + df['salary_max'].astype(str)

    # Count how many rows we would have if we dropped duplicates based on unique_id
    original_count = len(df)
    unique_count = df['unique_id'].nunique()
    duplicates_removed = original_count - unique_count
    
    print(f"Original rows: {original_count}")
    print(f"Unique rows: {unique_count}")
    df['id'] = df['id'].astype(int)
    # Drop rows using unique_id. Keep earliest posting (lowest id)
    df = df.sort_values('id').drop_duplicates(subset='unique_id', keep='first').reset_index(drop=True)
    
    
    both = df[(df['salary_min'] > 0) & (df['salary_max'] > 0)]
    ratio = (both['salary_max'] / both['salary_min']).median()
    
    # Fill in empty text data
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df['missing_long_lat'] = df['longitude'].isna() | df['latitude'].isna()
    
    df['imputed_salary_min'] = df.apply(lambda row: row['salary_min'] if row['salary_min'] > 0 else row['salary_max'] / ratio, axis=1)
    df['imputed_salary_max'] = df.apply(lambda row: row['salary_max'] if row['salary_max'] > 0 else row['salary_min'] * ratio, axis=1)
    df['avg_salary'] = (df['imputed_salary_min'] + df['imputed_salary_max']) / 2

    below_lower = (df['avg_salary'] < lower).sum()
    df = df[df['avg_salary'] >= lower].copy()
    
    median_salary_before_clip = df['avg_salary'].median()
    above_upper = (df['avg_salary'] > upper).sum()
    median_salary_after_clip = df['avg_salary'].median()
    
    df['avg_salary'] = df['avg_salary'].clip(upper=upper)
    company_counts = df['company.display_name'].value_counts()
    df['company_listing_count'] = df['company.display_name'].map(company_counts)
    
    for col in ['location_region', 'location_city']:
        counts = df[col].value_counts()
        # Create new column for the abridged versions of the location 
        df[f'{col}_abridged'] = df[col].where(df[col].map(counts) >= min_location_count, 'Other')
    
    # Rename the location.area_length column to location_area_length
    df = df.rename(columns={
        'category.label': 'category_label',
        'location.area_length': 'location_area_length',
    })
    
    output_path = PROJECT_ROOT / "data/processed/feature_engineered_data.csv"
    safe_save_csv(df, output_path)
    
    
    # For the locations, we log the gini coefficient (gets the distribution in one number), proportoin in the top 1, and proportion "Other" (except for State)
    location_levels = [
    ('location_state', 'location_state'),
    ('location_region', 'location_region_abridged'),
    ('location_city', 'location_city_abridged'),
    ]   
    
    for level, col in location_levels:
        proportions = df[col].value_counts(normalize=True)
        mlflow.log_metric(f"engineer/{level}_gini", gini(proportions.values))
        mlflow.log_metric(f"engineer/{level}_top1_pct", round(proportions.iloc[0] * 100, 1))
        
        if col.endswith('_abridged'):
            mlflow.log_metric(f"engineer/{level}_other_pct", 
                            round(proportions.get("Other", 0) * 100, 1))
    

    # Store parameters around the feature engineering process. 
    mlflow.log_metrics({
        "engineer/original_count": original_count,
        "engineer/unique_count": unique_count,
        "engineer/duplicates_removed": duplicates_removed,
        "engineer/removed_below_lower": below_lower,
        "engineer/clipped_above_upper": above_upper,
        "engineer/median_salary_before_clip": median_salary_before_clip,
        "engineer/median_salary_after_clip": median_salary_after_clip,
        "engineer/final_count": len(df),
    })
    
    mlflow.log_params({
        "engineer/lower_bound": lower,
        "engineer/upper_bound": upper,
        "engineer/min_location_count": min_location_count,
    })
    
    # Also save just the redirect URLs in JSON format
    temp_path = (PROJECT_ROOT / "data/raw/redirect_urls.json").with_suffix(".tmp")
    df['redirect_url'].to_json(temp_path, orient='records')
    shutil.move(temp_path, PROJECT_ROOT / "data/raw/redirect_urls.json")
    
if __name__ == "__main__":
    engineer_features()