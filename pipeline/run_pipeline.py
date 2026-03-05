import mlflow
from collect_data import collect_data
from engineer_features import engineer_features
from scrape_descriptions import scrape_descriptions
from build_model import build_model
import typer
from pathlib import Path

# Find the root directory, no matter where we are. 
def find_project_root(marker="README.md"):
    p = Path.cwd()
    while p != p.parent:
        if (p / marker).exists():
            return p
        p = p.parent
    raise RuntimeError("Project root not found")

PROJECT_ROOT = find_project_root()

def run_pipeline(test: bool = False):
    print("Starting pipeline run...")

    # Point our tracking writing at the mlruns folder - this is where the dashboard will read from 
    # we point at the SQL database because MLflow seems to really want that. 
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
    # mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("salary-predictor")

    with mlflow.start_run():
        mlflow.log_param("test_mode", test)

        collect_data(test=test)
        print("Data collection complete.")

        engineer_features()
        print("Feature engineering complete.")

        scrape_descriptions(test=test)
        print("Description scraping complete.")

        build_model()
        print("Model training complete.")

    print("Pipeline run finished.")

if __name__ == "__main__":
    typer.run(run_pipeline)