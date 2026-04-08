import fcntl
import os
from datetime import datetime

import mlflow
import typer
from dotenv import load_dotenv

from collect_data import collect_data
from engineer_features import engineer_features
from scrape_descriptions import scrape_descriptions
from train_model import train_model

# PROJECT_ROOT = find_project_root()
# lock_file = PROJECT_ROOT / "pipeline.lock"

load_dotenv()

def run_pipeline(test: bool = False):

    start_time = datetime.now()
    # Check if pipeline is already running
    # lock = open(lock_file, "w")
    # try:
    #     fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # except OSError:
    #     print("Pipeline is already running. Exiting.")
    #     return

    print("Starting pipeline run...")

    # try:
    mlflow.end_run()

    # Point our tracking writing at the mlruns folder - this is where the dashboard will read from
    # we point at the SQL database because MLflow seems to really want that.
    print(f'MLflow database at: {os.environ.get("MLFLOW_TRACKING_URI")}')
    
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    print(f"Tracking URI at: {mlflow.get_tracking_uri()}")
    
    # mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("salary-predictor")
    
    

    with mlflow.start_run():

        if test:
            mlflow.set_tag("run_type", "test")
        else:
            mlflow.set_tag("run_type", "production")

        collect_data(test=test)
        print("Data collection complete.")

        engineer_features()
        print("Feature engineering complete.")

        scrape_descriptions(test=test)
        print("Description scraping complete.")

        train_model()
        print("Model training complete.")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric(
                "overall/total_total_time_elapsed", elapsed
                )

    print("Pipeline run finished.")
    
    # finally:
    #    fcntl.flock(lock, fcntl.LOCK_UN)
    #    lock.close()


if __name__ == "__main__":
    typer.run(run_pipeline)
