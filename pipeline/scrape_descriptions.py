import json
import os
import random
import time
from datetime import datetime

import mlflow
import pandas as pd
import requests
import typer
from bs4 import BeautifulSoup

from utils import find_project_root, safe_save_csv

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.5",
}


def get_full_description(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        scripts = soup.find_all("script", type="application/ld+json")

        for script in scripts:
            data = json.loads(script.string)
            if isinstance(data, list):
                data = data[0]
            if "description" in data:
                return data.get("description")
        return None

    except Exception as e:
        print(f"Error fetching {url[:70]}: {e}")
        return None


def scrape_descriptions(test: bool = False):
    PROJECT_ROOT = find_project_root()
    urls_source = PROJECT_ROOT / "data/raw/redirect_urls.json"
    output_file = PROJECT_ROOT / "data/raw/urls_with_descriptions.csv"
    # checkpoint_file = PROJECT_ROOT / "data/progress_tracking/urls_with_descriptions.csv"
    save_interval = 50

    # Load all redirect URLs and filter to adzuna detail pages
    with open(urls_source, "r") as f:
        redirect_urls = json.load(f)

    urls = pd.DataFrame({"redirect_url": redirect_urls})
    urls = urls[urls["redirect_url"].str.contains("details")]
    urls = urls.reset_index(drop=True)
    print(f"Loaded {len(urls)} /details/ URLs from {urls_source}")

    # Load previously scraped results and exclude already-done URLs
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        already_scraped = set(existing["redirect_url"])
        urls = urls[~urls["redirect_url"].isin(already_scraped)].reset_index(drop=True)
        print(
            f"Skipping {len(already_scraped)} already-scraped URLs, {len(urls)} remaining"
        )
    else:
        existing = pd.DataFrame(columns=["redirect_url", "description"])

    if len(urls) == 0:
        print("Nothing new to scrape.")
        mlflow.log_metric("scrape/urls_to_scrape", 0)
        mlflow.log_metric("scrape/success_pct", 0)
        return

    if test:
        print("Running in test mode")
        urls = urls[:5]

    start_time = datetime.now()

    mlflow.log_metric("scrape/urls_to_scrape", len(urls))

    urls["description"] = None
    total = len(urls)
    start_time = datetime.now()

    for i, (index, row) in enumerate(urls.iterrows()):
        url = row["redirect_url"]
        description = get_full_description(url)
        urls.at[index, "description"] = description

        status = "ok" if description else "no description"
        print(f"[{i+1}/{total}] {status}: {url[:70]}")

        # Save checkpoint
        if (i + 1) % save_interval == 0:
            combined = pd.concat([existing, urls.iloc[: i + 1]], ignore_index=True)
            safe_save_csv(combined, output_file)

            elapsed = datetime.now() - start_time
            avg = elapsed.total_seconds() / (i + 1)
            remaining = (total - i - 1) * avg / 3600
            print(f"Saved. {elapsed} elapsed, ~{remaining:.1f}h remaining")

        if i + 1 < total:
            time.sleep(random.uniform(2, 3))

    # Final save
    combined = pd.concat([existing, urls], ignore_index=True)
    safe_save_csv(combined, output_file)
    # combined.to_csv(checkpoint_file, index=False)

    # log success rate of this batch only
    success_pct = round(urls["description"].notna().mean() * 100, 1)
    elapsed = (datetime.now() - start_time).total_seconds()

    mlflow.log_metrics(
        {"scrape/success_pct": success_pct, "scrape/elapsed_seconds": elapsed}
    )

    print(
        f"Done. {len(combined)} total URLs with descriptions. Success rate: {success_pct}%"
    )


if __name__ == "__main__":
    typer.run(scrape_descriptions)
