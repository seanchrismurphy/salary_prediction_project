import json
import os
import time
from datetime import datetime

import mlflow
import requests
import typer
from dotenv import load_dotenv

from utils import find_project_root, safe_save_json


def collect_data(test: bool = False):

    load_dotenv()

    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    PROJECT_ROOT = find_project_root()

    search_terms = [
        # Data & Analytics
        "data scientist",
        "data analyst",
        "data engineer",
        "analytics engineer",
        "business intelligence analyst",
        "quantitative analyst",
        "statistician",
        "data architect",
        "machine learning engineer",
        "AI researcher",
        "NLP engineer",
        "computer vision engineer",
        "MLOps engineer",
        "research scientist",
        # Software & Engineering (Technical)
        "software engineer",
        "backend engineer",
        "frontend engineer",
        "full stack developer",
        "mobile developer",
        "iOS developer",
        "Android developer",
        "embedded systems engineer",
        "firmware engineer",
        "platform engineer",
        "site reliability engineer",
        "devops engineer",
        "cloud architect",
        "solutions architect",
        "security engineer",
        "penetration tester",
        "blockchain developer",
        "game developer",
        "QA engineer",
        "database administrator",
        "network engineer",
        # Product & Design
        "product manager",
        "product designer",
        "UX designer",
        "UI designer",
        "UX researcher",
        "graphic designer",
        "motion designer",
        "brand designer",
        "creative director",
        "design director",
        "content strategist",
        "information architect",
        # Project & Operations Management
        "project manager",
        "program manager",
        "scrum master",
        "agile coach",
        "operations manager",
        "supply chain manager",
        "logistics coordinator",
        "procurement manager",
        "facilities manager",
        "change management consultant",
        "chief of staff",
        # Business & Strategy
        "business analyst",
        "strategy consultant",
        "management consultant",
        "corporate development manager",
        "business development manager",
        "partnerships manager",
        "growth manager",
        "venture capital analyst",
        "private equity associate",
        "investment analyst",
        "mergers and acquisitions analyst",
        # Finance & Accounting
        "accountant",
        "financial analyst",
        "financial controller",
        "CFO",
        "actuary",
        "tax accountant",
        "forensic accountant",
        "treasury analyst",
        "credit analyst",
        "risk analyst",
        "compliance officer",
        "auditor",
        "wealth manager",
        "financial planner",
        "insurance underwriter",
        "mortgage broker",
        # Sales & Marketing
        "sales manager",
        "account executive",
        "sales development representative",
        "customer success manager",
        "marketing manager",
        "digital marketing manager",
        "SEO specialist",
        "performance marketing manager",
        "email marketing specialist",
        "social media manager",
        "copywriter",
        "technical writer",
        "public relations manager",
        "communications manager",
        "market research analyst",
        "pricing analyst",
        "revenue operations manager",
        "category manager",
        "e-commerce manager",
        # People & Culture
        "human resources manager",
        "HR business partner",
        "talent acquisition specialist",
        "recruiter",
        "learning and development manager",
        "compensation analyst",
        "organisational development consultant",
        "workforce planning analyst",
        "diversity and inclusion manager",
        "employee relations specialist",
        # Legal & Compliance
        "lawyer",
        "solicitor",
        "paralegal",
        "legal counsel",
        "compliance manager",
        "privacy officer",
        "contract manager",
        "intellectual property analyst",
        "regulatory affairs specialist",
        # Engineering (Physical Disciplines)
        "civil engineer",
        "mechanical engineer",
        "electrical engineer",
        "structural engineer",
        "geotechnical engineer",
        "environmental engineer",
        "chemical engineer",
        "industrial engineer",
        "aerospace engineer",
        "biomedical engineer",
        "process engineer",
        "project engineer",
        "systems engineer",
        # Healthcare & Allied Health
        "nurse",
        "registered nurse",
        "nurse practitioner",
        "physician",
        "surgeon",
        "pharmacist",
        "physiotherapist",
        "occupational therapist",
        "speech pathologist",
        "psychologist",
        "psychiatrist",
        "radiographer",
        "medical researcher",
        "clinical trial manager",
        "health economist",
        "health informatician",
        "hospital administrator",
        # Education & Research
        "teacher",
        "principal",
        "curriculum designer",
        "instructional designer",
        "academic researcher",
        "university lecturer",
        "policy analyst",
        "economist",
        "sociologist",
        "urban planner",
        "archivist",
        "librarian",
        # Executive & Leadership
        "CEO",
        "COO",
        "CTO",
        "CMO",
        "CHRO",
        "general manager",
        "managing director",
        "vice president",
        "director of engineering",
        "head of product",
        "head of data",
        "head of finance",
        "head of marketing",
        # Sustainability & ESG
        "sustainability manager",
        "ESG analyst",
        "carbon analyst",
        "environmental consultant",
        "climate risk analyst",
        "corporate social responsibility manager",
        # Architecture & Built Environment
        "architect",
        "interior designer",
        "landscape architect",
        "quantity surveyor",
        "building surveyor",
        "town planner",
        "construction manager",
        "property manager",
        "real estate analyst",
        "valuer",
        # Media, Publishing & Creative
        "journalist",
        "editor",
        "film producer",
        "video editor",
        "podcast producer",
        "photographer",
        "illustrator",
        "game designer",
        # Science & Research
        "biologist",
        "chemist",
        "physicist",
        "geologist",
        "materials scientist",
        "data science researcher",
        "epidemiologist",
        "biostatistician",
        "laboratory manager",
        "clinical psychologist",
        # Trades & Construction
        "electrician",
        "plumber",
        "carpenter",
        "cabinet maker",
        "bricklayer",
        "plasterer",
        "tiler",
        "roofer",
        "painter",
        "concreter",
        "welder",
        "boilermaker",
        "refrigeration mechanic",
        "air conditioning technician",
        "auto mechanic",
        "diesel mechanic",
        "panel beater",
        "glazier",
        "locksmith",
        "scaffolder",
        "rigger",
        "crane operator",
        "forklift operator",
        "earthmoving operator",
        "arborist",
        "gardener",
        "landscaper",
        "pool technician",
        # Retail & Customer Service
        "retail assistant",
        "store manager",
        "visual merchandiser",
        "checkout operator",
        "customer service representative",
        "call centre operator",
        "pharmacy assistant",
        "optical dispenser",
        "florist",
        "butcher",
        "baker",
        "deli assistant",
        "petrol station attendant",
        # Hospitality & Food Service
        "chef",
        "head chef",
        "sous chef",
        "pastry chef",
        "cook",
        "kitchen hand",
        "waiter",
        "barista",
        "bartender",
        "front of house manager",
        "restaurant manager",
        "hotel receptionist",
        "concierge",
        "housekeeper",
        "event coordinator",
        "catering manager",
        # Care & Domestic
        "nanny",
        "au pair",
        "childcare worker",
        "early childhood educator",
        "disability support worker",
        "aged care worker",
        "personal care assistant",
        "home care worker",
        "social worker",
        "youth worker",
        "community support worker",
        "domestic cleaner",
        "commercial cleaner",
        "office cleaner",
        "laundry attendant",
        # Transport & Logistics
        "truck driver",
        "delivery driver",
        "courier",
        "bus driver",
        "taxi driver",
        "rideshare driver",
        "train driver",
        "pilot",
        "flight attendant",
        "warehouse worker",
        "storeperson",
        "pick packer",
        "inventory controller",
        "freight coordinator",
        "customs broker",
        # Security & Emergency Services
        "security guard",
        "loss prevention officer",
        "crowd controller",
        "police officer",
        "firefighter",
        "paramedic",
        "ambulance officer",
        "corrections officer",
        # Personal Services
        "hairdresser",
        "barber",
        "beauty therapist",
        "nail technician",
        "massage therapist",
        "personal trainer",
        "fitness instructor",
        "swim instructor",
        "dog groomer",
        "veterinary nurse",
        # Administration
        "receptionist",
        "office administrator",
        "executive assistant",
        "data entry operator",
        "accounts payable officer",
        "payroll officer",
        "medical receptionist",
        "legal secretary",
        "court registrar",
    ]

    if test:
        print("Running in test mode")
        search_terms = ["data scientist"]

    progress_file = PROJECT_ROOT / "data/raw/api_progress_fixed.json"

    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            all_data = json.load(f)
        print(f"Loaded {len(all_data)} existing results")
    else:
        all_data = []

    initial_count = len(all_data)

    seen_ids = {job["id"] for job in all_data}
    api_calls = 0

    start_time = datetime.now()

    for term in search_terms:
        for page in range(1, 6):  # 5 pages per term = 250 results per term
            url = (
                f"https://api.adzuna.com/v1/api/jobs/au/search/{page}"
                f"?app_id={app_id}&app_key={app_key}"
                f"&results_per_page=50&salary_min=1"
                f"&what={term.replace(' ', '%20')}"
            )

            try:
                response = requests.get(url)
                response.raise_for_status()
                results = response.json()["results"]

                new = [job for job in results if job["id"] not in seen_ids]
                for job in new:
                    job["search_term"] = term
                    seen_ids.add(job["id"])

                all_data.extend(new)

                if len(new) == 0:
                    print(f"No new results for '{term}' page {page}, moving on")
                    break

                api_calls += 1
                print(
                    f"[{api_calls}] '{term}' page {page}: {len(new)} new ({len(all_data)} total)"
                )

                time.sleep(1.5)

            except Exception as e:
                print(f"Error on '{term}' page {page}: {e}")
                time.sleep(5)

            # Save every 10 calls
            if api_calls % 10 == 0:
                safe_save_json(all_data, progress_file)
                print(f"Saved {len(all_data)} results")

        # Final save
        safe_save_json(all_data, progress_file)

    final_count = len(all_data)
    new_records = final_count - initial_count
    elapsed = (datetime.now() - start_time).total_seconds()

    has_salary = [
        job for job in all_data if job.get("salary_min") or job.get("salary_max")
    ]
    print(
        f"\nTotal: {len(all_data)} jobs, {len(has_salary)} with salary data ({len(has_salary)/len(all_data)*100:.1f}%)"
    )
    print(f"\nStarted with {initial_count} jobs, ended with {final_count} jobs!")

    mlflow.log_metrics(
        {
            "collect/initial_count": initial_count,
            "collect/final_count": final_count,
            "collect/new_records": new_records,
            "collect/elapsed_seconds": elapsed,
            "collect/pct_with_salary": (len(has_salary) / final_count * 100),
        }
    )


if __name__ == "__main__":
    typer.run(collect_data)
