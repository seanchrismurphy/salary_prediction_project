import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def find_project_root(marker="README.md"):
    p = Path.cwd()
    while p != p.parent:
        if (p / marker).exists():
            return p
        p = p.parent
    raise RuntimeError("Project root not found")

def safe_save_csv(df: pd.DataFrame, path: Path):
    """Write to a temp file then atomically replace the target."""
    path = Path(path)
    dir_ = path.parent
    with tempfile.NamedTemporaryFile('w', dir=dir_, delete=False, suffix='.tmp') as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    os.replace(tmp_path, path)
    
def safe_save_json(data, path: Path):
    path = Path(path)
    dir_ = path.parent
    with tempfile.NamedTemporaryFile('w', dir=dir_, delete=False, suffix='.tmp') as tmp:
        json.dump(data, tmp)
        tmp_path = tmp.name
    os.replace(tmp_path, path)