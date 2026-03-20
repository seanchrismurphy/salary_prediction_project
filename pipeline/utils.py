import io
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

STORAGE_ACCOUNT = "salaryprdata"
CONTAINER = "pipeline-data"

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

def _blob_client(blob_name: str):
    credential = DefaultAzureCredential()
    url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
    service = BlobServiceClient(url, credential=credential)
    return service.get_blob_client(container=CONTAINER, blob=blob_name)

def blob_exists(blob_name: str) -> bool:
    return _blob_client(blob_name).exists()

# --- JSON ---
def save_json_to_blob(data, blob_name: str) -> None:
    _blob_client(blob_name).upload_blob(
        json.dumps(data, ensure_ascii=False),
        overwrite=True,
    )

def load_json_from_blob(blob_name: str):
    return json.loads(_blob_client(blob_name).download_blob().readall())

# --- Parquet ---

def save_parquet_to_blob(df: pd.DataFrame, blob_name: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression='gzip')
    buf.seek(0)
    
    # Implement manual chunking because we were having trouble getting inbuilt azure chunking to work
    client = _blob_client(blob_name)
    chunk_size = 1024 * 1024  # 1MB
    block_list = []
    block_id = 0
    
    while True:
        chunk = buf.read(chunk_size)
        if not chunk:
            break
        bid = f"{block_id:08d}"
        client.stage_block(bid, chunk)
        block_list.append(bid)
        block_id += 1
    
    client.commit_block_list(block_list)
    
def load_parquet_from_blob(blob_name: str) -> pd.DataFrame:
    buf = io.BytesIO(_blob_client(blob_name).download_blob().readall())
    return pd.read_parquet(buf)