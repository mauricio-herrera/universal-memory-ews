from __future__ import annotations
from pathlib import Path
import requests
from tqdm.auto import tqdm

CHIRPS_MONTHLY_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"

def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024, timeout: int = 60) -> Path:
    """
    Stream-download a potentially large file with a progress bar.
    If the file exists and is >10MB, we assume it is already downloaded.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 10_000_000:
        print(f"[OK] Already exists: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
        return out_path

    print(f"[INFO] Downloading: {url}")
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for data in r.iter_content(chunk_size=chunk_size):
            if data:
                f.write(data)
                pbar.update(len(data))

    print(f"[OK] Download complete: {out_path}")
    return out_path
