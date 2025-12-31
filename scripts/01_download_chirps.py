from pathlib import Path
import argparse
from memory_ews.chirps_download import download_file, CHIRPS_MONTHLY_URL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=CHIRPS_MONTHLY_URL)
    ap.add_argument("--out", default="data_raw/chirps_monthly.nc")
    args = ap.parse_args()
    download_file(args.url, Path(args.out))

if __name__ == "__main__":
    main()
