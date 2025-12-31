from pathlib import Path
import argparse
from memory_ews.config import PipelineConfig
from memory_ews.pipeline import run_all_regions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True, help="Path to CHIRPS monthly netCDF")
    ap.add_argument("--out", default="outputs_memory_ews", help="Output directory")
    args = ap.parse_args()

    cfg = PipelineConfig().with_default_regions()
    summary = run_all_regions(Path(args.nc), Path(args.out), cfg)
    print("\n[OK] Summary saved. Head:\n", summary.head())

if __name__ == "__main__":
    main()
