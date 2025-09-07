import sys
import warnings
import time
import pandas as pd
from pathlib import Path

BASE_DIR = Path('/home/spandanjit2005/Documents/brfss-data')


def to_csv(xpt_path: Path, csv_path: Path):

    file_start_time = time.perf_counter()
    print(f"[INFO] Converting {xpt_path.name}")

    try:

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            df = pd.read_sas(xpt_path, encoding='latin1')

        df.to_csv(csv_path, index=False)

        duration = time.perf_counter() - file_start_time
        duration_minutes = int(duration) // 60
        duration_seconds = float(duration) % 60

        print(f"[SUCCESS] Converted {xpt_path.name} to {csv_path.name} in {duration_minutes} minutes and {duration_seconds:.2f} seconds.")
        return True

    except FileNotFoundError:
        print(f"[ERROR] Input file not found at {xpt_path}", file=sys.stderr)
        return False

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while processing {xpt_path.name}: {e}", file=sys.stderr)
        return False


def main(base_dir: Path = BASE_DIR):

    total_start_time = time.perf_counter()
    converted_count = 0

    subdirectories = sorted([
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])

    if not subdirectories:
        print("[INFO] No subdirectories found to process.")
        return

    for subdir in subdirectories:
        xpt_files = [p for p in subdir.glob('*') if p.suffix.lower() == '.xpt']

        if not xpt_files:
            print(f"[INFO] No .XPT files found in {subdir.name}")
            continue

        if len(xpt_files) > 1:
            print(f"[WARNING] Multiple .XPT files found. Processing only {xpt_files[0].name}", file=sys.stderr)

        xpt_file_path = xpt_files[0]
        dir_name = subdir.name
        csv_file_name = f"{dir_name}_BRFSS_RAW.csv"
        csv_file_path = subdir / csv_file_name

        if to_csv(xpt_file_path, csv_file_path):
            converted_count += 1

    total_duration = time.perf_counter() - total_start_time
    total_minutes = int(total_duration)// 60
    total_seconds = float(total_duration) % 60

    print(f"[SUCCESS] Converted {converted_count} files in {total_minutes} minutes and {total_seconds:.2f} seconds.")


if __name__ == "__main__":
    main()
