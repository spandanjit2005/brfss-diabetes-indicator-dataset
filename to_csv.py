import gc
import sys
import csv
import time

import pyreadstat
from pathlib import Path

BASE_DIR = Path('/home/spandanjit2005/Documents/brfss-data')


def to_csv(xpt_path: Path, csv_path: Path) -> bool:

    file_start_time = time.perf_counter()
    print(f"[INFO] Converting {xpt_path.name}")

    try:
        df, _ = pyreadstat.read_xport(str(xpt_path), encoding='latin1')

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:

            writer = csv.writer(f)
            writer.writerow(df.columns.tolist())
            writer.writerows(df.itertuples(index=False))

        duration_seconds = time.perf_counter() - file_start_time

        print(f"[SUCCESS] Converted {xpt_path.name} to {csv_path.name} in {duration_seconds:.2f} seconds.")
        gc.collect()
        return True

    except FileNotFoundError:
        print(f"[ERROR] Input file not found at {xpt_path}", file=sys.stderr)
        gc.collect()
        return False

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while processing {xpt_path.name}: {e}", file=sys.stderr)
        gc.collect()
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
        csv_file_path = subdir / f"{subdir.name}_BRFSS_RAW.csv"

        if to_csv(xpt_file_path, csv_file_path):
            converted_count += 1

    total_duration = time.perf_counter() - total_start_time
    total_minutes = int(total_duration) // 60
    total_seconds = float(total_duration) % 60

    print(f"[SUCCESS] Converted {converted_count} files in {total_minutes} minutes and {total_seconds:.2f} seconds.")


if __name__ == "__main__":
    main()