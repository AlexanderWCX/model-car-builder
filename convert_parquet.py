import os
import glob
import pandas as pd

input_dir = os.environ.get("INPUT_DIR", "/datasets")
output_dir = os.environ.get("OUTPUT_DIR", "/datasets")
remove_parquet = os.environ.get("REMOVE_PARQUET", "1") == "1"

# Find all parquet files recursively
parquet_files = sorted(glob.glob(os.path.join(input_dir, "**/*.parquet"), recursive=True))

if not parquet_files:
    print("No parquet files found. Nothing to convert.")
    exit(0)

print(f"Found {len(parquet_files)} parquet files to convert.")

for pq_file in parquet_files:
    rel_path = os.path.relpath(pq_file, input_dir)
    jsonl_name = os.path.splitext(rel_path)[0] + ".jsonl"
    jsonl_path = os.path.join(output_dir, jsonl_name)

    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    print(f"  Converting: {rel_path} -> {jsonl_name}")
    df = pd.read_parquet(pq_file)
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    if remove_parquet:
        os.remove(pq_file)

print(f"Converted {len(parquet_files)} files to JSONL.")