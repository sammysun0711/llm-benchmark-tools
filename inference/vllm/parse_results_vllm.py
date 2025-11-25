#!/usr/bin/env python3
import json
import os
import re
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Extract benchmark metrics from JSON files in a directory")
    parser.add_argument("--input_dir", required=True, help="Directory with JSON files")
    parser.add_argument("--output_file", required=True, help="Path to save Excel or CSV file")
    args = parser.parse_args()

    records = []

    # Regex to extract isl, osl, c values regardless of prefix/suffix
    pattern = re.compile(r"isl(\d+)_osl(\d+)_c(\d+)")

    file_paths = []
    for fname in os.listdir(args.input_dir):
        # Only process JSON files
        if fname.endswith(".json"):
            match = pattern.search(fname)
            if match:
                file_paths.append(os.path.join(args.input_dir, fname))

    # Sort by isl, osl, then c ascending
    sorted_files = sorted(
            file_paths,
            key=lambda path: tuple(map(int, pattern.search(os.path.basename(path)).groups()))
    )
    # Print sorted results
    for file in sorted_files:
        print(file)
    for filename in sorted_files:
        if filename.lower().endswith(".json"):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"❌ Skipping {filename}: {e}")
                continue

            record = {
                "File": os.path.basename(filename),
                "Max concurrency": data.get("max_concurrency"),
                "Request throughput (req/s)": data.get("request_throughput"),
                "Output token throughput (tok/s)": data.get("output_throughput"),
                "Total token throughput (tok/s)": data.get("total_token_throughput"),
                "Mean TTFT (ms)": data.get("mean_ttft_ms"),
                "Median TTFT (ms)": data.get("median_ttft_ms"),
                "P95 TTFT (ms)": data.get("p95_ttft_ms"),
                "P99 TTFT (ms)": data.get("p99_ttft_ms"),
                "Mean TPOT (ms)": data.get("mean_tpot_ms"),
                "Median TPOT (ms)": data.get("median_tpot_ms"),
                "P99 TPOT (ms/token)": data.get("p99_tpot_ms"),
                "P95 TPOT (ms/token)": data.get("p95_tpot_ms"),
                "Mean ITL (ms/token)": data.get("mean_itl_ms"),
                "Median ITL (ms/token)": data.get("median_itl_ms"),
                "P95 ITL (ms/token)": data.get("p95_itl_ms"),
                "P99 ITL (ms/token)": data.get("p99_itl_ms"),
                "Mean E2E latency (s)": (data.get("mean_e2el_ms") or 0) / 1000.0 if "mean_e2el_ms" in data else None,
                "Median E2E latency (s)": (data.get("median_e2el_ms") or 0) / 1000.0 if "median_e2el_ms" in data else None,
            }
            records.append(record)

    df = pd.DataFrame(records)

    column_order = [
        "File",
        "Max concurrency",
        "Request throughput (req/s)",
        "Output token throughput (tok/s)",
        "Total token throughput (tok/s)",
        "Mean TTFT (ms)",
        "Median TTFT (ms)",
        "P95 TTFT (ms)",
        "P99 TTFT (ms)",
        "Mean TPOT (ms)",
        "Median TPOT (ms)",
        "P95 TPOT (ms/token)",
        "P99 TPOT (ms/token)",
        "Mean ITL (ms/token)",
        "Median ITL (ms/token)",
        "P95 ITL (ms/token)",
        "P99 ITL (ms/token)",
        "Mean E2E latency (s)",
        "Median E2E latency (s)",
    ]
    for col in column_order:
        if col not in df.columns:
            df[col] = None

    df = df[column_order]

    if args.output_file.lower().endswith(".xlsx"):
        df.to_excel(args.output_file, index=False)
    else:
        df.to_csv(args.output_file, index=False)

    print(f"✅ Extracted {len(records)} records from {args.input_dir} -> {args.output_file}")

if __name__ == "__main__":
    main()
