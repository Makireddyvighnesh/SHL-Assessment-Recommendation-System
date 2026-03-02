"""
Generate predictions CSV for the unlabeled test set from Excel input.

Input:
    .xlsx file with:
        Sheet name: "Test-Set"
        Column name: "Query"

Output format:
    query, Assessment_url
    Query 1, https://...
    Query 1, https://...
    Query 2, https://...

Usage:
    python predictions/generate_predictions.py \
        --test  predictions/test_queries.xlsx \
        --api   http://localhost:8000 \
        --out   predictions/test_predictions.csv
"""

import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import requests


# ----------------------------
# Load queries from Excel
# ----------------------------
def load_test_queries(path: str) -> list[str]:
    """Load test queries from Excel file (.xlsx)."""
    try:
        df = pd.read_excel(
            path,
            sheet_name="Test-Set",
            engine="openpyxl"
        )

        if "Query" not in df.columns:
            raise ValueError("Column 'Query' not found in sheet 'Test-Set'")

        queries = (
            df["Query"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        return queries

    except Exception as e:
        print(f"ERROR loading Excel file: {e}")
        return []


# ----------------------------
# Call recommendation API
# ----------------------------
def get_recommendations(api_base: str, query: str) -> list[str]:
    """Call API and return list of recommended URLs."""
    try:
        response = requests.post(
            f"{api_base}/recommend",
            json={"query": query},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        return [
            item["url"]
            for item in data.get("recommended_assessments", [])
            if "url" in item
        ]

    except Exception as e:
        print(f"  ERROR for query: {e}")
        return []


# ----------------------------
# Generate predictions CSV
# ----------------------------
def generate_predictions(
    test_path: str,
    api_base: str,
    output_path: str,
    delay: float = 1.0,
):
    """Generate predictions and write CSV."""
    queries = load_test_queries(test_path)

    if not queries:
        print("No queries found. Exiting.")
        return

    print(f"Generating predictions for {len(queries)} test queries...")
    print(f"API: {api_base}")
    print(f"Output: {output_path}\n")

    rows = []

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query[:80]}...")
        urls = get_recommendations(api_base, query)
        print(f"  → {len(urls)} recommendations")

        for url in urls:
            rows.append({
                "query": query,
                "Assessment_url": url
            })

        time.sleep(delay)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "Assessment_url"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Predictions saved to {output_path}")
    print(f"   Total rows: {len(rows)}")
    print(f"   Queries: {len(queries)}")

    avg = len(rows) / len(queries) if len(queries) > 0 else 0
    print(f"   Avg recommendations per query: {avg:.2f}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate test set predictions (Excel input)")
    parser.add_argument(
        "--test",
        default="./Gen_AI Dataset.xlsx",
        help="Path to test queries Excel file (.xlsx)",
    )
    parser.add_argument(
        "--api",
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--out",
        default="predictions/test_predictions.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls (seconds)",
    )

    args = parser.parse_args()

    if not Path(args.test).exists():
        print(f"ERROR: Test file not found at {args.test}")
        print("Expected Excel file with:")
        print("  - Sheet name: Test-Set")
        print("  - Column name: Query")
        return

    generate_predictions(
        args.test,
        args.api,
        args.out,
        delay=args.delay
    )


if __name__ == "__main__":
    main()