"""
Evaluation Script — Mean Recall@10
Reads train data from Excel file (Train-Set sheet)

Usage:
    python evaluation/evaluate.py --data dataset/your_file.xlsx --api http://localhost:8000
"""

import argparse
import json
import time
from pathlib import Path

import requests
import pandas as pd


def normalize_url(url: str) -> str:
    """
    Normalize SHL URLs to a common format for comparison.
    Strips /solutions/ prefix variation so both formats match.
    e.g.
      https://www.shl.com/solutions/products/product-catalog/view/X/
      https://www.shl.com/products/product-catalog/view/X/
    Both → 'product-catalog/view/X'
    """
    url = url.lower().rstrip("/")
    # Extract just the slug after 'view/'
    if "view/" in url:
        return "view/" + url.split("view/")[-1]
    return url


def recall_at_k(recommended_urls: list[str], relevant_urls: list[str], k: int = 10) -> float:
    if not relevant_urls:
        return 0.0
    top_k    = {normalize_url(u) for u in recommended_urls[:k]}
    relevant = {normalize_url(u) for u in relevant_urls}
    return len(top_k & relevant) / len(relevant)

def mean_recall_at_k(results: list[dict], k: int = 10) -> float:
    if not results:
        return 0.0
    return sum(r["recall@k"] for r in results) / len(results)


def load_train_data(excel_path: str, sheet_name: str = "Train-Set") -> list[dict]:
    """
    Load train data from Excel.
    Format: LONG format — one row per (query, assessment_url) pair.
    Groups by query to collect all relevant URLs.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    print(f"Sheet         : {sheet_name}")
    print(f"Columns       : {df.columns.tolist()}")
    print(f"Total rows    : {len(df)}")
    print(df.head(3))
    print()

    # Auto-detect query and URL columns
    query_col = None
    url_col   = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if "query" in col_lower:
            query_col = col
        if "url" in col_lower or "assessment" in col_lower:
            url_col = col

    if not query_col:
        raise ValueError(f"No query column found. Columns: {df.columns.tolist()}")
    if not url_col:
        raise ValueError(f"No URL column found. Columns: {df.columns.tolist()}")

    print(f"Query column  : {query_col}")
    print(f"URL column    : {url_col}")
    print()

    # Drop rows where query or url is empty
    df = df[[query_col, url_col]].dropna(subset=[query_col, url_col])
    df[query_col] = df[query_col].astype(str).str.strip()
    df[url_col]   = df[url_col].astype(str).str.strip()
    df = df[df[query_col] != "nan"]
    df = df[df[url_col].str.startswith("http")]

    # Group by query → collect all relevant URLs
    grouped = (
        df.groupby(query_col, sort=False)[url_col]
        .apply(list)
        .reset_index()
    )
    grouped.columns = ["query", "Assessment_url"]

    train_data = grouped.to_dict(orient="records")

    print(f"Unique queries : {len(train_data)}")
    print(f"Avg URLs/query : {sum(len(d['Assessment_url']) for d in train_data) / len(train_data):.1f}")
    print()

    # Preview first 2
    for i, d in enumerate(train_data[:2]):
        print(f"  Query {i+1}: {d['query'][:80]}...")
        print(f"  URLs  ({len(d['Assessment_url'])}): {d['Assessment_url']}")
        print()

    return train_data


def evaluate(excel_path: str, api_base: str, k: int = 10, delay: float = 1.5):
    train_data = load_train_data(excel_path)

    print(f"Evaluating {len(train_data)} queries against {api_base}")
    print(f"Metric: Mean Recall@{k}")
    print("=" * 70)

    results = []
    for i, item in enumerate(train_data, 1):
        query         = item["query"]
        relevant_urls = item["Assessment_url"]

        print(f"\n[{i}/{len(train_data)}] {query[:80]}...")

        try:
            response = requests.post(
                f"{api_base}/recommend",
                json={"query": query},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            recommended_urls = [r["url"] for r in data.get("recommended_assessments", [])]
        except Exception as e:
            print(f"  ERROR: {e}")
            recommended_urls = []

        r_at_k = recall_at_k(recommended_urls, relevant_urls, k)
        results.append({
            "query":             query,
            "relevant_count":    len(relevant_urls),
            "recommended_count": len(recommended_urls),
            "recall@k":          r_at_k,
            "recommended_urls":  recommended_urls,
            "relevant_urls":     relevant_urls,
        })

        print(f"  Relevant: {len(relevant_urls)} | Recommended: {len(recommended_urls)} | Recall@{k}: {r_at_k:.3f}")
        found  = set(recommended_urls[:k]) & set(relevant_urls)
        missed = set(relevant_urls) - set(recommended_urls[:k])
        if found:  print(f"  ✅ Found : {[u.split('/')[-2] for u in found]}")
        if missed: print(f"  ❌ Missed: {[u.split('/')[-2] for u in list(missed)[:5]]}")

        time.sleep(delay)

    print("\n" + "=" * 70)
    mean_r = mean_recall_at_k(results, k)
    print(f"\nMean Recall@{k}: {mean_r:.4f}")
    print("=" * 70)

    # Per-query table
    print(f"\n{'Query':<60} {'Recall@'+str(k):>10}")
    print("-" * 72)
    for r in results:
        q = r["query"][:57] + "..." if len(r["query"]) > 57 else r["query"]
        print(f"{q:<60} {r['recall@k']:>10.3f}")

    # Save results
    output = {
        "mean_recall_at_k": mean_r,
        "k":                k,
        "num_queries":      len(results),
        "per_query":        results,
    }
    out_path = Path("evaluation/eval_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")
    return mean_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="dataset/data.xlsx",    help="Path to Excel file")
    parser.add_argument("--api",   default="http://localhost:8000", help="API base URL")
    parser.add_argument("--k",     type=int,   default=10,         help="K for Recall@K")
    parser.add_argument("--delay", type=float, default=1.5,        help="Delay between API calls (seconds)")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"ERROR: File not found at {args.data}")
        return

    evaluate(args.data, args.api, k=args.k, delay=args.delay)


if __name__ == "__main__":
    main()
