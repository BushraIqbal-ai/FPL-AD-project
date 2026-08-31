import torch
import time
import csv
import os

from train_utils import train_one_category
from eval_utils import evaluate_one_category

VALIDATION_SET = [
    ("mvtec_ad", "bottle"),
    ("mvtec_ad", "grid"),
    ("mvtec_ad", "screw"),
    ("mpdd", "bracket_black"),
    ("wfdd", "pink_flower"),
]

TOPK_VALUES = [5, 10, 20, 50, 100]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_CSV = "tune_topk_results.csv"


def load_existing_results():
    if not os.path.exists(RESULTS_CSV):
        return {}
    results = {}
    with open(RESULTS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["category"], int(row["top_k"]))
            results[key] = {
                "dataset": row["dataset"],
                "category": row["category"],
                "top_k": int(row["top_k"]),
                "i_auroc": float(row["i_auroc"]),
                "p_auroc": float(row["p_auroc"]),
                "pro": float(row["pro"]),
            }
    return results


def save_results(results_dict):
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "category", "top_k",
                                                "i_auroc", "p_auroc", "pro"])
        writer.writeheader()
        for key in results_dict:
            writer.writerow(results_dict[key])


def main():
    results = load_existing_results()

    for dataset_name, category in VALIDATION_SET:
        # Check if we already have all top_k values for this category
        already_done = all((dataset_name, category, k) in results for k in TOPK_VALUES)
        if already_done:
            print(f"[SKIP] {dataset_name}/{category} (all top_k values already tested)")
            continue

        print(f"\n=== Training {dataset_name}/{category} (epochs=20, one-time) ===")
        start = time.time()
        state = train_one_category(dataset_name, category, DEVICE, epochs=20, verbose=False)
        train_time = time.time() - start
        print(f"  Training done ({train_time:.1f}s). Now evaluating at each top_k...")

        for top_k in TOPK_VALUES:
            key = (dataset_name, category, top_k)
            if key in results:
                print(f"  [SKIP] top_k={top_k} (already done)")
                continue

            metrics = evaluate_one_category(dataset_name, category, DEVICE, state, top_k=top_k)
            print(f"  top_k={top_k:<4} -> I-AUROC: {metrics['i_auroc']:.2f} | "
                  f"P-AUROC: {metrics['p_auroc']:.2f} | PRO: {metrics['pro']:.2f}")

            results[key] = {
                "dataset": dataset_name,
                "category": category,
                "top_k": top_k,
                "i_auroc": metrics["i_auroc"],
                "p_auroc": metrics["p_auroc"],
                "pro": metrics["pro"],
            }
            save_results(results)

    # ---- Print comparison table ----
    print("\n\n=== SUMMARY: Top-k Comparison ===")
    print(f"{'Dataset/Category':<28} {'Top-k':<8} {'I-AUROC':<10} {'P-AUROC':<10} {'PRO':<10}")
    for dataset_name, category in VALIDATION_SET:
        for top_k in TOPK_VALUES:
            key = (dataset_name, category, top_k)
            if key in results:
                r = results[key]
                print(f"{dataset_name+'/'+category:<28} {top_k:<8} "
                      f"{r['i_auroc']:<10.2f} {r['p_auroc']:<10.2f} {r['pro']:<10.2f}")
        print()


if __name__ == "__main__":
    main()