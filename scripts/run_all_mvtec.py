import torch
import time
import csv
import os

from train_utils import train_one_category
from eval_utils import evaluate_one_category

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper"
]

DATASET_NAME = "mvtec_ad"
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_CSV = "results_mvtec_fpl_ad.csv"


def load_existing_results():
    if not os.path.exists(RESULTS_CSV):
        return {}
    results = {}
    with open(RESULTS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["category"]] = {
                "category": row["category"],
                "i_auroc": float(row["i_auroc"]),
                "p_auroc": float(row["p_auroc"]),
                "pro": float(row["pro"]),
            }
    return results


def save_results(results_dict):
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "i_auroc", "p_auroc", "pro"])
        writer.writeheader()
        for cat in MVTEC_CATEGORIES:
            if cat in results_dict:
                writer.writerow(results_dict[cat])


def main():
    results = load_existing_results()
    already_done = set(results.keys())
    if already_done:
        print(f"Resuming: {len(already_done)} categories already completed: {sorted(already_done)}")

    for i, category in enumerate(MVTEC_CATEGORIES):
        if category in already_done:
            print(f"\n=== [{i+1}/{len(MVTEC_CATEGORIES)}] {DATASET_NAME}/{category} — SKIPPED (already done) ===")
            continue

        print(f"\n=== [{i+1}/{len(MVTEC_CATEGORIES)}] {DATASET_NAME}/{category} ===")
        start = time.time()

        try:
            state = train_one_category(DATASET_NAME, category, DEVICE,
                                        epochs=EPOCHS, verbose=False)
            metrics = evaluate_one_category(DATASET_NAME, category, DEVICE, state)

            elapsed = time.time() - start
            print(f"  I-AUROC: {metrics['i_auroc']:.2f} | "
                  f"P-AUROC: {metrics['p_auroc']:.2f} | "
                  f"PRO: {metrics['pro']:.2f} | "
                  f"({elapsed:.1f}s)")

            results[category] = {
                "category": category,
                "i_auroc": metrics["i_auroc"],
                "p_auroc": metrics["p_auroc"],
                "pro": metrics["pro"],
            }

        except Exception as e:
            print(f"  FAILED: {e}")
            results[category] = {
                "category": category,
                "i_auroc": float("nan"),
                "p_auroc": float("nan"),
                "pro": float("nan"),
            }

        # Save after EVERY category — critical for resume capability
        save_results(results)
        print(f"  [saved progress to {RESULTS_CSV}]")

    valid = [r for r in results.values() if r["i_auroc"] == r["i_auroc"]]
    if valid:
        mean_i = sum(r["i_auroc"] for r in valid) / len(valid)
        mean_p = sum(r["p_auroc"] for r in valid) / len(valid)
        mean_pro = sum(r["pro"] for r in valid) / len(valid)
        print(f"\n=== MEAN across {len(valid)}/{len(MVTEC_CATEGORIES)} categories ===")
        print(f"I-AUROC: {mean_i:.2f}")
        print(f"P-AUROC: {mean_p:.2f}")
        print(f"PRO:     {mean_pro:.2f}")
        print(f"\nPaper's reported (Table 2): I-AUROC 99.6 / P-AUROC 98.3 / PRO 94.6")

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()