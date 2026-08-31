import torch
import time
import csv
import os

from train_utils import train_one_category
from eval_utils import evaluate_one_category

# ---- Validation set: one representative category per difficulty tier ----
VALIDATION_SET = [
    ("mvtec_ad", "bottle"),        # easy baseline
    ("mvtec_ad", "grid"),          # worst MVTec outlier
    ("mvtec_ad", "screw"),         # second worst MVTec outlier
    ("mpdd", "bracket_black"),     # worst MPDD outlier
    ("wfdd", "pink_flower"),       # worst WFDD outlier, root cause diagnosed
]

EPOCHS_TO_TEST = [20, 50, 100]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_CSV = "tune_epochs_results.csv"


def load_existing_results():
    if not os.path.exists(RESULTS_CSV):
        return {}
    results = {}
    with open(RESULTS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["category"], int(row["epochs"]))
            results[key] = {
                "dataset": row["dataset"],
                "category": row["category"],
                "epochs": int(row["epochs"]),
                "i_auroc": float(row["i_auroc"]),
                "p_auroc": float(row["p_auroc"]),
                "pro": float(row["pro"]),
                "time_s": float(row["time_s"]),
            }
    return results


def save_results(results_dict):
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "category", "epochs",
                                                "i_auroc", "p_auroc", "pro", "time_s"])
        writer.writeheader()
        for key in results_dict:
            writer.writerow(results_dict[key])


def main():
    results = load_existing_results()

    for dataset_name, category in VALIDATION_SET:
        for epochs in EPOCHS_TO_TEST:
            key = (dataset_name, category, epochs)
            if key in results:
                print(f"[SKIP] {dataset_name}/{category} @ epochs={epochs} (already done)")
                continue

            print(f"\n=== {dataset_name}/{category} @ epochs={epochs} ===")
            start = time.time()

            try:
                state = train_one_category(dataset_name, category, DEVICE,
                                            epochs=epochs, verbose=False)
                metrics = evaluate_one_category(dataset_name, category, DEVICE, state)
                elapsed = time.time() - start

                print(f"  I-AUROC: {metrics['i_auroc']:.2f} | "
                      f"P-AUROC: {metrics['p_auroc']:.2f} | "
                      f"PRO: {metrics['pro']:.2f} | ({elapsed:.1f}s)")

                results[key] = {
                    "dataset": dataset_name,
                    "category": category,
                    "epochs": epochs,
                    "i_auroc": metrics["i_auroc"],
                    "p_auroc": metrics["p_auroc"],
                    "pro": metrics["pro"],
                    "time_s": elapsed,
                }

            except Exception as e:
                print(f"  FAILED: {e}")
                results[key] = {
                    "dataset": dataset_name, "category": category, "epochs": epochs,
                    "i_auroc": float("nan"), "p_auroc": float("nan"),
                    "pro": float("nan"), "time_s": 0.0,
                }

            save_results(results)

    # ---- Print comparison table ----
    print("\n\n=== SUMMARY: Epochs Comparison ===")
    print(f"{'Dataset/Category':<28} {'Epochs':<8} {'I-AUROC':<10} {'P-AUROC':<10} {'PRO':<10}")
    for dataset_name, category in VALIDATION_SET:
        for epochs in EPOCHS_TO_TEST:
            key = (dataset_name, category, epochs)
            if key in results:
                r = results[key]
                print(f"{dataset_name+'/'+category:<28} {epochs:<8} "
                      f"{r['i_auroc']:<10.2f} {r['p_auroc']:<10.2f} {r['pro']:<10.2f}")
        print()


if __name__ == "__main__":
    main()