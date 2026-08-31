import torch

from train_utils import train_one_category
from eval_utils import evaluate_one_category

DATASET_NAME = "mvtec_ad"
CATEGORY = "bottle"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20

LAMBDA_H_VALUES = [0.0, 0.01, 0.1]

results = []

for lam_h in LAMBDA_H_VALUES:
    print(f"\n=== Training with lambda_h={lam_h} ===")
    state = train_one_category(
        DATASET_NAME, CATEGORY, DEVICE,
        epochs=EPOCHS, lambda_h=lam_h, verbose=False
    )
    metrics = evaluate_one_category(DATASET_NAME, CATEGORY, DEVICE, state)
    print(f"  I-AUROC: {metrics['i_auroc']:.2f} | "
          f"P-AUROC: {metrics['p_auroc']:.2f} | "
          f"PRO: {metrics['pro']:.2f}")
    results.append((lam_h, metrics))

print("\n=== Summary ===")
print(f"{'lambda_h':<10} {'I-AUROC':<10} {'P-AUROC':<10} {'PRO':<10}")
for lam_h, m in results:
    print(f"{lam_h:<10} {m['i_auroc']:<10.2f} {m['p_auroc']:<10.2f} {m['pro']:<10.2f}")

print(f"\nReference (baseline, no entropy reg, from earlier manual run):")
print(f"{'---':<10} {'100.00':<10} {'97.05':<10} {'88.30':<10}")