import torch
from train_utils import train_one_category
from eval_utils import evaluate_one_category

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = "mpdd"
CATEGORY = "bracket_black"

print("=== Run 1 ===")
state1 = train_one_category(DATASET_NAME, CATEGORY, DEVICE, epochs=20, verbose=False)
metrics1 = evaluate_one_category(DATASET_NAME, CATEGORY, DEVICE, state1)
print(f"Run 1: I-AUROC={metrics1['i_auroc']:.4f} P-AUROC={metrics1['p_auroc']:.4f} PRO={metrics1['pro']:.4f}")

print("\n=== Run 2 (same everything) ===")
state2 = train_one_category(DATASET_NAME, CATEGORY, DEVICE, epochs=20, verbose=False)
metrics2 = evaluate_one_category(DATASET_NAME, CATEGORY, DEVICE, state2)
print(f"Run 2: I-AUROC={metrics2['i_auroc']:.4f} P-AUROC={metrics2['p_auroc']:.4f} PRO={metrics2['pro']:.4f}")

print("\n=== Difference ===")
print(f"I-AUROC diff: {abs(metrics1['i_auroc'] - metrics2['i_auroc']):.6f}")
print(f"P-AUROC diff: {abs(metrics1['p_auroc'] - metrics2['p_auroc']):.6f}")
print(f"PRO diff:     {abs(metrics1['pro'] - metrics2['pro']):.6f}")