import numpy as np
from skimage import measure

# Handle both old and new NumPy versions
try:
    trapz_fn = np.trapezoid
except AttributeError:
    trapz_fn = np.trapz


def compute_pro(masks, amaps, num_thresholds=200):
    """
    masks: numpy array [N, H, W], binary ground truth (0/1)
    amaps: numpy array [N, H, W], anomaly score maps (float)
    Returns: PRO score (float, 0-1 range, multiply by 100 for %)
    """
    thresholds = np.linspace(amaps.min(), amaps.max(), num_thresholds)
    pro_values = []
    fpr_values = []

    for th in thresholds:
        binary_amaps = (amaps >= th).astype(np.uint8)

        pros = []
        for i in range(len(masks)):
            mask = masks[i]
            if mask.sum() == 0:
                continue  # skip images with no defect (used for FPR only)
            labeled = measure.label(mask)
            for region_label in np.unique(labeled):
                if region_label == 0:
                    continue
                region_mask = (labeled == region_label)
                overlap = np.logical_and(region_mask, binary_amaps[i]).sum()
                pros.append(overlap / region_mask.sum())

        pro_values.append(np.mean(pros) if pros else 0.0)

        # false positive rate on normal (mask-free) regions
        fp = 0
        total_normal_pixels = 0
        for i in range(len(masks)):
            mask = masks[i]
            normal_region = (mask == 0)
            fp += np.logical_and(normal_region, binary_amaps[i]).sum()
            total_normal_pixels += normal_region.sum()
        fpr_values.append(fp / total_normal_pixels if total_normal_pixels > 0 else 0.0)

    # integrate PRO over FPR in [0, 0.3] (standard convention in anomaly detection literature)
    fpr_values = np.array(fpr_values)
    pro_values = np.array(pro_values)
    valid = fpr_values <= 0.3
    if valid.sum() < 2:
        return 0.0

    fpr_valid = fpr_values[valid]
    pro_valid = pro_values[valid]
    sort_idx = np.argsort(fpr_valid)
    fpr_valid = fpr_valid[sort_idx]
    pro_valid = pro_valid[sort_idx]

    pro_auc = trapz_fn(pro_valid, fpr_valid) / fpr_valid.max() if fpr_valid.max() > 0 else 0.0
    return pro_auc