import os
from torch.utils.data import Dataset
from PIL import Image
import torch

DATASET_CONFIGS = {
    "mvtec_ad": dict(
        root="data/mvtec_ad", good_dirname="good", mask_suffix="_mask",
        img_ext=".png", mask_ext=".png", gt_folder_matches_defect=True
    ),
    "mpdd": dict(
        root="data/mpdd", good_dirname="good", mask_suffix="_mask",
        img_ext=".png", mask_ext=".png", gt_folder_matches_defect=True
    ),
    "btad": dict(
        root="data/btad/BTech_Dataset_transformed", good_dirname="ok", mask_suffix="",
        img_ext=".bmp", mask_ext=".png", gt_folder_matches_defect=False,  # always "ok"
        gt_folder_name="ok"
    ),
    "wfdd": dict(
        root="data/wfdd/WFDD", good_dirname="good", mask_suffix="_mask",
        img_ext=".png", mask_ext=".png", gt_folder_matches_defect=True
    ),
}


class AnomalyDataset(Dataset):
    def __init__(self, dataset_name, category, split="train",
                 transform=None, mask_transform=None):
        cfg = DATASET_CONFIGS[dataset_name]
        self.transform = transform
        self.mask_transform = mask_transform
        root = cfg["root"]
        good = cfg["good_dirname"]
        mask_suffix = cfg["mask_suffix"]
        mask_ext = cfg["mask_ext"]

        self.image_paths, self.mask_paths, self.labels = [], [], []

        if split == "train":
            good_dir = os.path.join(root, category, "train", good)
            for f in sorted(os.listdir(good_dir)):
                self.image_paths.append(os.path.join(good_dir, f))
                self.mask_paths.append(None)
                self.labels.append(0)
        else:
            test_dir = os.path.join(root, category, "test")
            gt_dir = os.path.join(root, category, "ground_truth")
            for defect_type in sorted(os.listdir(test_dir)):
                defect_path = os.path.join(test_dir, defect_type)
                for f in sorted(os.listdir(defect_path)):
                    self.image_paths.append(os.path.join(defect_path, f))
                    if defect_type == good:
                        self.mask_paths.append(None)
                        self.labels.append(0)
                    else:
                        base = os.path.splitext(f)[0]
                        gt_subfolder = defect_type if cfg["gt_folder_matches_defect"] else cfg["gt_folder_name"]

                        # Try the configured mask extension first, then fall back
                        # to other common ones (handles inconsistent packaging,
                        # e.g. BTAD category 03 using .bmp masks while 01/02 use .png)
                        candidate_exts = [mask_ext, ".png", ".bmp", ".jpg"]
                        mask_path = None
                        for ext in candidate_exts:
                            candidate = os.path.join(gt_dir, gt_subfolder, f"{base}{mask_suffix}{ext}")
                            if os.path.exists(candidate):
                                mask_path = candidate
                                break

                        if mask_path is None:
                            # fall back to the originally configured path even if missing,
                            # so the error message downstream is clear about what's wrong
                            mask_path = os.path.join(gt_dir, gt_subfolder, f"{base}{mask_suffix}{mask_ext}")

                        self.mask_paths.append(mask_path)
                        self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]

        if self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            if self.mask_transform:
                mask = self.mask_transform(mask)
        else:
            mask = torch.zeros(1, img.shape[-2], img.shape[-1])

        return img, mask, label