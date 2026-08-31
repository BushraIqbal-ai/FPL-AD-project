from torchvision import transforms
from datasets.anomaly_dataset import AnomalyDataset, DATASET_CONFIGS
import os

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
mask_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

for ds_name, cfg in DATASET_CONFIGS.items():
    categories = sorted(os.listdir(cfg["root"]))
    categories = [c for c in categories if os.path.isdir(os.path.join(cfg["root"], c))]
    for cat in categories:
        try:
            train_ds = AnomalyDataset(ds_name, cat, split="train", transform=transform)
            test_ds = AnomalyDataset(ds_name, cat, split="test", transform=transform, mask_transform=mask_transform)
            # sanity check one sample loads
            img, mask, label = test_ds[len(test_ds)-1]
            print(f"{ds_name}/{cat}: train={len(train_ds)}, test={len(test_ds)}, sample_img={img.shape}, sample_mask={mask.shape}")
        except Exception as e:
            print(f"{ds_name}/{cat}: FAILED — {e}")


            