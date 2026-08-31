import torch
from train_utils import train_one_category

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Run with verbose=True to see per-epoch loss breakdown
state = train_one_category("mvtec_ad", "hazelnut", DEVICE, epochs=20, verbose=True)