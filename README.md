# FPL-AD Reimplementation

Independent reimplementation of FPL-AD, an image anomaly detection framework combining Multi-Factor Gated Fusion (MFGF) and Boundary-Aware Pseudo-Anomaly Synthesis (BAPS) on top of the SimpleNet feature-synthesis paradigm. Built from the original paper's equations and architecture description, without access to the authors' source code.

## Overview

The study is organized into three experiments:

- **Experiment 1** — Module-verified implementation, benchmarked across four public datasets (MVTec AD, MPDD, BTAD, WFDD; 28 categories total).
- **Experiment 2** — Hyperparameter investigation that uncovers and fixes a non-deterministic GPU training defect, producing corrected results.
- **Experiment 3** — Root-cause diagnosis via loss-curve inspection and heatmap visualization, identifying a shortcut-learning failure mode in structurally regular categories.

## Key Findings

- Complete, tensor-shape-verified reimplementation of every FPL-AD component.
- Discovery and correction of a non-deterministic GPU training defect (`cudnn` algorithm auto-selection), verified via bit-for-bit identical repeat runs.
- Identification of a shortcut-learning failure mode: on structurally regular categories, the discriminator learns geometric structure instead of defect-specific features.
- Full benchmark results across all 28 categories, compared against the original paper.

## Installation

```bash
git clone https://github.com/BushraIqbal-ai/FPL-AD-project.git
cd FPL-AD-project
pip install -r requirements.txt
```

**Requirements:** Python 3.x, PyTorch (CUDA), torchvision, numpy, opencv-python, scikit-learn

## Datasets

| Dataset | Categories | Link |
|---|---|---|
| MVTec AD | 15 | https://www.mvtec.com/company/research/datasets/mvtec-ad |
| MPDD | 6 | https://github.com/stepanje/MPDD |
| BTAD | 3 | https://avires.dimi.uniud.it/papers/btad/btad.zip |
| WFDD | 4 | https://github.com/cqylunlun/GLASS |

## Usage

The pipeline follows three stages, consistent with the study's methodology:

1. **Module verification** — Each component (backbone, adapter, MFGF, BAPS, discriminator) is unit-tested on synthetic tensors to confirm correct output shapes before assembly.
2. **Training and evaluation** — The full model is trained per category under the one-class protocol (normal images only), then evaluated on the corresponding test split using I-AUROC, P-AUROC, and PRO.
3. **Deterministic reproduction** — All training runs use a fixed seed (42) with deterministic cuDNN settings enabled, ensuring identical results across repeated runs.

Diagnostic analysis (loss-curve inspection and heatmap visualization) is applied to categories showing degraded performance, to distinguish training instability from shortcut-learning failure.

## Reproducibility

Fixed random seed: 42. Deterministic training enforced via:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

with a seeded `DataLoader` generator for reproducible batch order.

## Results Summary

| Dataset | I-AUROC (Ours) | P-AUROC (Ours) | PRO (Ours) | Paper |
|---|---|---|---|---|
| MVTec AD | 88.99 | 86.45 | 69.69 | 99.6 / 98.3 / 94.6 |
| MPDD | 87.17 | 82.52 | 68.23 | 98.8 / 99.0 / 95.4 |
| BTAD | 73.10 | 66.39 | — | 96.0 / 97.9 / — |
| WFDD | 88.11 | 92.80 | — | 99.3 / 98.6 / — |

## Limitations

- Several hyperparameters are estimated due to undisclosed values in the source paper.
- Results reflect a single fixed seed.
- Shortcut-learning diagnosis covers five collapse cases; other categories may exhibit milder, undetected versions of the same failure mode.

## Citation

```bibtex
@article{huang2026fplad,
  title   = {FPL-AD: Fine-grained perception and localization for image anomaly detection},
  author  = {Huang, L. and Zhang, H. and Zheng, H. and Yang, M. and Pan, L.},
  journal = {Knowledge-Based Systems},
  volume  = {351},
  pages   = {116602},
  year    = {2026}
}
```


 **Full Report:** [FPL-AD_IEEE_Style_Report](https://docs.google.com/document/d/18kthUH7qKItxUkuFYzQ6jJxQ1ysLWhuv/edit?usp=sharing&ouid=117639944094007826260&rtpof=true&sd=true)
