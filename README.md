# CLARA-SaGC-MM
[Under-review] "Read Between the Lines: Multimodal Inverse Planning for Evaluating Consistency and Reliability in Robot Task Planning" dataset

### Original Dataset
https://github.com/jeongeun980906/CLARA-Dataset

## Data Usage Notice

This repository does **not** contain or redistribute any dataset files.

It only provides code and scripts to reproduce the postprocessing pipeline used in the paper.  
Users must obtain all required input files directly from the original source and are responsible for complying with the applicable usage terms.

---
# Generation Pipeline

Generate augmented datasets with image metadata from `agument.json` (obtain from the official CLARA source).

## Minimum Required Files

| File | Role |
|------|------|
| `agument.json` | Original dataset (obtain from CLARA) |
| `scene_groups_with_goal_label.py` | Builds scene groups from `agument.json` |
| `json_parsing.py` | Helper for `qwen_image.py` to load scene groups |
| `qwen_image.py` | Generates scene images via Qwen text-to-image |
| `postprocessing_agument.py` | Produces the 6 final JSON outputs |

## Execution Order

```bash
# 1. Build scene groups from agument.json
python scene_groups_with_goal_label.py

# 2. Generate images (requires GPU, torch, diffusers)
python qwen_image.py

# 3. Produce augmented datasets
python postprocessing_agument.py
```

## Outputs

- `agument_with_img_unique.json`
- `agument_with_img_unique_cal.json`
- `agument_with_img_unique_val.json`
- `agument_with_img_unique_subset_180.json`
- `agument_with_img_unique_subset_180_cal.json`
- `agument_with_img_unique_subset_180_val.json`



### Citation
```
@article{park2023clara,
  title={Clara: classifying and disambiguating user commands for reliable interactive robotic agents},
  author={Park, Jeongeun and Lim, Seungwon and Lee, Joonhyung and Park, Sangbeom and Chang, Minsuk and Yu, Youngjae and Choi, Sungjoon},
  journal={IEEE Robotics and Automation Letters},
  volume={9},
  number={2},
  pages={1059--1066},
  year={2023},
  publisher={IEEE}
}
```

This project builds upon the CLARA dataset for academic research purposes.
