# CLARA-SaGC-MM
[Under-review] "Read Between the Lines: Multimodal Inverse Planning for Evaluating Consistency and Reliability in Robot Task Planning" dataset

### Original Dataset
https://github.com/jeongeun980906/CLARA-Dataset

## Data Usage Notice

This repository does **not** contain or redistribute any dataset files.

It only provides code and scripts to reproduce the postprocessing pipeline used in the paper.  
Users must obtain all required input files directly from the original source and are responsible for complying with the applicable usage terms.

---

## Reproducibility (Postprocessing Pipeline)

### Required inputs (to be obtained by the user)

Place the following files in:
src/MLIP/mlip/data/
- `agument.json`  
- `scene_groups_with_goal_label_with_img.json`

If you intend to regenerate synthesized scene images, additional scene/goal metadata
required by `qwen_image.py` must also be provided (see comments in the script).

---

### (Optional) Generate synthesized scene images

To generate scene images from text prompts and update the image mapping JSON, run:

```bash
python src/MLIP/mlip/data/qwen_image.py
```

Generated images should be saved under a user-specified directory (e.g., qwen2image_v*/).
This step can be skipped if the image mapping JSON and images are already available.


### Run postprocessing

Run the following script to reproduce the postprocessing used in the paper:
```bash
python src/MLIP/mlip/data/postprocessing_agument.py
```

This script performs the following operations:
injects group_id and img_path into entries,
removes duplicate entries per (group_id, goal),
creates calibration and validation splits,
and generates subset JSON files used for evaluation.


### Outputs

The following files are generated under src/MLIP/mlip/data/:
- agument_with_img_unique.json
- agument_with_img_unique_cal.json
- agument_with_img_unique_val.json
- agument_with_img_unique_subset_180.json
- agument_with_img_unique_subset_180_cal.json
- agument_with_img_unique_subset_180_val.json

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
