# 🌟Knowledge-Guided Adversarial Training for Infrared Object Detection via Thermal Radiation Modeling

 [![arXiv](https://img.shields.io/badge/arXiv-2603.25170-b31b1b.svg)](https://arxiv.org/abs/2603.25170)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://arxiv.org/pdf/2603.25170)

## 📰 Top News

- **`2026-03-26`** 🎉 **Great news!** Our paper has been accepted by the International Journal of Computer Vision (IJCV). We're thrilled to share this milestone with the community!
- **`2025-09-13`** The current version now supports adversarial training for AMTD, ACLS, and ALOC objectives, as well as the enhanced KGAT+MTD training framework.


## 📥 Quickstart
### 💡 1. Dataset Preparation

This project supports VOC format annotations. You'll need to prepare your custom dataset following the structure below.

**Before training, organize your dataset as follows:**

1. Place XML label files in `your_dataset/Annotations/` (only XML format is supported)
2. Place clean images in `your_dataset/clean/`
3. Create train/val split lists in `your_dataset/Main/train.txt` and `your_dataset/Main/val.txt`

**Dataset Directory Structure:**
```
your_dataset/                  # Dataset root directory
├── Annotations/               # XML label files
│   ├── image_001.xml
│   ├── image_002.xml
│   └── ...
├── clean/                     # Clean images
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── Main/                      # Train/validation splits
│   ├── train.txt              # Training set (image names without suffix)
│   │   ├── image_001
│   │   ├── image_003
│   │   └── ...
│   └── val.txt                # Validation set (same format)
│       ├── image_002
│       ├── image_004
│       └── ...
└── dataset_relation/          # Graph relation file
    └── your_dataset_relation.pt
```


###  🗺️ 2. Data Annotation Processing

After organizing your dataset, run `annotation.py` to generate `train.txt` and `val.txt` splits.

**Configuration steps:**
1. Create a class definition file (e.g., `model_data/your_data.txt`) with one class per line:
   ```
   cat
   dog
   ...
   ```

2. Update `classes_path` in `annotation.py` to point to your class definition file

3. Run `annotation.py` to generate the training and validation splits


### 🔬 3. Model Training

**Important:** The `classes_path` parameter must match the class definition file used in the annotation step!

**Training modes:**
- `train.py` - Standard YOLOv8 training
- `train_adversarial.py` - Adversarial training for robustness

**Steps:**
1. Update `classes_path` in the training script to match your dataset
2. Configure adversarial training parameters if using `train_adversarial.py`
3. Run the training script - trained weights will be saved to the `save_dir` folder


### 🔬 4. Model Inference & Evaluation

This project provides three prediction modes for different evaluation scenarios:

**Prediction scripts:**
- `get_map_for_clean_samples.py` - Evaluate on clean, unperturbed images
- `get_map_for_adv_attacks.py` - Evaluate robustness to adversarial perturbations
- `get_map_for_common_corruptions.py` - Evaluate robustness to common corruptions

**Configuration:**
1. Update `model_path` and `classes_path` in `yolo.py`
2. For clean/corruption evaluation: update `classes_path` in respective scripts
3. For adversarial evaluation: update both `classes_path` and `model_path` in `get_map_for_adv_attacks.py`

**Running inference:**
```bash
python predict.py
```
When prompted, enter the image path to start detection.


---

## 📚 Citation

If you find our work helpful, please cite:

```bibtex
@article{zhao2026knowledge,
  title={Knowledge-Guided Adversarial Training for Infrared Object Detection via Thermal Radiation Modeling},
  author={Zhao, Shiji and Xiong, Shukun and Yuan, Maoxun and Huang, Yao and Duan, Ranjie and Guo, Qing and Chen, Jiansheng and Duan, Haibin and Wei, Xingxing},
  journal={arXiv preprint arXiv:2603.25170},
  year={2026}
}
```



