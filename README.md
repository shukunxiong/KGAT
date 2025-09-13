## KGAT yolo-v8 version

## Top News
**`2025-09-13`**:**The current version supports adversarial training for Amtd, Acls, and Aloc, as well as training for the KGAT+MTD version.**  

## Training step
### 1. Dataset Preparation

**This article uses the VOC format for training. You need to create your own dataset before training.**    

#### Before training, 

1.Place the label files under your_dataset/Annotations.The code only support xml labels.  
2.Place the clean images under your_dataset/clean.
3.Place the data for train and val under your_dataset/Main/train.txt and your_dataset/Main/val.txt,respectivel.
In conclusion, the directory structure of your dataset should be organized as follows:
```python
your_dataset/                  # Dataset root directory
├── Annotations/               # Store XML label files
│   ├── image_001.xml
│   ├── image_002.xml
│   └── ...
├── clean/                     # Store clean images
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── Main/                      # Store training/validation split files
│   ├── train.txt              # Training set list (each line is an image name without suffix)
│   │   ├── image_001
│   │   ├── image_003
│   │   └── ...
│   └── val.txt                # Validation set list (same format as train.txt)
│       ├── image_002
│       ├── image_004
│       └── ...
└── dataset_relation/          # Store dataset relation file
    └── your_dataset_relation.pt
```

### 2. Dataset Processing 

After organizing the dataset, we need to use annotation.py to generate train.txt and val.txt for training.  
Modify the parameters in annotation.py. For the first training, you only need to modify classes_path, which is used to point to the TXT file corresponding to the detection classes.  
When training your own dataset, you can create a your_data.txt file containing the classes you need to distinguish.  
The content of model_data/your_data.txt is as follows:      
```python
cat
dog
...
```
Modify classes_path in annotation.py to correspond to your_data.txt, then run annotation.py. 

### 3. Starting Network Training

**classes_path is used to point to the TXT file corresponding to the detection classes, which is the same TXT file used in annotation.py! It must be modified when training your own dataset!**  

After modifying classes_path, you can run train.py or train_adversarial to start training.train.py is for the origin training for yolov8, and train_adversarial is for the adversarial training for yolov8.Before start adversarial training, you should set the relevant parameters. After training for several epochs, the weights will be generated in the save_dir folder.

### 4. Training Result Prediction  

Two files are required for training result prediction: yolo.py and one of the flies in test aaccording to your need. We support three prediction versions, get_map_for_clean_samples.py is for prediction on clean samples, get_map_for_adv_attacks.py is for prediction on samples under adversarial perturbations and get_map_for_common_corruptions.py is for prediction on samples samples under universal perturbations.
**First**, you should modify model_path and classes_path in yolo.py.
**Secondly**,if you want to predict for the clean samples or common corruptions samples, you should modify classes_path in get_map_for_clean_samples.py and get_map_for_common_corruptions.py, respectively. If you want to predict for the adversarial samples, you shoud modify classes_path and model_path in get_map_for_adv_attacks.py. 
After making the modifications, you can run predict.py to perform detection. Once running, enter the image path to start detection.  


