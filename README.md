# F1 AI Team Detection

**Computer vision project using YOLO to detect and classify Formula 1 teams in race images and videos. The project also estimates and visualizes the distance between cars in real-time, displaying the gap in both meters and seconds, enabling dynamic race analysis.**

---



## Overview

This repository contains code and notebooks for an AI-powered Formula 1 team detection system. Leveraging the YOLO (You Only Look Once) architecture, the model detects and classifies F1 team cars in images and videos. Additionally, it estimates the distance between cars, providing real-time gap metrics in both meters and seconds. This enables dynamic analysis and visualization for racing events.


> [!WARNING]
> **Model weights are not included in this repository.** You will need to train the models yourself using the provided notebooks.
>
> **To access the dataset**, please visit: [F1 Car 2023 Dataset
> ](https://app.roboflow.com/vforvitorio/f1-car-2023-1bsn2)

---



## Features

- **YOLO-based Detection:** Identifies and classifies Formula 1 cars by team in real-time.
- **Distance Estimation:** Calculates and visualizes the distance between detected cars (meters and seconds).
- **Result Visualization:** Outputs annotated real-time video with team labels and gap information.
- **False Positive Filtering:** Intelligent system to eliminate ghost detections.
- **Object Tracking:** Maintains car identity throughout the video.

---


## Project Structure

├── .gitignore
├── data_augmentation.py            # Script for training data augmentation
├── gap_calculation.ipynb           # Notebook for calculating and visualizing distances between cars
├── README.md
├── YOLO_fine_tune.ipynb            # Main notebook for training and fine-tuning YOLO models
├── yolo11n.pt                      # Pre-trained YOLO model (nano)
├── yolo12m.pt                      # Pre-trained YOLO model (medium)
├── yolo12s.pt                      # Pre-trained YOLO model (small)
├── f1-dataset/                     # Dataset for training and validation
│   ├── data.yaml                   # Class configuration and paths
│   ├── test/                       # Test images and labels
│   ├── train/                      # Training images and labels
│   └── valid/                      # Validation images and labels
├── videos/                         # F1 videos for processing
├── weights/                        # Saved trained models
│   ├── fine_tuned.pt               # Final optimized model
│   ├── yolo_medium_detection.pt    # Fine-tuned medium YOLO model
│   └── yolo_small_detection.pt     # Fine-tuned small YOLO model
└── yolo-files/                     # Files generated during training
    └── runs/                       # Training and evaluation results

---



## Performance Metrics

The final optimized model achieved the following results:

| Metric    | Value |
| --------- | ----- |
| mAP50     | 0.940 |
| mAP50-95  | 0.781 |
| Precision | 0.925 |
| Recall    | 0.771 |

### Team-specific Performance

| Team         | Precision | Recall | mAP50 | mAP50-95 |
| ------------ | --------- | ------ | ----- | -------- |
| Kick Sauber  | 1.000     | 0.526  | 0.809 | 0.642    |
| Racing Bulls | 0.848     | 1.000  | 0.995 | 0.796    |
| Alpine       | 1.000     | 0.447  | 0.962 | 0.851    |
| Ferrari      | 1.000     | 0.819  | 0.995 | 0.910    |
| Haas         | 0.881     | 1.000  | 0.995 | 0.895    |
| McLaren      | 1.000     | 0.378  | 0.859 | 0.709    |
| Mercedes     | 0.975     | 1.000  | 0.995 | 0.796    |
| Williams     | 0.698     | 1.000  | 0.913 | 0.651    |

---


## Getting Started

### Requirements

python 3.10+ torch 2.5.1 torchvision ultralytics 8.3.137 opencv-python numpy pandas

### Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/F1_yolo_team_detection.git
cd F1_yolo_team_detection
```

### Usage

1. **Model Training:**
   * Open the [YOLO_fine_tune.ipynb](vscode-file://vscode-app/c:/Users/victo/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) notebook in Jupyter
   * Adjust parameters as needed
   * Run the cells to train the model
2. **Distance Calculation and Visualization:**
   * Open the [gap_calculation.ipynb](vscode-file://vscode-app/c:/Users/victo/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) notebook
   * Configure the video path for analysis
   * Run to process the video and visualize results

---



## Examples

The system automatically calculates:

* Distance in meters between consecutive cars
* Time difference in seconds (based on 300 km/h speed)
* Identifies teams with a custom confidence threshold per class
* Displays visual labels in real-time
* Eliminates ghost detections through overlap analysis

---



## License

This project is licensed under the MIT License.
