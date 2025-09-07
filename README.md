# 🏎️ F1 AI Team Detection

**Computer vision project using YOLO to detect and classify Formula 1 teams in race images and videos. The project also estimates and visualizes the distance between cars in real-time, displaying the gap in both meters and seconds, enabling dynamic race analysis.**

---

<p align="center">
  <img src="weights/f1_gif_2.gif" alt="F1 AI Detection Demo" style="width:70%;"/>
</p>

---

## 📄 Overview

This repository contains code and notebooks for an AI-powered Formula 1 team detection system. Leveraging the YOLO (You Only Look Once) architecture, the model detects and classifies F1 team cars in images and videos with high accuracy. Additionally, it estimates the distance between cars, providing real-time gap metrics in both meters and seconds, enabling comprehensive racing analysis and visualization.

> [!NOTE]
> **To access the dataset**, please visit: [F1 Car 2023 Dataset](https://app.roboflow.com/vforvitorio/f1-car-2023-1bsn2)

> [!WARNING]
> **Some materials are excluded** by `.gitignore` and not uploaded to this repository, including training results, cache files, and processed datasets. Generate them by running the provided notebooks.

---

## ✨ Key Features

- **🎯 YOLO-based Detection:** Real-time identification and classification of Formula 1 cars by team
- **📏 Distance Estimation:** Precise calculation and visualization of car-to-car distances (meters and seconds)
- **🎬 Result Visualization:** Annotated real-time video output with team labels and gap information
- **🔍 False Positive Filtering:** Intelligent system to eliminate ghost detections through overlap analysis
- **🏃 Object Tracking:** Maintains consistent car identity throughout video sequences
- **⚙️ Custom Confidence Thresholds:** Team-specific detection optimization

---

## 📁 Project Structure

```
F1_AI_team_detection/
├── weights/                     # Trained models and demo files
│   ├── f1_gif_2.gif            # Demo animation
│   ├── fine_tuned.pt           # Final optimized model
│   ├── yolo_medium_detection.pt # Fine-tuned medium YOLO model
│   └── yolo_small_detection.pt  # Fine-tuned small YOLO model
├── .gitignore
├── LICENSE
├── README.md
├── YOLO_fine_tune.ipynb         # Main YOLO training and fine-tuning notebook
├── data_augmentation.py         # Training data augmentation script
└── gap_calculation.ipynb        # Distance calculation and visualization notebook
```

---

## 📊 Performance Metrics

The final optimized model achieved exceptional results across all metrics:

### Overall Performance
| Metric    | Value |
|-----------|-------|
| **mAP50**     | **0.940** |
| **mAP50-95**  | **0.781** |
| **Precision** | **0.925** |
| **Recall**    | **0.771** |

### Team-specific Performance
| Team         | Precision | Recall | mAP50 | mAP50-95 |
|--------------|-----------|--------|-------|----------|
| **Kick Sauber**  | 1.000     | 0.526  | 0.809 | 0.642    |
| **Racing Bulls** | 0.848     | 1.000  | 0.995 | 0.796    |
| **Alpine**       | 1.000     | 0.447  | 0.962 | 0.851    |
| **Ferrari**      | 1.000     | 0.819  | 0.995 | 0.910    |
| **Haas**         | 0.881     | 1.000  | 0.995 | 0.895    |
| **McLaren**      | 1.000     | 0.378  | 0.859 | 0.709    |
| **Mercedes**     | 0.975     | 1.000  | 0.995 | 0.796    |
| **Williams**     | 0.698     | 1.000  | 0.913 | 0.651    |

---

## 🚀 Getting Started

### 📋 Requirements

```
python 3.10+
torch 2.5.1
torchvision
ultralytics 8.3.137
opencv-python
numpy
pandas
jupyter
```

### 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VforVitorio/F1_AI_team_detection.git
   cd F1_AI_team_detection
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision ultralytics opencv-python numpy pandas jupyter
   ```

3. **Download the dataset** from [Roboflow](https://app.roboflow.com/vforvitorio/f1-car-2023-1bsn2) and place it in the `f1-dataset/` directory.

### 🎯 Usage

#### 1. **Model Training**
- Open `YOLO_fine_tune.ipynb` in Jupyter Notebook
- Configure training parameters as needed
- Execute cells to train and fine-tune the YOLO model
- Monitor training progress and validation metrics

#### 2. **Distance Calculation and Visualization**
- Open `gap_calculation.ipynb` notebook
- Set your video path for analysis
- Run the notebook to process videos and visualize results
- Output includes annotated videos with real-time distance calculations

#### 3. **Data Augmentation** (Optional)
- Use `data_augmentation.py` to expand your training dataset
- Improves model robustness and performance

---

## 🔍 Key Capabilities

The system provides comprehensive race analysis by automatically calculating:

- **📐 Precise Distance Metrics:** Real-time distance in meters between consecutive cars
- **⏱️ Time Gap Analysis:** Conversion to time differences in seconds (based on 300 km/h reference speed)
- **🏁 Team Recognition:** Accurate identification of all F1 teams with custom confidence thresholds
- **🎥 Live Visualization:** Real-time visual labels and annotations
- **🚫 Ghost Detection Elimination:** Advanced overlap analysis to filter false positives
- **🔄 Continuous Tracking:** Maintains car identity throughout video sequences

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the project.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Roboflow** for the F1 Car 2023 dataset
- **Ultralytics** for the YOLO implementation
- **Formula 1** community for inspiration and testing

---

<div align="center">

**⭐ If you find this project useful, please give it a star! ⭐**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/VforVitorio/F1_AI_team_detection)

</div>
