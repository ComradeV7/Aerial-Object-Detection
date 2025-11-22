# Aerial Object Classification & Detection

An AI-powered surveillance system capable of distinguishing between **Birds** and **Drones** in aerial imagery. This project leverages **Deep Learning (CNNs & Transfer Learning)** for binary classification and **YOLOv8** for real-time object detection, deployed via a user-friendly **Streamlit** application.

---

<img width="1909" height="959" alt="image" src="https://github.com/user-attachments/assets/72a0e3bd-8729-4799-84dc-9cb4a6efda3e" />


## Project Overview

Distinguishing between birds and drones is a critical challenge for airspace security, airport safety (bird strikes), and wildlife monitoring. This solution provides an automated pipeline to:
1.  **Classify** an image as "Bird" or "Drone" using a fine-tuned **ResNet18** model.
2.  **Detect** and localize objects in real-time using **YOLOv8n**.

## Key Features

* **Dual-Mode Analysis:** Switch between simple Classification (What is it?) and Object Detection (Where is it?).
* **High Accuracy:** Achieved **94.91% Accuracy** with ResNet18 Transfer Learning.
* **Real-Time Detection:** YOLOv8n integration for bounding box localization.
* **Interactive Dashboard:** Built with **Streamlit** for easy image uploading and visualization.
* **Robust Preprocessing:** Automated resizing, normalization, and augmentation pipelines.

## 🛠️ Tech Stack

* **Language:** Python 3.11
* **Deep Learning:** PyTorch, Torchvision
* **Object Detection:** YOLOv8 (Ultralytics)
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy, OpenCV
* **Visualization:** Matplotlib, Seaborn

---

## Dataset Structure

The project uses two separate datasets:
1.  **Classification Data:** Organized by class folders (`/bird`, `/drone`).
2.  **Detection Data:** YOLO format with images and `.txt` label files.

*Note: The raw dataset is not included in this repo to save space.*

## Model Performance

We trained and compared multiple models. Below are the final results on the Test Set:

| Model | Accuracy | Precision | Recall (Drone) | F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet18 (Transfer)** | **94.91%** | **0.92** | **0.97** | **0.94** | **Selected** |
| Custom CNN | 84.72% | 0.80 | 0.87 | 0.83 | Discarded |

* **Insight:** ResNet18 demonstrated superior recall (97%), ensuring minimal missed drone threats compared to the Custom CNN.

---

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/ComradeV7/Aerial-Object-Detection.git
cd Aerial-Object-Detection
```

### 2. Install Dependencies (use Virtual Environment)

```bash
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

## 📁 Directory Structure

```text
Aerial-Object-Detection/
├── .gitignore               # Git ignore rules
├── README.md                # Project Documentation
├── app.py                   # Main Streamlit Application
├── best_custom_cnn.pth      # Trained Custom CNN Weights
├── best_transfer_model.pth  # Trained ResNet18 Weights (Used in App)
├── data-preprocessing.ipynb # Notebook for data loading & augmentation
├── data.yaml                # YOLOv8 Configuration file
├── requirements.txt         # Python Dependencies
├── train-YOLO.ipynb         # Notebook for training YOLOv8
├── train-classification.ipynb # Notebook for training CNNs
├── yolo11n.pt               # Pre-trained YOLOv11n weights
├── yolov8n.pt               # Pre-trained YOLOv8n weights
├── runs/                    # YOLOv8 Training Outputs (Auto-generated)
│   └── detect/weights/best.pt
└── data/                    # Dataset Directory (Local only, not on GitHub)
    ├── classification_dataset/
    └── object_detection_Dataset/
```

## Future Improvements

* Implement Video Inference for live webcam feeds.
* Add "Negative Samples" (Empty Sky) to reduce YOLO false positives.
* Deploy to Cloud (AWS EC2 or Streamlit Community Cloud).

## Result

<img width="1910" height="950" alt="image" src="https://github.com/user-attachments/assets/3056a280-4685-43a0-bb21-b3266f68b289" />
