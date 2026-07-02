# Object-Detection-using-YOLOv8-and-Transfer-Learning
Built an advanced object detection system using YOLOv8 with transfer learning to improve detection performance on custom datasets. Implemented data preprocessing techniques and optimized model performance for higher accuracy. Integrated OpenCV for real-time video capture and visualization of detected objects.
## 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Core programming language |
| YOLOv8 | Object detection model |
| Transfer Learning | Fine-tuning pretrained model |
| OpenCV | Image processing and visualization |
| Jupyter Notebook | Model training and experimentation |
| Ultralytics | YOLOv8 framework |

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Object-Detection-using-YOLOv8-and-Transfer-Learning.git

cd Object-Detection-using-YOLOv8-and-Transfer-Learning
```

### 2. Install dependencies

```bash
pip install ultralytics opencv-python numpy matplotlib jupyter
```

### 3. Train the Model

Open and run:

```text
yolov8_training.ipynb
```

### 4. Run Object Detection

```bash
python object_detection.py
```

---

## 📁 Project Structure

```text
Object-Detection-using-YOLOv8-and-Transfer-Learning/
├── .gitignore
├── LICENSE
├── README.md
├── object_detection.py      # Performs object detection
├── pasacal.ipynb            # Dataset preprocessing/annotation
└── yolov8_training.ipynb    # Model training using YOLOv8
```

---

## 📸 Demo

| Input | Output |
|-------|--------|
| Image | Detects objects with bounding boxes and labels |
| Video | Performs real-time object detection |
| Custom Dataset | Detects trained object classes with improved accuracy |

---

## 📊 Model Performance

- Model: YOLOv8
- Training: Transfer Learning
- Dataset: Pascal VOC / Custom Dataset
- Metrics: Precision, Recall, mAP

---

## 🚀 Future Improvements

- Deploy as a Streamlit web application
- Add webcam-based live detection
- Export model to ONNX/TensorRT
- Improve accuracy with larger datasets
- Support additional object classes
