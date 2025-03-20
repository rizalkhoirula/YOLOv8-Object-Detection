<h1 align="center">YOLOv8 Object Detection</h1>
<p align="center">
  <img src="https://github.com/ultralytics/assets/raw/main/im/yolov8_banner.png" alt="YOLOv8" width="600"/>
</p>

## 🚀 Introduction
<p>YOLOv8 (You Only Look Once) is the latest version of the state-of-the-art real-time object detection model developed by Ultralytics. This repository provides an implementation for training, evaluating, and deploying YOLOv8 for object detection tasks.</p>

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yolov8-object-detection.git
cd yolov8-object-detection

# Install dependencies
pip install ultralytics opencv-python numpy torch torchvision
```

## 🔥 Quick Start

### Run Inference
```python
from ultralytics import YOLO
import cv2

# Load model
yolo_model = YOLO("yolov8n.pt")  # Use yolov8s.pt, yolov8m.pt, etc. for other versions

# Load image
image_path = "path/to/your/image.jpg"
image = cv2.imread(image_path)

# Run detection
results = yolo_model(image)

# Show results
results.show()
```

### Train a Custom Model
```bash
# Train on a custom dataset (COCO format or YOLO format)
yolo train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```

### Evaluate Model
```bash
# Run validation on the dataset
yolo val model=runs/train/exp/weights/best.pt data=dataset.yaml
```

## 📂 Dataset Format
<p>To train a custom model, ensure your dataset follows the COCO or YOLO format. Example YOLO format:</p>

```yaml
data.yaml:
  train: ./data/train/images  # Path to training images
  val: ./data/val/images  # Path to validation images
  test: ./data/test/images  # Path to test images (optional)
  nc: 2  # Number of classes
  names: ['class1', 'class2']
```

## ⚡ Deployment
<p>After training, deploy the model using ONNX or TensorRT for optimized performance.</p>

```bash
# Export to ONNX
yolo export model=runs/train/exp/weights/best.pt format=onnx
```

## 📖 References
- <a href="https://github.com/ultralytics/ultralytics">YOLOv8 Official Repository</a>
- <a href="https://docs.ultralytics.com">YOLOv8 Documentation</a>

## 🎯 License
<p>This project is released under the MIT License.</p>
