# Image Processing Projects

This repository contains multiple image processing projects in Python using OpenCV and YOLO. It includes basic tutorials, live detection, Haar cascade, and YOLO object detection examples.

## Setup & Installation

1. Clone the repository:

```bash
git clone https://github.com/nisanrsenel/image-processing-projects.git
cd image-processing-projects
```
2. Install required packages:
   
```bash
pip install opencv-python numpy
pip install ultralytics  # for YOLOv8 projects
```

Basic Steps
-Learn about pixels, grayscale conversion, thresholding, and masks.
-Scripts: basic_steps.py - object_detection.py

Live Detection
-Detect red objects in video using OpenCV.
-Script: live_detection.py
-Shows live object count with bounding boxes.

Haar Cascade Object Detection
-Detect cars in images or video using pre-trained Haar classifiers.
-Script: haar_cascade.py

YOLO Object Detection
-Test videos using trained YOLO models.
-Scripts: yolo.py - yolo_test_application.py 
