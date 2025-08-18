# 🧠 Image Processing Projects

Bu repository, Python kullanarak OpenCV ve YOLO ile geliştirilmiş çeşitli görüntü işleme projelerini içerir. Hem temel kavramlar hem de gerçek zamanlı nesne tespiti örnekleri sunulmuştur. / This repository contains various image processing projects developed with OpenCV and YOLO using Python. Both fundamental concepts and real-time object detection examples are presented.

---

## 📚 Proje İçeriği / Project Content

1. 🧩 Temel Görüntü İşleme Adımları / Basic Image Processing Steps
-Piksel yapısı / Pixel structure
-Grayscale dönüşümü / Grayscale transformation
-Thresholding / Thresholding
-Mask kullanımı / Mask usage

Script: basic_steps.py, object_detection.py

2. 🎥 Gerçek Zamanlı Tespit / Real-Time Detection (Live Detection)
-Kırmızı nesneleri video akışında tespit eder / Detects red objects in the video stream
-Bounding box ile nesne sayısını gösterir / Displays the number of objects with a bounding box

Script: live_detection.py

3. 🚗 Haar Cascade ile Nesne Tespiti / Object Detection with Haar Cascade
-Hazır Haar sınıflandırıcılarla araç tespiti / Vehicle detection with pre-built Haar classifiers
-Görüntü veya video üzerinde çalışır / Works on images or video

Script: haar_cascade.py

4. 🦾 YOLO ile Nesne Tespiti / Object Detection with YOLO
-Eğitilmiş YOLO modelleriyle video analizi / Video analysis with trained YOLO models
-Nesne sınıflandırma ve bounding box çizimi / Object classification and bounding box drawing

Scripts: yolo.py, yolo_test_application.py

---

## 🛠️ Kullanılan Teknolojiler / Technologies Used
-Python ile görüntü işleme / Used in Python Image processing
-OpenCV kullanımı / Using OpenCV
-YOLOv8 ile nesne tespiti / Object detection with YOLOv8
-Gerçek zamanlı video analizi / Real-time video analysis
-Haar cascade algoritmaları / Haar cascade algorithms

---

## 📁 Kurulum & Başlangıç / Installation and Getting Started

### 🔧 Repository'yi klonlayın / Clone the Repository

```bash
git clone https://github.com/nisanrsenel/image-processing-projects.git
cd image-processing-projects
```

📦 Gerekli paketleri yükleyin / Install necessary packages
```bash
pip install opencv-python numpy
pip install ultralytics  # YOLOv8 projeleri için
```
