# 🔍 Vision AI — Real-Time Suspicious Activity Detection

A real-time suspicious activity detection system using YOLOv5, Deep SORT, CNN, and Grad-CAM. The system detects suspicious objects/activities, tracks them, and sends automated alerts via Email and WhatsApp with snapshots.

---

## 📌 Features

- 🎯 Real-time object detection using YOLOv5
- 👤 Multi-person tracking using Deep SORT
- 🧠 Action recognition using CNN model
- 🔥 Explainable AI heatmaps using Grad-CAM
- 📧 Automated alerts via Email with snapshot
- 📱 WhatsApp alerts with snapshot
- 🗂️ Supports both webcam and video file input

---

## ⚠️ Large Files (Download Manually)

These files exceed GitHub's size limit and must be downloaded separately:

### 1. YOLOv3 Weights
Download from official source:
```
https://pjreddie.com/media/files/yolov3.weights
```
Place it in:
```
projects/PROJECT/yolo-coco/yolov3.weights
```

### 2. Action Model Checkpoint (`checkpoint.pth`)
Download from Google Drive:
```
[Add your Google Drive link here]
```
Place it in:
```
projects/vision_aiii/action_model/checkpoint.pth
```

### 3. Test Video (`test.mp4`)
You can use any `.mp4` video file for testing.
Place it in:
```
projects/PROJECT/test.mp4
```

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/shamaonti/VISION-AI.git
cd VISION-AI
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r projects/vision_aiii/requirements.txt
```

---

## ▶️ How to Run

```bash
cd projects/vision_aiii
python main.py
```

---

## 📁 Project Structure

```
projects/
├── PROJECT/
│   ├── yolo-coco/
│   │   ├── coco.names
│   │   ├── yolov3.cfg
│   │   └── yolov3.weights        ← Download manually
│   └── test.mp4                  ← Add your own video
│
└── vision_aiii/
    ├── main.py                   ← Entry point
    ├── requirements.txt
    ├── action_model/
    │   ├── model.py
    │   ├── predict.py
    │   └── checkpoint.pth        ← Download manually
    ├── alerts/
    │   └── send_email.py
    ├── deep_sort/
    │   ├── detection.py
    │   ├── feature_extractor.py
    │   ├── nn_matching.py
    │   └── tracker.py
    ├── explainable_ai/
    │   └── grad_cam.py
    └── data/
        └── data.yaml
```

---

## 🔧 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| YOLOv5 | Object detection |
| Deep SORT | Multi-object tracking |
| CNN | Action recognition |
| Grad-CAM | Explainable AI heatmaps |
| OpenCV | Video processing |
| smtplib | Email alerts |
| PyWhatKit | WhatsApp alerts |

---

## 👩‍💻 Developer

**Shama Bandenavaj Onti**  
📧 shamaonti2@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/shama-onti) | [GitHub](https://github.com/shamaonti)
