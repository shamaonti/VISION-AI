# Vision AI 🚨

### Real-Time Suspicious Activity Detection System

---

## 📌 Overview

Vision AI is a real-time surveillance system that detects suspicious activities such as theft, violence, or abnormal behavior using computer vision and deep learning. The system processes live video feeds, tracks objects, recognizes actions, and sends alerts with visual evidence.

---

## 🚀 Features

* 🎯 Real-time object detection using YOLOv5
* 🧍 Multi-object tracking using Deep SORT
* 🧠 Action recognition using CNN model
* 🔍 Explainable AI using Grad-CAM heatmaps
* 📩 Automated email alerts with snapshots
* 📷 Live webcam/video feed processing

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * OpenCV
  * PyTorch
  * YOLOv5
  * Deep SORT
* **Concepts Used:**

  * Computer Vision
  * Deep Learning
  * Explainable AI (XAI)

---

## 📂 Project Structure

```
vision_ai/
├── yolov5/
├── deep_sort/
│   ├── detection.py
│   ├── tracker.py
│   ├── nn_matching.py
│   └── feature_extractor.py
├── action_model/
│   ├── model.py
│   └── predict.py
├── explainable_ai/
│   └── grad_cam.py
├── alerts/
│   └── send_email.py
├── utils/
│   └── snapshot.py
├── output/
├── main.py
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/vision-ai.git
cd vision-ai
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python main.py
```

* The system will start webcam/video processing
* Detect suspicious activities
* Send alerts with snapshots

---

## 📸 Output

* Snapshot images stored in `/output`
* Email alerts with attached evidence
* Grad-CAM heatmaps for explainability

---

## 📧 Alert System

* Sends email notifications when suspicious activity is detected
* Includes:

  * Captured image
  * Detection details

---

## 🎯 Applications

* Smart surveillance systems
* Crime detection
* Security monitoring
* Public safety systems

---

## 👩‍💻 Author

**Shama Bandenavaj Onti**

* Computer Science Engineering Student
* AI & Full Stack Developer

---

## 📌 Future Improvements

* WhatsApp/SMS alerts integration
* Cloud deployment (AWS)
* Mobile app integration
* Advanced behavior prediction

---

## ⭐ Conclusion

Vision AI demonstrates how AI can be used for real-time security monitoring by combining object detection, tracking, action recognition, and explainable AI into a single intelligent system.
