Vision AI (YOLOv5 + Deep SORT + Action Recognition + Grad-CAM + Email Alerts)
---------------------------------------------------------------------------
Setup steps:
1. Create and activate a Python virtual environment.
2. Install requirements:
   pip install -r requirements.txt
3. Clone YOLOv5 into the yolov5/ folder if you prefer local:
   git clone https://github.com/ultralytics/yolov5.git yolov5
   pip install -r yolov5/requirements.txt
4. Place action model weights at action_model/action_model.pth
5. Edit alerts/send_email.py and place your Gmail app password into APP_PASSWORD.
6. Run: python main.py
