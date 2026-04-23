import cv2
import torch
from ultralytics import YOLO
import numpy as np
import smtplib
import ssl
from email.message import EmailMessage
import os
from datetime import datetime

from explainable_ai.grad_cam import SimpleGradCAM, generate_gradcam
from action_model.predict import load_action_model, predict_action

# =========================
# CONFIG
# =========================
YOLO_MODEL_PATH = r"C:\Users\HP LAPTOP\Desktop\projectss\projects\vision_aiii\runs\detect\knife_detection_run\weights\best.pt"
ACTION_MODEL_PATH = "action_model/final_action_model.pth"  # Updated to latest trained model
OUTPUT_DIR = "output"

CLASS_NAMES = ["knife", "no_knife"]  # exactly from data.yaml

EMAIL_SENDER = "shamaonti2@gmail.com"
EMAIL_RECEIVER = "shamaonti18@gmail.com"
EMAIL_PASSWORD = "ioxv zdup ijwc mfzb"  # Gmail App Password

ALERT_COOLDOWN = 10  # seconds
last_alert_time = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Email Alert Function
# =========================
def send_email_alert(snapshot_path, heatmap_path):
    try:
        msg = EmailMessage()
        msg["Subject"] = "🚨 Suspicious Activity Detected"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg.set_content("Knife action detected! Check attached snapshot and heatmap.")

        for file_path in [snapshot_path, heatmap_path]:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    file_data = f.read()
                    file_name = os.path.basename(file_path)
                msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("[INFO] Email alert sent ✅")
    except Exception as e:
        print(f"[ERROR] Email sending failed: {e}")

# =========================
# Main Detection
# =========================
def main():
    global last_alert_time

    # ---- Load YOLO ----
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"[ERROR] YOLO weights not found at {YOLO_MODEL_PATH}")
        return
    print("[INFO] Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    # ---- Load Action Model ----
    if not os.path.exists(ACTION_MODEL_PATH):
        print(f"[ERROR] Action model not found at {ACTION_MODEL_PATH}")
        return
    print("[INFO] Loading Action model...")
    action_model = load_action_model(ACTION_MODEL_PATH, num_classes=len(CLASS_NAMES))

    # ---- Setup Grad-CAM ----
    target_layer = action_model.layer4[-1].conv2  # ResNet last conv layer
    gradcam = SimpleGradCAM(action_model, target_layer)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            alert_triggered = False
            roi_resized = None  # track last ROI for Grad-CAM

            # ================== YOLO DETECTION ==================
            results = yolo_model(frame)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)

                    # Draw bounding box
                    color = (0, 0, 255) if label == "knife" else (0, 255, 0)
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Crop ROI and check action
                    roi = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    if roi.size > 0:
                        roi_resized = cv2.resize(roi, (224, 224))
                        action_label = predict_action(action_model, roi_resized, CLASS_NAMES)
                        cv2.putText(frame, f"Action: {action_label}", (xyxy[0], xyxy[3]+25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # ✅ Only trigger alert if BOTH conditions are satisfied
                        if label == "knife" and conf > 0.70 and action_label == "knife":
                            alert_triggered = True

            # ================== IF ALERT TRIGGERED ==================
            current_time = datetime.now().timestamp()
            if alert_triggered and (current_time - last_alert_time > ALERT_COOLDOWN):
                last_alert_time = current_time
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_path = os.path.join(OUTPUT_DIR, f"alert_{timestamp}.jpg")
                cv2.imwrite(snapshot_path, frame)

                # Grad-CAM (save heatmap)
                try:
                    if roi_resized is not None:
                        heatmap_path = os.path.join(OUTPUT_DIR, f"heatmap_{timestamp}.jpg")
                        overlay, _ = generate_gradcam(
                            gradcam,
                            roi_resized,
                            class_idx=CLASS_NAMES.index("knife"),
                            alpha=0.6,
                            save_path=heatmap_path
                        )
                    else:
                        heatmap_path = snapshot_path
                except Exception as e:
                    print(f"[WARN] Grad-CAM failed: {e}")
                    heatmap_path = snapshot_path

                send_email_alert(snapshot_path, heatmap_path)

            # Show live window
            display_frame = cv2.resize(frame, (640, 480))
            cv2.imshow("Suspicious Activity Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        gradcam.close()


# =========================
# Run
# =========================
if __name__ == "__main__":
    main()
