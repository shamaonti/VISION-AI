import os, cv2
from datetime import datetime

def save_snapshot(frame, out_dir='output', prefix='snapshot'):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path
