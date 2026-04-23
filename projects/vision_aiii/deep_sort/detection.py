import numpy as np

class Detection:
    """Container for a single detection in tlwh format."""
    def __init__(self, tlwh, confidence, feature=None, class_id=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32) if feature is not None else None
        self.class_id = class_id

    def to_tlbr(self):
        x, y, w, h = self.tlwh
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    def to_xyah(self):
        x, y, w, h = self.tlwh
        cx = x + w / 2.
        cy = y + h / 2.
        return np.array([cx, cy, w / float(h), h], dtype=np.float32)
