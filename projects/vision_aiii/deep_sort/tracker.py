import numpy as np
from scipy.optimize import linear_sum_assignment
from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric

class Track:
    def __init__(self, tlwh, track_id, feature=None):
        self.tlwh = np.array(tlwh, dtype=np.float32)
        self.track_id = track_id
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self.time_since_update = 0

    def to_tlbr(self):
        x, y, w, h = self.tlwh
        return [x, y, x + w, y + h]

    def update(self, tlwh, feature=None):
        self.tlwh = np.array(tlwh, dtype=np.float32)
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > 100:
                self.features = self.features[-100:]
        self.time_since_update = 0

class DeepSort:
    def __init__(self, max_age=30, max_cosine_distance=0.3, nn_budget=100):
        self.tracks = []
        self._next_id = 1
        self.max_age = max_age
        self.metric = NearestNeighborDistanceMetric(matching_threshold=max_cosine_distance, budget=nn_budget)

    def predict(self):
        for t in self.tracks:
            t.time_since_update += 1

    def _iou_cost(self, detections):
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        cost = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            tx1, ty1, tx2, ty2 = tr.to_tlbr()
            for j, det in enumerate(detections):
                dx1, dy1, dx2, dy2 = det.to_tlbr()
                ix1 = max(tx1, dx1); iy1 = max(ty1, dy1)
                ix2 = min(tx2, dx2); iy2 = min(ty2, dy2)
                iw = max(0., ix2 - ix1); ih = max(0., iy2 - iy1)
                inter = iw * ih
                area_t = (tx2 - tx1) * (ty2 - ty1)
                area_d = (dx2 - dx1) * (dy2 - dy1)
                union = area_t + area_d - inter + 1e-6
                iou = inter / union
                cost[i, j] = 1.0 - iou
        return cost

    def update(self, detections, features):
        self.predict()
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        feat_arr = np.asarray(features) if len(features) > 0 else np.zeros((0, 2048), dtype=np.float32)

        if n_tracks > 0 and feat_arr.shape[0] > 0:
            for t in self.tracks:
                if len(t.features) == 0:
                    self.metric.samples.setdefault(t.track_id, [np.zeros(feat_arr.shape[1])])
                else:
                    self.metric.samples[t.track_id] = t.features[-self.metric.budget:] if hasattr(self.metric, 'budget') and self.metric.budget else t.features
            appearance_cost = self.metric.distance(feat_arr, [t.track_id for t in self.tracks])
        else:
            appearance_cost = np.zeros((n_tracks, feat_arr.shape[0]), dtype=np.float32)

        iou_cost = self._iou_cost(detections) if n_tracks > 0 else np.zeros((0, feat_arr.shape[0]), dtype=np.float32)

        if n_tracks > 0:
            cost_matrix = 0.6 * appearance_cost + 0.4 * iou_cost
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            matches = []
            assigned_tracks, assigned_dets = [], []
            for r, c in zip(row_idx, col_idx):
                if cost_matrix[r, c] > self.metric.matching_threshold:
                    continue
                matches.append((r, c))
                assigned_tracks.append(r); assigned_dets.append(c)
            unmatched_tracks = [i for i in range(n_tracks) if i not in assigned_tracks]
            unmatched_dets = [j for j in range(n_dets) if j not in assigned_dets]
        else:
            matches = []
            unmatched_tracks = []
            unmatched_dets = list(range(n_dets))

        for track_idx, det_idx in matches:
            tr = self.tracks[track_idx]; det = detections[det_idx]
            feat = features[det_idx] if det_idx < len(features) else None
            tr.update(det.tlwh, feat)

        for idx in unmatched_dets:
            det = detections[idx]
            feat = features[idx] if idx < len(features) else None
            new_tr = Track(det.tlwh, self._next_id, feat)
            self._next_id += 1
            self.tracks.append(new_tr)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        outs = []
        for t in self.tracks:
            x1, y1, x2, y2 = t.to_tlbr()
            outs.append([float(x1), float(y1), float(x2), float(y2), int(t.track_id), -1])
        return outs
