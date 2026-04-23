import numpy as np
from scipy.spatial.distance import cdist

def cosine_distance(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
    return 1.0 - np.dot(a_norm, b_norm.T)

class NearestNeighborDistanceMetric:
    def __init__(self, matching_threshold=0.3, budget=None):
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for f, t in zip(features, targets):
            self.samples.setdefault(t, []).append(f)
            if self.budget is not None:
                self.samples[t] = self.samples[t][-self.budget:]
        self.samples = {k: v for k, v in self.samples.items() if k in active_targets}

    def distance(self, features, targets):
        if len(targets) == 0 or len(features) == 0:
            return np.zeros((len(targets), len(features)), dtype=np.float32)
        feats = np.asarray(features)
        cost = np.zeros((len(targets), feats.shape[0]), dtype=np.float32)
        for i, t in enumerate(targets):
            target_feats = np.vstack(self.samples.get(t, np.zeros((1, feats.shape[1]))))
            d = cdist(target_feats, feats, metric='cosine')
            cost[i, :] = d.min(axis=0)
        return cost
