import torch
import torchvision.transforms as T
from torchvision import models
import numpy as np
from PIL import Image

class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device).eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def extract(self, crops):
        if len(crops) == 0:
            return np.zeros((0, 2048), dtype=np.float32)
        imgs = [self.transform(Image.fromarray(c[:, :, ::-1])) for c in crops]
        batch = torch.stack(imgs).to(self.device)
        with torch.no_grad():
            feats = self.model(batch).cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6
        feats = feats / norms
        return feats.astype(np.float32)
