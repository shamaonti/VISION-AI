# explainable_ai/grad_cam.py
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class SimpleGradCAM:
    """
    Grad-CAM for CNN action recognition.
    Pass the SAME model instance used for prediction and a conv layer (e.g., model.conv2).
    """
    def __init__(self, model, target_layer, device=None):
        self.model = model
        self.model.eval()
        self.device = device or next(model.parameters()).device

        self.activations = None
        self.gradients = None

        # Forward hook to save activations
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        # Backward hook to save gradients
        def bwd_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._fwd = target_layer.register_forward_hook(fwd_hook)
        self._bwd = target_layer.register_backward_hook(bwd_hook)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _to_tensor(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        return x

    def generate(self, frame_bgr, class_idx=None, alpha=0.4):
        """
        Returns: overlay_bgr, raw_cam (0..1)
        """
        x = self._to_tensor(frame_bgr)
        x.requires_grad_(True)

        # Forward
        logits = self.model(x)

        # Target class
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        # Backward
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        self.model.zero_grad(set_to_none=True)
        logits.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            return frame_bgr.copy(), None

        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()

        # Compute weights
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for c, w in enumerate(weights):
            cam += w * acts[c]

        cam = np.maximum(cam, 0)
        if cam.max() > 1e-8:
            cam /= (cam.max() + 1e-8)

        # Resize and overlay
        H, W = frame_bgr.shape[:2]
        cam_resized = cv2.resize(cam, (W, H))
        heat_uint8 = np.uint8(255 * cam_resized)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame_bgr, 1.0 - alpha, heat_color, alpha, 0)

        return overlay, cam_resized

    def close(self):
        try:
            self._fwd.remove()
            self._bwd.remove()
        except Exception:
            pass


def generate_gradcam(cam_instance, frame_bgr, class_idx=None, alpha=0.4, save_path=None):
    """
    Convenience function for live frames.
    cam_instance: pre-initialized SimpleGradCAM object
    """
    overlay, raw_cam = cam_instance.generate(frame_bgr, class_idx=class_idx, alpha=alpha)
    if save_path is not None:
        cv2.imwrite(save_path, overlay)
    return overlay, raw_cam
