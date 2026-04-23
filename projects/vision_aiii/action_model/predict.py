import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from torchvision import models

# =========================
# Load Action Model
# =========================
def load_action_model(model_path, num_classes=2, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model (ResNet18 backbone)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    else:
        raise ValueError("❌ Invalid checkpoint file, expected state_dict.")

    model = model.to(device)
    model.eval()
    return model

# =========================
# Predict Action
# =========================
def predict_action(model, frame, class_names, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]

    return predicted_class
