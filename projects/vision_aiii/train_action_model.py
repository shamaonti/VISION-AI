import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix

# =====================
# Custom Dataset (recursive)
# =====================
class KnifeDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.lbl_dir = self.root / "labels"
        self.transform = transform
        # Use rglob to find images in subfolders (knife, no_knife)
        self.images = list(self.img_dir.rglob("*.jpg")) + list(self.img_dir.rglob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # labels subfolder must mirror images subfolder structure
        label_path = self.lbl_dir / img_path.relative_to(self.img_dir).with_suffix(".txt")

        # Default label = no_knife (1)
        label = 1
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls = int(line.split()[0])
                    if cls == 0:  # knife present
                        label = 0
                        break

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# =====================
# Paths & Config
# =====================
output_dir = Path("action_model")
output_dir.mkdir(exist_ok=True)
checkpoint_path = output_dir / "checkpoint.pth"
best_model_path = output_dir / "best_action_model.pth"

# =====================
# Transforms
# =====================
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =====================
# Load datasets
# =====================
train_dataset = KnifeDataset("data/train", transform=train_transform)
val_dataset   = KnifeDataset("data/valid", transform=val_test_transform)
test_dataset  = KnifeDataset("data/test", transform=val_test_transform)

print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}, Test images: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

# =====================
# Model
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: knife, no_knife
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =====================
# Resume from checkpoint if exists
# =====================
start_epoch = 0
best_val_acc = 0.0
if checkpoint_path.exists():
    print("🔄 Resuming from checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    print(f"Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.2f}")

# =====================
# Training loop
# =====================
total_epochs = 200
patience = 10
counter = 0

for epoch in range(start_epoch, total_epochs):
    # Train
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    val_acc = correct / total

    print(f"Epoch [{epoch+1}/{total_epochs}] — Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}")

    # Save checkpoint every epoch
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc
    }, checkpoint_path)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        counter = 0
        print("✅ Best model saved!")
    else:
        counter += 1
        if counter >= patience:
            print(f"⏱ Early stopping triggered at epoch {epoch+1}")
            break

# =====================
# Final Test Accuracy
# =====================
model.load_state_dict(torch.load(best_model_path))
model.eval()
all_labels = []
all_preds = []
correct, total = 0, 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

print(f"🔥 Test Accuracy: {correct/total:.2f}")
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# =====================
# Save final model
# =====================
final_model_path = output_dir / "final_action_model.pth"
torch.save(model.state_dict(), final_model_path)
print(f"✅ Final model saved at {final_model_path}")
