import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# Evaluation imports
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# Hyperparameters / constants
root = "../ShapeNetCore"  # path to dataset folder
val_size = 0.2
seed = 42
batch_size = 4
num_samples=3
random_checks=10

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BINVOX reader

def read_binvox(filepath):
    with open(filepath, 'rb') as f:
        line = f.readline().strip()
        if not line.startswith(b'#binvox'):
            raise ValueError("Not a binvox file")

        dims, translate, scale = None, None, None
        line = f.readline().strip()
        while line:
            parts = line.split()
            if parts[0] == b'dim':
                dims = tuple(map(int, parts[1:4]))
            elif parts[0] == b'translate':
                translate = tuple(map(float, parts[1:4]))
            elif parts[0] == b'scale':
                scale = float(parts[1])
            elif parts[0] == b'data':
                break
            line = f.readline().strip()

        raw_data = f.read()
        values = []
        i = 0
        while i < len(raw_data):
            value = raw_data[i]
            count = raw_data[i + 1]
            values.extend([value] * count)
            i += 2

        arr = np.asarray(values, dtype=np.uint8).reshape(dims)
        return arr.astype(np.float32)  # occupancy grid

# ----------------------
# Dataset class
# ----------------------
class ShapeNetBinvoxDataset(Dataset):
    def __init__(self, root_dir, transform=None, preload=False): 
        self.transform = transform
        self.preload = preload
        # find all .binvox files recursively
        self.files = glob.glob(os.path.join(root_dir, "**/*.binvox"), recursive=True)
        if len(self.files) == 0:
            raise RuntimeError(f"No .binvox files found under {root_dir}")

        # use category folder (02808440, etc.) as label
        #self.labels = [os.path.normpath(f).split(os.sep)[1] for f in self.files]
        self.labels = [os.path.normpath(f).split(os.sep)[-4] for f in self.files]
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.labels_idx = [self.class_to_idx[lbl] for lbl in self.labels]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vox = read_binvox(self.files[idx])
        vox = np.expand_dims(vox, axis=0)  # (1, D, H, W)
        if self.transform:
            vox = self.transform(vox)
        return torch.from_numpy(vox), self.labels_idx[idx]

# ====================================================
# Dataset sanity checks + visualization
# ====================================================

dataset = ShapeNetBinvoxDataset(root)

print("Classes:", dataset.classes)
print("Total samples:", len(dataset))

# Get one sample
vox, label = dataset[0]
print("Voxel shape:", vox.shape, "Label idx:", label, "Label name:", dataset.classes[label])

# DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
for x, y in loader:
    print("Batch voxel shape:", x.shape)  # (B,1,D,H,W)
    print("Batch labels:", y)
    break

# 2D slice visualization
plt.imshow(vox[0, vox.shape[1]//2])  # middle slice
plt.show()

# 3D visualization
vox, _ = dataset[0]
vox = vox[0].numpy()  # (D,H,W)

# Show a few slices along depth
for i in range(0, vox.shape[0], 16):  # every 16th slice
    plt.imshow(vox[i], cmap="viridis")
    plt.title(f"Slice {i}")
    plt.show()

# Downsample for visualization (from 128³ → 32³)
vox = torch.from_numpy(vox)
vox_down = F.interpolate(
    vox.unsqueeze(0).unsqueeze(0).float(), 
    size=(32, 32, 32), 
    mode="trilinear", 
    align_corners=False
)
vox_down = (vox_down > 0.5).squeeze().numpy()  # binarize, shape (32,32,32)

# Plot as voxels
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection="3d")
ax.voxels(vox_down, facecolors="blue", edgecolor="k")
plt.title(f"3D Voxel Visualization - Class {dataset.classes[label]}")
plt.show()

# Stratified train/val split
indices = list(range(len(dataset)))
labels = dataset.labels_idx
train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42)

train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

print("Train size:", len(train_idx), "Val size:", len(val_idx))
print("Unique train labels:", set([dataset.labels_idx[i] for i in train_idx]))
print("Unique val labels:", set([dataset.labels_idx[i] for i in val_idx]))

# Check voxels are not empty
for i in range(num_samples):
    vox, lbl = dataset[i]
    print(f"Sample {i} - Label: {lbl}, Voxel sum: {vox.sum().item()}")

# Check random samples
for i in random.sample(range(len(dataset)), random_checks):
    vox, lbl = dataset[i]
    print(f"Sample {i} | Label: {lbl} ({dataset.classes[lbl]}) "
    f"| Non-zeros: {(vox > 0).sum().item()}")

#3D CNN Model
import torch.nn as nn
class VoxelCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*16*16*16, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
#Train

model = VoxelCNN(num_classes=len(dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for voxels, labels in loader:
        voxels, labels = voxels.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(voxels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * voxels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for voxels, labels in loader:
            voxels, labels = voxels.to(device), labels.to(device)
            outputs = model(voxels)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * voxels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total
# Training loop
num_epochs = 20
best_val_acc = 0.0
target_acc = 0.98  # stop when val acc >= 98%

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
          f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_voxelcnn.pth")
        print(f"New best model saved (Val Acc: {val_acc:.4f})")

    # Early stopping condition
    if val_acc >= target_acc:
        print(f"Target accuracy {target_acc*100:.1f}% reached at epoch {epoch+1}. Stopping training.")
        break
        
print("Model saved to voxelcnn.pth")     
# Evaluate and Visualize
model = VoxelCNN(num_classes=len(dataset.classes)).to(device)
model.load_state_dict(torch.load("best_voxelcnn.pth", map_location=device))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for vox, labels in val_loader:
        vox, labels = vox.to(device), labels.to(device)
        outputs = model(vox)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Print classification report
print(classification_report(all_labels, all_preds, target_names=dataset.classes))


# Confusion Matrix

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=dataset.classes,
            yticklabels=dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png") 
plt.close()


# Per-Class Accuracy

correct = [0] * len(dataset.classes)
total = [0] * len(dataset.classes)

for y_true, y_pred in zip(all_labels, all_preds):
    total[y_true] += 1
    if y_true == y_pred:
        correct[y_true] += 1

acc_per_class = [c / t if t > 0 else 0 for c, t in zip(correct, total)]

plt.figure(figsize=(8, 6))
plt.bar(np.arange(len(dataset.classes)), acc_per_class)
plt.xticks(np.arange(len(dataset.classes)), dataset.classes, rotation=45)
plt.ylabel("Accuracy")
plt.xlabel("Class")
plt.title("Per-Class Accuracy")
plt.tight_layout()
plt.savefig("per_class_accuracy.png")  
plt.close()