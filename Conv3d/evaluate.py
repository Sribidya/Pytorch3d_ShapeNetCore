import torch
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from dataset_conv3d import ShapeNetBinvoxDataset
from Conv3dmodel import VoxelCNN
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# ----------------------
# Setup
# ----------------------
root = "ShapeNetCore"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Load dataset and validation set
# ----------------------
dataset = ShapeNetBinvoxDataset(root)  # load dataset
indices = list(range(len(dataset)))
labels = dataset.labels_idx
_, val_idx = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=42
)
# Here, val_loader should already exist if reusing from training
# Otherwise, create it using saved indices:
#import numpy as np
#val_idx = np.load("val_indices.npy", allow_pickle=True)

val_set = Subset(dataset, val_idx)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

# ----------------------
# Load model
# ----------------------
model = VoxelCNN(num_classes=len(dataset.classes)).to(device)
model.load_state_dict(torch.load("best_voxelcnn.pth", map_location=device))
model.eval()

# ----------------------
# Evaluation
# ----------------------
y_true, y_pred = [], []

with torch.no_grad():
    for voxels, labels in val_loader:
        voxels, labels = voxels.to(device), labels.to(device)
        outputs = model(voxels)
        _, preds = outputs.max(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ----------------------
# Results
# ----------------------
acc = accuracy_score(y_true, y_pred)
print(f"Validation Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=dataset.classes, zero_division=0))
