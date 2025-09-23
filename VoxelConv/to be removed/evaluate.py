# evaluate.py
from sklearn.metrics import classification_report, confusion_matrix
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Conv3dmodel import VoxelCNN
from dataset_conv3d import get_splits
from torch.utils.data import DataLoader

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset, _, val_loader, _, _ = get_splits(
    root="../ShapeNetCore", val_size=0.2, seed=42, batch_size=8
)

# Load saved classes to ensure consistency
with open("classes.json", "r") as f:
    class_names = json.load(f)

# Load model
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
