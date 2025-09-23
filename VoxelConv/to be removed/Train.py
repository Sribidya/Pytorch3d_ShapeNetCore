import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset_conv3d import  ShapeNetBinvoxDataset  # import your Dataset class
from Conv3dmodel import VoxelCNN  # import the CNN class
import torch.optim as optim
import torch.nn as nn
import random
import json
from dataset_conv3d import get_splits, inspect_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset, train_loader, val_loader, train_idx, val_idx = get_splits(
    root="../ShapeNetCore", val_size=0.2, seed=42, batch_size=2
)

inspect_split(dataset, train_idx, val_idx)

# Create model
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
num_epochs = 2
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
        with open("classes.json", "w") as f:
          json.dump(dataset.classes, f)  # save class names
        print(f"New best model saved (Val Acc: {val_acc:.4f})")

    # Early stopping condition
    if val_acc >= target_acc:
        print(f"Target accuracy {target_acc*100:.1f}% reached at epoch {epoch+1}. Stopping training.")
        break
        
print("Model saved to voxelcnn.pth")       