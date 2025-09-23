#Point Cloud Classification using PointNet on ShapeNet dataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from pytorch3d.datasets import ShapeNetCore # Import ShapeNetCore
from pytorch3d.datasets.utils import collate_batched_meshes # Import collate_batched_meshes
import torch
from pathlib import Path
import torch.nn as nn

# For evaluation graphs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Hyperparameters / constants
BATCH_SIZE = 4
HIDDEN_DIM = 64
LR = 1e-3
NUM_EPOCHS = 20
SHAPENET_PATH = Path("C:/Users/msrib/OneDrive - OsloMet/3D CNN/ShapeNetCore")

# Dynamically list all synset directories (WordNet IDs)
SYNSETS = [d.name for d in SHAPENET_PATH.iterdir() if d.is_dir()]


dataset = ShapeNetCore(SHAPENET_PATH,
    synsets=SYNSETS,
    version=2,
    load_textures=False # Set load_textures to False to avoid texture loading errors
)
# Print info
print(f"Total models: {len(dataset)}")
print(f"Total synsets: {len(SYNSETS)}")
print("First few synsets:", SYNSETS[:5])

# 80% training, 20% validation
# Map synset IDs to integer class indices
synset_to_idx = {synset: i for i, synset in enumerate(SYNSETS)}

# Generate integer labels for stratification
labels = [synset_to_idx[sample['synset_id']] for sample in dataset]

# Stratified train/val split
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=42
)

# Create Subsets
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

print(f"Total dataset size: {len(dataset)}")
print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Check sum: {len(train_dataset) + len(val_dataset)} == {len(dataset)}")

# Define a custom collate function to exclude textures and handle varying vertex counts
def custom_collate_fn(batch):
    verts_list = []
    synset_ids_list = []
    max_points = 700 # Fixed number of points for PointNet input

    for sample in batch:
        # Extract vertices
        verts = sample['verts']

        # Handle varying vertex counts by padding
        if verts.shape[0] < max_points:
            # Pad with zeros
            padding_size = max_points - verts.shape[0]
            padding = torch.zeros((padding_size, verts.shape[1]), dtype=verts.dtype, device=verts.device)
            padded_verts = torch.cat([verts, padding], dim=0)
            verts_list.append(padded_verts)
        else:
             verts_list.append(verts[:max_points]) # Simple truncation if > max_points

        # Extract synset_id
        synset_ids_list.append(sample['synset_id'])

    # Stack the padded vertices to create a batch of point clouds
    batched_verts = torch.stack(verts_list, dim=0)

    # Return a dictionary with the batched point clouds and synset_ids
    return {'verts': batched_verts, 'synset_id': synset_ids_list}


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)



class PointNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Shared MLP layers (applied point-wise)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Classification layers
        self.fc1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x is expected to be of shape (batch_size, num_points, in_channels)
        # PointNet expects input as (batch_size, in_channels, num_points)
        x = x.transpose(2, 1)

        # Apply shared MLPs
        x = self.mlp1(x)

        # Max pooling across points
        x = torch.max(x, 2)[0]

        # Apply classification layers
        x = self.fc1(x)
        x = self.fc2(x)

        return x
    import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Training loop functions
# ----------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        pc = batch['verts'].to(device)  # (B,P,3)
        labels = torch.tensor(
            [synset_to_idx[s] for s in batch['synset_id']],
            dtype=torch.long, device=device
        )

        optimizer.zero_grad()
        preds = model(pc)  # (B,C)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pc.size(0)
        correct += preds.argmax(1).eq(labels).sum().item()
        total += pc.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in loader:
            pc = batch['verts'].to(device)
            labels = torch.tensor(
                [synset_to_idx[s] for s in batch['synset_id']],
                dtype=torch.long, device=device
            )
            preds = model(pc)
            loss = criterion(preds, labels)

            total_loss += loss.item() * pc.size(0)
            correct += preds.argmax(1).eq(labels).sum().item()
            total += pc.size(0)

    return total_loss / total, correct / total


# ----------------------
# Main function
# ----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet(num_classes=len(SYNSETS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    all_y_true, all_y_pred = [], []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Collect predictions for validation set
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            pc = batch['verts'].to(device)
            labels = torch.tensor(
                [synset_to_idx[s] for s in batch['synset_id']],
                dtype=torch.long, device=device
            )
            logits = model(pc)
            preds = logits.argmax(dim=1)
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SYNSETS, yticklabels=SYNSETS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("PointCloud Confusion Matrix")
    plt.tight_layout()
    plt.savefig("pointCloud_confusion_matrix.png")
    plt.close()

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=SYNSETS, y=per_class_acc)
    plt.ylabel("Accuracy")
    plt.xlabel("Class")
    plt.title("PointCloud Per-class Accuracy")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("pointcloud_per_class_accuracy.png")
    plt.close()


if __name__ == "__main__":
    main()