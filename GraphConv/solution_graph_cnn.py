#conda install -c conda-forge tabulate
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split 
from pytorch3d.datasets.utils import collate_batched_meshes
from pytorch3d.datasets import ShapeNetCore
from pytorch3d.ops import GraphConv
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


# For evaluation graphs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Hyperparameters / constants
BATCH_SIZE = 4
#BATCH_SIZE = 16 # increased batch size for better GPU utilization
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

def custom_collate_fn(batch):
    # Filter out the 'textures' key from each dictionary in the batch list
    filtered_batch = [{key: value for key, value in sample.items() if key != 'textures'} for sample in batch]
    # Use PyTorch3D's collate_batched_meshes on the filtered batch
    return collate_batched_meshes(filtered_batch)


# DataLoaders with custom collate_fn
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)


# Inspect a batch

def validate_batch(loader, name="Train"):
    batch = next(iter(loader))  
    # batch is a dict in your version
    print(f"\n=== {name} batch ===")
    print("Type of batch:", type(batch))
    print("Batch keys:", batch.keys()) 
   # see what keys are present
    # Display some batch info
    print("\nSample synset IDs:", batch['synset_id'][:5])
    print("Sample labels:", batch['label'][:5])
    print("Sample model IDs:", batch['model_id'][:5])
    # Extract the first object that is a Meshes instance
    meshes = None
    for v in batch.values():
        if isinstance(v, torch.nn.Module) or hasattr(v, "verts_packed"):
            meshes = v
            break
    if meshes is None:
        raise ValueError("No Meshes object found in the batch")

    # Inspect mesh
    print("Type of meshes:", type(meshes))
    print("Packed vertices shape:", meshes.verts_packed().shape)
    print("Packed faces shape:", meshes.faces_packed().shape)
    print("Vertices per mesh:", meshes.num_verts_per_mesh())
    print("Faces per mesh:", meshes.num_faces_per_mesh())


# Validate train and validation batches
validate_batch(train_loader, "Train")
validate_batch(val_loader, "Validation")

#####################
#GraphConv classifier
#####################

class MeshClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=HIDDEN_DIM, num_classes=10):
        super().__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, meshes):
        verts = meshes.verts_packed()
        edges = meshes.edges_packed()
        x = torch.relu(self.conv1(verts, edges))
        x = torch.relu(self.conv2(x, edges))
        x = torch.relu(self.conv3(x, edges))

        # global average pooling per mesh
        mesh_feats = []
        start = 0
        for n in meshes.num_verts_per_mesh():
            pooled = x[start:start+n].mean(dim=0)
            mesh_feats.append(pooled)
            start += n
        mesh_feats = torch.stack(mesh_feats)
        return self.fc(mesh_feats)

###############################
# Training and validation
###############################

# Map synset IDs to integer class indices (outside the functions)
synset_to_idx = {synset: i for i, synset in enumerate(SYNSETS)}

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        meshes = batch['mesh'].to(device)
        labels = torch.tensor([synset_to_idx[s] for s in batch['synset_id']],
                      dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(meshes)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for batch in loader:
        meshes = batch['mesh'].to(device)
        labels = torch.tensor([synset_to_idx[s] for s in batch['synset_id']],
                      dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(meshes)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return total_loss / len(loader.dataset), correct / total

###########################
# Main
###########################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeshClassifier(num_classes=len(SYNSETS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    all_y_true, all_y_pred = [], []

    for epoch in range(NUM_EPOCHS):
        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        # Logging
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ----------------------
    # Collect predictions for entire validation set
    # ----------------------
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            meshes = batch['mesh'].to(device)
            labels = torch.tensor([synset_to_idx[s] for s in batch['synset_id']],
                      dtype=torch.long, device=device)
            logits = model(meshes)
            preds = logits.argmax(dim=1)
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())

    # ----------------------
    # Confusion Matrix
    # ----------------------
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SYNSETS, yticklabels=SYNSETS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("GraphCNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig("graphcnn_confusion_matrix.png") 
    

    # ----------------------
    # Per-class accuracy
    # ----------------------
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=SYNSETS, y=per_class_acc)
    plt.ylabel("Accuracy")
    plt.xlabel("Class")
    plt.title("GraphCNN Per-class Accuracy")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("graphcnn_per_class_accuracy.png")
   
    
if __name__ == "__main__":
    main()