import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import random
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


# ----------------------
# BINVOX reader
# ----------------------
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


#verify the dataset and visualize samples

if __name__ == "__main__":
    root = "../ShapeNetCore"  # path to dataset folder
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
    #2d slice visualization
    plt.imshow(vox[0, vox.shape[1]//2])  # middle slice
    plt.show()
    # 3d visualization
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



def get_splits(root="../ShapeNetCore", val_size=0.2, seed=42, batch_size=4):
    # Load dataset
    dataset = ShapeNetBinvoxDataset(root_dir="../ShapeNetCore")
    indices = list(range(len(dataset)))
    labels = dataset.labels_idx

    # Stratified train/val split
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return dataset, train_loader, val_loader, train_idx, val_idx

# random checks on the splits
def inspect_split(dataset, train_idx, val_idx, num_samples=3, random_checks=10):
    
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