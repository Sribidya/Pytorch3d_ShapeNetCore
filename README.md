# 3D Deep Learning on ShapeNet Dataset

This README provides setup and usage instructions for three 3D deep learning approaches on the **ShapeNet dataset** using PyTorch3D: Voxel CNN, GraphConv Mesh Classifier, and PointNet for Point Clouds.

---

## 1. Prerequisites

* **Python 3.9.23**
* **CUDA 12.8** (ensure PATH is set)
* **Visual Studio 2022 / Build Tools** with C++ module
* Install PyTorch and PyTorch3D:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
SET DISTUTILS_USE_SDK=1
python setup.py install
```

* Optional packages:

```bash
conda install -c conda-forge tabulate matplotlib seaborn scikit-learn
```

---

## 2. Dataset Setup

* Path to ShapeNet:

```python
SHAPENET_PATH = Path("../ShapeNet")
SYNSETS = [d.name for d in SHAPENET_PATH.iterdir() if d.is_dir()]
```

* Load dataset:

```python
from pytorch3d.datasets import ShapeNetCore
dataset = ShapeNetCore(SHAPENET_PATH, synsets=SYNSETS, version=2, load_textures=False)
```


## 3. Voxel CNN

* **Dataset**: `ShapeNetBinvoxDataset` reads `.binvox` files.
* **Model**: 3D CNN with Conv3D → MaxPool → FC layers.
* **Training**: CrossEntropyLoss + Adam, early stopping at target accuracy.
* **Evaluation**: Confusion matrix and per-class accuracy plots.
* DataLoader batch size: 4 (adjust for GPU).

---

## 4. GraphConv Mesh Classifier

* **Dataset**: `ShapeNetCore` meshes.
* **Collate**: `collate_batched_meshes` excluding textures.
* **Model**: 3 GraphConv layers + FC classifier, global average pooling per mesh.
* **Training**: CrossEntropyLoss + Adam.
* **Evaluation**: Confusion matrix, per-class accuracy.
* DataLoader batch size: 4.

---

## 5. PointNet for Point Clouds

* **Dataset**: `ShapeNetCore` meshes converted to point clouds (vertices).
* **Collate**: Pad or truncate points to fixed number (e.g., 700).
* **Model**: PointNet with shared MLP + max pooling + FC layers.
* **Training**: CrossEntropyLoss + Adam.
* **Evaluation**: Confusion matrix, per-class accuracy.
* DataLoader batch size: 4.

---

## 6. Common Notes & Tips

* Adjust batch size and max points according to GPU memory.
* Validate batch contents before training.
* Stratified splitting ensures balanced class distribution.
* Save models after best validation accuracy.
* Use `matplotlib` and `seaborn` to visualize confusion matrices and accuracy.

---

## 7. References

* [PyTorch3D GitHub](https://github.com/facebookresearch/pytorch3d)
* [ShapeNetCore Dataset](https://www.shapenet.org/)
* [PointNet Paper](https://arxiv.org/abs/1612.00593)
