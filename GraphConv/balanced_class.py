# analyze_classes.py

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def analyze_class_distribution(dataset, key='synset_id', threshold=0.2, plot=True):
    """
    Analyze class/synset distribution in a dataset.

    Args:
        dataset: Dataset object (ShapeNetCore or custom Dataset)
        key: Attribute or dict key for class labels ('synset_id' or 'labels_idx')
        threshold: Relative difference threshold to consider imbalance
        plot: Whether to show a bar plot

    Returns:
        class_counts: dict of class -> count
        recommendation: string recommending stratified or random split
    """
    # Extract labels
    if hasattr(dataset, 'labels_idx'):  # e.g., voxel dataset
        all_labels = dataset.labels_idx
    else:  # assume PyTorch3D ShapeNetCore
        all_labels = [sample[key] for sample in dataset]

    class_counts = Counter(all_labels)
    counts = np.array(list(class_counts.values()))
    max_count, min_count = counts.max(), counts.min()
    rel_diff = (max_count - min_count) / max_count

    recommendation = ("Classes are imbalanced." if rel_diff > threshold
                      else "Classes are roughly balanced.")

    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xticks(rotation=45)
        plt.ylabel("Number of samples")
        plt.title("Class/Synset distribution")
        plt.show()

    print(f"Total classes: {len(class_counts)}")
    print(f"Max samples: {max_count}, Min samples: {min_count}, Relative diff: {rel_diff:.2f}")
    print("Recommendation:", recommendation)

    return class_counts, recommendation


# Example usage for PyTorch3D ShapeNetCore
if __name__ == "__main__":
    from pathlib import Path
    from pytorch3d.datasets import ShapeNetCore

    SHAPENET_PATH = Path("C:/Users/msrib/OneDrive - OsloMet/3D CNN/ShapeNetCore")
    SYNSETS = [d.name for d in SHAPENET_PATH.iterdir() if d.is_dir()]

    dataset = ShapeNetCore(
        SHAPENET_PATH,
        synsets=SYNSETS,
        version=2,
        load_textures=False
    )

    # Analyze class distribution
    class_counts, recommendation = analyze_class_distribution(dataset)
