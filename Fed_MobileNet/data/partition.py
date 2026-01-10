import numpy as np
from torch.utils.data import Subset

def get_labels(dataset):
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)

    return np.array([dataset[i][1] for i in range(len(dataset))])

# Perfect partition (IID)
def IID_partition(dataset, num_clients=20, seed=845):
    rng = np.random.default_rng(seed)
    labels = get_labels(dataset)
    num_classes = int(labels.max() + 1)
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        splits = np.array_split(idx_c, num_clients)

        for k in range(num_clients):
            client_indices[k].extend(splits[k].tolist())

    for k in range(num_clients):
        rng.shuffle(client_indices[k])

    return [Subset(dataset, inds) for inds in client_indices]

def NIID_partition(dataset, num_clients=20, alpha=0.5, seed=845, min_size=10):
    rng = np.random.default_rng(seed)
    labels = get_labels(dataset)
    num_classes = int(labels.max() + 1)
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    while True:
        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            idx_c = class_indices[c]
            n_c = len(idx_c)
            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            counts = (proportions * n_c).astype(int)
            diff = n_c - counts.sum()

            if diff > 0:
                for k in np.argsort(-proportions)[:diff]:
                    counts[k] += 1
            start = 0

            for k in range(num_clients):
                end = start + counts[k]
                if end > start:
                    client_indices[k].extend(idx_c[start:end].tolist())
                start = end

        if min(len(ci) for ci in client_indices) >= min_size:
            break

    for k in range(num_clients):
        rng.shuffle(client_indices[k])

    return [Subset(dataset, inds) for inds in client_indices]

# Check data distribution
def print_label_counts(dataset, client_subsets, num_classes=10):
    labels = get_labels(dataset)

    for i, sub in enumerate(client_subsets):
        idx = sub.indices
        hist = np.bincount(labels[idx], minlength = num_classes)
        print(f"[client {i+1:03d}] n={len(idx)} counts={hist.tolist()}")