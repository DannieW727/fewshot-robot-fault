import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import os

import random

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_by_labels(X, y, include_labels):
    mask = np.isin(y, include_labels)
    return X[mask], y[mask]

class FewShotDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, n_way=5, k_shot=1, q_query=15, episodes_per_epoch=1000, fixed_classes=None):
        self.X = X
        self.y = y
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch

        self.class_to_indices = {}
        for label in np.unique(y):
            indices = np.where(y == label)[0]
            self.class_to_indices[label] = indices

        if fixed_classes:
            self.classes = fixed_classes
        else:
            self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return self.episodes_per_epoch

    def __getitem__(self, idx):
        if len(self.classes) < self.n_way:
            raise ValueError(f"Requested n_way={self.n_way}, but only {len(self.classes)} classes available.")

        selected_classes = random.sample(self.classes, self.n_way)
        support_x, support_y, query_x, query_y = [], [], [], []

        for i, cls in enumerate(selected_classes):
            indices = self.class_to_indices[cls]
            selected = np.random.choice(indices, self.k_shot + self.q_query, replace=False)
            support_indices = selected[:self.k_shot]
            query_indices = selected[self.k_shot:]

            support_x.append(self.X[support_indices])
            support_y.extend([i] * self.k_shot)
            query_x.append(self.X[query_indices])
            query_y.extend([i] * self.q_query)

        support_x = torch.tensor(np.concatenate(support_x, axis=0), dtype=torch.float32)
        query_x = torch.tensor(np.concatenate(query_x, axis=0), dtype=torch.float32)
        support_y = torch.tensor(support_y, dtype=torch.long)
        query_y = torch.tensor(query_y, dtype=torch.long)

        return support_x, support_y, query_x, query_y
    
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_tsne(model, X, y, device, label_set, model_name="Model"):
    model.eval()
    support_embed_list, query_embed_list = [], []
    support_label_list, query_label_list = [], []

    dataset = FewShotDataset(X, y, n_way=len(label_set), k_shot=3, q_query=15, episodes_per_epoch=1)
    support_x, support_y, query_x, query_y = [b.squeeze(0).to(device) for b in next(iter(DataLoader(dataset)))]

    with torch.no_grad():
        support_embed = model(support_x)
        query_embed = model(query_x)

    support_embed_np = support_embed.cpu().numpy()
    query_embed_np = query_embed.cpu().numpy()
    support_y_np = support_y.cpu().numpy()
    query_y_np = query_y.cpu().numpy()

    X_all = np.concatenate([support_embed_np, query_embed_np], axis=0)
    y_all = np.concatenate([support_y_np, query_y_np], axis=0)
    is_support = np.array(['Support'] * len(support_embed_np) + ['Query'] * len(query_embed_np))

    tsne = TSNE(n_components=2, random_state=42, perplexity=20)
    X_tsne = tsne.fit_transform(X_all)

    plt.figure(figsize=(8, 6))
    for label in np.unique(y_all):
        idx_s = (is_support == 'Support') & (y_all == label)
        idx_q = (is_support == 'Query') & (y_all == label)
        plt.scatter(X_tsne[idx_s, 0], X_tsne[idx_s, 1], label=f"Class {label} (Support)", marker='o', s=90, alpha=0.7)
        plt.scatter(X_tsne[idx_q, 0], X_tsne[idx_q, 1], label=f"Class {label} (Query)", marker='^', s=90, alpha=0.7)

    plt.legend(fontsize=18)
    plt.xlabel("Dim 1", fontsize=18)
    plt.ylabel("Dim 2", fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.title(f"t-SNE Visualization of {model_name}", fontsize=18)
    plt.tight_layout()
    plt.show()
