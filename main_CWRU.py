from src.model import CNNEncoder,FFT_MLP_Encoder,PTFM,DeepBDCHead,ViT1DEncoder,PTFM_NoFFT,PTFM_NoMixer,FCClassifierHead,prototypical_loss
from src.utils import FewShotDataset, visualize_tsne, split_by_labels,set_random_seed
from sklearn.metrics import f1_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import f1_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def evaluate_split(model, X, y, device, n_way, k_shot, label_set, name="Split", head=None):
    model.eval()
    dataset = FewShotDataset(X, y, n_way=n_way, k_shot=k_shot, q_query=15, episodes_per_epoch=100)
    loader = DataLoader(dataset, batch_size=1)
    total_acc = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            support_x, support_y, query_x, query_y = [b.squeeze(0).to(device) for b in batch]

            if head == "cls_model":
                logits = model(query_x)
                preds = logits.argmax(dim=1)
                acc = (preds == query_y).float().mean()

            else:
                support_embed = model(support_x)
                query_embed = model(query_x)

                if isinstance(head, nn.Module):
                    logits = head(support_embed, support_y, query_embed)
                    preds = logits.argmax(dim=1)
                    acc = (preds == query_y).float().mean()
                else:
                    prototypes = []
                    for i in range(n_way):
                        mask = (support_y == i)
                        prototypes.append(support_embed[mask].mean(dim=0))
                    prototypes = torch.stack(prototypes)
                    logits = -torch.cdist(query_embed, prototypes)
                    preds = logits.argmax(dim=1)
                    acc = (preds == query_y).float().mean()

            total_acc += acc.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(query_y.cpu().numpy())

    final_acc = total_acc / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    print(f"{name} Accuracy (n_way={n_way}): {final_acc:.4f}")
    print(f"{name} F1 Score (macro): {f1:.4f}")
    print(f"{name} Recall (macro): {recall:.4f}")
    return final_acc, f1, recall

def get_model(model_type, input_length=1500, num_classes=4):
    if model_type == "cnn":
        return CNNEncoder(), None
    elif model_type == "fft_mlp":
        return FFT_MLP_Encoder(), None
    elif model_type == "ptfm":
        return PTFM(input_length=input_length), None
    elif model_type == "deepbdc":
        return CNNEncoder(), DeepBDCHead(input_dim=128, output_dim=128)
    elif model_type == "vit_tiny":
        return ViT1DEncoder(input_length=input_length, patch_size=50, dim=128), None
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main_split(X_all_combined, y_all_combined, save_dir="./results", model_type="ptfm", seed=42):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_shot = 3
    q_query = 15
    episodes_per_epoch = 150
    epochs = 40

    seen_labels = [2, 3]
    unseen_labels = [0, 1]

    X_seen, y_seen = split_by_labels(X_all_combined, y_all_combined, seen_labels)
    X_unseen, y_unseen = split_by_labels(X_all_combined, y_all_combined, unseen_labels)

    encoder, head = get_model(model_type)
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    train_loader = DataLoader(
        FewShotDataset(X_seen, y_seen, n_way=2, k_shot=k_shot, q_query=q_query, episodes_per_epoch=episodes_per_epoch),
        batch_size=1,
        shuffle=True
    )

    print("Training on SEEN classes only...")
    for epoch in range(1, epochs + 1):
        encoder.train()
        total_loss, total_acc = 0, 0

        for batch in train_loader:
            support_x, support_y, query_x, query_y = [b.squeeze(0).to(device) for b in batch]
            support_embed = encoder(support_x)
            query_embed = encoder(query_x)

            loss, acc = prototypical_loss(support_embed, support_y, query_embed, query_y, n_way=2, k_shot=k_shot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc

        print(f"Epoch {epoch:02d} | Loss: {total_loss / len(train_loader):.4f} | Acc: {total_acc / len(train_loader):.4f}")

    print("\n--- Evaluation ---")
    seen_acc = evaluate_split(encoder, X_seen, y_seen, device, n_way=2, k_shot=k_shot, label_set=seen_labels, name="Seen")
    unseen_acc = evaluate_split(encoder, X_unseen, y_unseen, device, n_way=2, k_shot=k_shot, label_set=unseen_labels, name="Unseen")
    mixed_labels = seen_labels + unseen_labels
    mixed_acc = evaluate_split(encoder, X_all_combined, y_all_combined, device, n_way=4, k_shot=k_shot, label_set=mixed_labels, name="Mixed")