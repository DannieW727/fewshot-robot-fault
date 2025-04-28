import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.mlp(x)

class MixerBlock(nn.Module):
    def __init__(self, num_tokens, dim, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_mlp_dim),
            nn.GELU(),
            nn.Linear(token_mlp_dim, num_tokens)
        )

        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_mlp_dim),
            nn.GELU(),
            nn.Linear(channel_mlp_dim, dim)
        )

    def forward(self, x):
        y = x.permute(0, 2, 1)
        y = self.token_norm(y)
        y = self.token_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y

        y = self.channel_norm(x)
        y = self.channel_mlp(y)
        x = x + y
        return x


class PTFM(nn.Module):
    def __init__(self, input_length=1250, dim=128, mixer_tokens=10):
        super().__init__()
        self.input_length = input_length
        self.dim = dim
        self.mixer_tokens = mixer_tokens

        self.temporal_proj = nn.Linear(input_length, dim)

        self.fft_proj = nn.Linear(input_length // 2 + 1, dim)
        self.fusion_proj = nn.Linear(2, mixer_tokens)

        self.mixer = MixerBlock(num_tokens=mixer_tokens, dim=dim, token_mlp_dim=64, channel_mlp_dim=64)

        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mixer_tokens * dim, dim)
        )

    def forward(self, x):
        x = x.squeeze(1)
        h_time = self.temporal_proj(x)
        fft_feat = torch.fft.rfft(x, dim=1).abs()
        h_fft = self.fft_proj(fft_feat)
        h_stack = torch.stack([h_time, h_fft], dim=1)
        h_fused = self.fusion_proj(h_stack.permute(0, 2, 1)).permute(0, 2, 1)
        h_mixed = self.mixer(h_fused)
        out = self.output_proj(h_mixed)
        return out

def prototypical_loss(support, support_y, query, query_y, n_way, k_shot):
    prototypes = []
    for i in range(n_way):
        class_mask = support_y == i
        prototypes.append(support[class_mask].mean(dim=0))
    prototypes = torch.stack(prototypes)

    dists = torch.cdist(query, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p_y, query_y)
    acc = (log_p_y.argmax(dim=1) == query_y).float().mean()
    return loss, acc.item()


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.encoder(x).squeeze(-1)
        return x

class FFT_MLP_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(626, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = x.squeeze(1)
        x_fft = torch.fft.rfft(x, dim=1).abs()
        return self.encoder(x_fft)

class DeepBDCHead(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim * (input_dim + 1) // 2, output_dim)

    def forward(self, support_x, support_y, query_x):
        n_way = torch.unique(support_y).size(0)
        k_shot = support_x.size(0) // n_way
        emb_dim = support_x.size(-1)

        support = support_x.view(n_way, k_shot, emb_dim).mean(dim=1)  # (N, D)
        cov = torch.bmm(support.unsqueeze(2), support.unsqueeze(1))   # (N, D, D)
        tril_idx = torch.tril_indices(emb_dim, emb_dim)
        tril = cov[:, tril_idx[0], tril_idx[1]]                       # (N, D*(D+1)/2)
        proto = self.fc(tril)                                        # (N, output_dim)

        query_proj = query_x                                          # (Q, D)
        logits = -torch.cdist(query_proj, proto)                     # (Q, N)
        return logits

class ViT1DEncoder(nn.Module):
    def __init__(self, input_length=1250, patch_size=50, dim=128, depth=2, heads=4):
        super().__init__()
        assert input_length % patch_size == 0, "Input length must be divisible by patch size"
        self.n_patches = input_length // patch_size
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_head = nn.Linear(dim, dim)  # same dim output for ProtoNet

    def forward(self, x):
        # x: (B, 1, 1250)
        B = x.size(0)
        x = x.squeeze(1).unfold(1, self.patch_size, self.patch_size)  # (B, n_patches, patch_size)
        x = self.patch_embed(x) + self.pos_embed                      # (B, n_patches, dim)
        x = self.transformer(x)                                       # (B, n_patches, dim)
        x = x.mean(dim=1)                                             # (B, dim)
        return self.cls_head(x)

#1
class PTFM_NoFFT(nn.Module):
    def __init__(self, input_length=1250, dim=128, mixer_tokens=10):
        super().__init__()
        self.input_length = input_length
        self.dim = dim
        self.mixer_tokens = mixer_tokens

        self.temporal_proj = nn.Linear(input_length, dim)

        self.token_proj = nn.Linear(dim, dim)

        self.mixer = MixerBlock(num_tokens=mixer_tokens, dim=dim, token_mlp_dim=64, channel_mlp_dim=64)
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mixer_tokens * dim, dim)
        )

    def forward(self, x):  # (B, 1, 1250)
        x = x.squeeze(1)
        h_time = self.temporal_proj(x)  # (B, dim)
        h_repeat = h_time.unsqueeze(1).repeat(1, self.mixer_tokens, 1)  # (B, mixer_tokens, dim)
        h_proj = self.token_proj(h_repeat)
        h_mixed = self.mixer(h_proj)
        out = self.output_proj(h_mixed)
        return out

#2
class SimpleConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )

    def forward(self, x):  # (B, num_tokens, dim)
        x = x.permute(0, 2, 1)  # (B, dim, tokens)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x
    
class PTFM_NoMixer(nn.Module):
    def __init__(self, input_length=1250, dim=128, mixer_tokens=10):
        super().__init__()
        self.temporal_proj = nn.Linear(input_length, dim)
        self.fft_proj = nn.Linear(input_length // 2 + 1, dim)
        self.fusion_proj = nn.Linear(2, mixer_tokens)

        self.conv_block = SimpleConvBlock(dim=dim)
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mixer_tokens * dim, dim)
        )

    def forward(self, x):
        x = x.squeeze(1)
        h_time = self.temporal_proj(x)
        h_fft = self.fft_proj(torch.fft.rfft(x, dim=1).abs())
        h_stack = torch.stack([h_time, h_fft], dim=1)
        h_fused = self.fusion_proj(h_stack.permute(0, 2, 1)).permute(0, 2, 1)
        h_conv = self.conv_block(h_fused)
        return self.output_proj(h_conv)

#3
class FCClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
