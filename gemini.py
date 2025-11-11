# moe_model.py (Final, Canonical ResNet)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# Attentive Knight (Final, Definitive Version)
# - Architecture: A pure, deep, canonical ResNet. No more experiments.
# - Correct Capacity: Uses 10 standard residual blocks with 256 channels,
#   resulting in ~11.8M parameters to fairly compete with the baseline.
# - Proven Speed: Leverages the simple, powerful, and fast structure of
#   a standard ResNet, which is what hardware is most optimized for.
# - Performance Components: Uses ReLU for speed and GroupNorm for stability.
# ======================================================================

# --- Hyperparameters ---
NUM_CHANNELS = 256  # High capacity to match the baseline
NUM_BLOCKS = 10     # Deep stack to match the baseline
INPUT_CHANNELS = 12 + 1 + 4

# --- Canonical Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(32, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.relu(out + residual)
        return out

class AttentiveKnight(nn.Module):
    def __init__(self, policy_channels=4672):
        super().__init__()
        
        # 1. Initial Convolution
        self.initial_conv = nn.Conv2d(INPUT_CHANNELS, NUM_CHANNELS, kernel_size=3, padding=1, bias=False)
        self.initial_gn = nn.GroupNorm(32, NUM_CHANNELS)
        self.initial_relu = nn.ReLU(inplace=True)
        
        # 2. Deep Residual Backbone
        self.backbone = nn.Sequential(*[ResidualBlock(NUM_CHANNELS) for _ in range(NUM_BLOCKS)])
        
        # 3. Decoupled Policy Head
        self.policy_conv = nn.Conv2d(NUM_CHANNELS, 73, kernel_size=1)

        # 4. Decoupled Value Head
        self.value_conv = nn.Conv2d(NUM_CHANNELS, 32, kernel_size=1, bias=False)
        self.value_gn = nn.GroupNorm(4, 32)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)
        
        # Prepare full input tensor
        side_to_move = x[:, 12, 0, 0].view(batch_size, 1, 1, 1).expand(-1, -1, 8, 8)
        castling_flags = x[:, 13:17, :, :]
        full_input = torch.cat([x[:, :12, :, :], side_to_move, castling_flags], dim=1)

        # --- Shared Backbone ---
        features = self.initial_relu(self.initial_gn(self.initial_conv(full_input)))
        features = self.backbone(features)

        # --- Policy Branch ---
        policy_logits = self.policy_conv(features).view(batch_size, -1)
        
        # --- Value Branch ---
        value_features = self.value_relu(self.value_gn(self.value_conv(features)))
        value_features_flat = value_features.view(batch_size, -1)
        value_out = self.value_relu(self.value_fc1(value_features_flat))
        value = self.value_tanh(self.value_fc2(value_out)).squeeze(-1)
        
        return F.log_softmax(policy_logits, dim=-1), value

def load_model(path=None, device="cpu"):
    model = AttentiveKnight(policy_channels=4672).to(device)
    if path and os.path.exists(path):
        print(f"[AttentiveKnight] loading model from {path}")
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print("[AttentiveKnight] initializing new model")
        
    return model