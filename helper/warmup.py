import math
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

# -----------------------------
# 1. Optimizer Configuration
# -----------------------------
# beta2=0.98 and eps=1e-9 stabilize Transformer attention updates
optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.98),
    eps=1e-9
)

# -----------------------------
# 2. Linear Warmup + Decay Schedule
# -----------------------------
# For a full dataset, the paper used 8,000 warmup steps.
# For small/toy datasets, pick a smaller number (e.g., 50-200 steps).
WARMUP_STEPS = 8000 

def lr_lambda(current_step: int):
    # Avoid zero division on the very first step
    if current_step < WARMUP_STEPS:
        # Linear ramp: 0 -> 1.0 over WARMUP_STEPS
        return float(current_step) / float(max(1, WARMUP_STEPS))
    
    # Inverse square-root decay (standard Vaswani / Attention Is All You Need decay)
    return max(0.0, (WARMUP_STEPS ** 0.5) * (current_step ** -0.5))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)