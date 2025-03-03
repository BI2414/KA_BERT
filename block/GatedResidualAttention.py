import torch.nn as nn
import torch
class GatedResidualAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, residual, attn_output):
        # 动态门控融合
        combined = torch.cat([residual, attn_output], dim=-1)
        gate = self.gate(combined)
        return residual * gate + attn_output * (1 - gate)