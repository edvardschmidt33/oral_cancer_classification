import torch
import torch.nn as nn
import timm

class GatedFusionModel(nn.Module):
    """Two ConvNeXt V2 encoders (BF + FL). Features are gated with a GMU
    (Arevalo et al., 2017), then the gated branches are concatenated and
    passed through a small classification head.
    """
    def __init__(self, model_name='convnextv2_nano.fcmae_ft_in22k_in1k',
                 pretrained=True):
        super().__init__()
        self.bf_enc = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.fl_enc = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        dim = self.bf_enc.num_features  # 640 for Nano

        self.bf_proj = nn.Linear(dim, dim)
        self.fl_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, dim)

        self.head = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Dropout(0.4),
            nn.Linear(dim * 2, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, bf, fl):
        f_bf = self.bf_enc(bf)
        f_fl = self.fl_enc(fl)

        h_bf = torch.tanh(self.bf_proj(f_bf))
        h_fl = torch.tanh(self.fl_proj(f_fl))
        z = torch.sigmoid(self.gate(torch.cat([f_bf, f_fl], dim=1)))
        fused = torch.cat([z * h_bf, (1 - z) * h_fl], dim=1)  # [B, 2*dim]
        return self.head(fused)

    def set_encoders_freeze_strategy(self, strategy: str):
        """Freeze strategy for both encoders.
        - 'full':    freeze everything (warmup phase)
        - 'partial': only stages 2-3 and norms are trainable (fine-tune phase)
        - 'none':    unfreeze all
        """
        for enc in [self.bf_enc, self.fl_enc]:
            for name, p in enc.named_parameters():
                if strategy == 'full':
                    p.requires_grad = False
                elif strategy == 'partial':
                    p.requires_grad = any(k in name for k in ['stages.2', 'stages.3', 'norm'])
                elif strategy == 'none':
                    p.requires_grad = True
                else:
                    raise ValueError(f"unknown freeze strategy: {strategy!r}")