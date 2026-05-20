import torch
import torch.nn as nn
import timm

class GatedFusionModel(nn.Module):
    """Two ConvNeXt V2 encoders (BF + FL) with a GMU (Arevalo et al., 2017)
    fusing their features via tanh projections and a sigmoid gate.
    """
    def __init__(self, model_name='convnextv2_nano.fcmae_ft_in22k_in1k',
                 pretrained=True):
        super().__init__()
        self.bf_enc = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.fl_enc = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        dim = self.bf_enc.num_features  # 640 for Nano

        # Modality projections (tanh activation, as in the GMU paper)
        self.bf_proj = nn.Linear(dim, dim)
        self.fl_proj = nn.Linear(dim, dim)

        # Gate: takes both raw features, outputs sigmoid weighting
        self.gate = nn.Linear(dim * 2, dim)

        # Classification head (now 640-dim input instead of 1280)
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(0.3),
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, bf, fl):
        f_bf = self.bf_enc(bf)
        f_fl = self.fl_enc(fl)

        # GMU: project through tanh, gate with sigmoid, weighted sum
        h_bf = torch.tanh(self.bf_proj(f_bf))
        h_fl = torch.tanh(self.fl_proj(f_fl))
        z = torch.sigmoid(self.gate(torch.cat([f_bf, f_fl], dim=1)))
        fused = z * h_bf + (1 - z) * h_fl  # [B, 640]

        return self.head(fused)

    def set_encoders_frozen(self, frozen: bool):
        for p in self.bf_enc.parameters():
            p.requires_grad = not frozen
        for p in self.fl_enc.parameters():
            p.requires_grad = not frozen