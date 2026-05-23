import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import LayerNorm2d


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


### Cros-Attention Network models

class ModalityProjection(nn.Module):
    """Lifts a 3-channel modality to a `out_ch`-dim feature space at full
    spatial resolution. Learned from scratch.
    Here, it expands the channels to 64 from 3 and pereserves the spatial dimensions (128x128)"""
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class LocalCrossAttention(nn.Module):
    """Per-pixel cross-attention between two modalities with a local
    `window_size`x`window_size` neighborhood. Q from one modality attends to
    K/V from the other modality within the local window around each spatial
    position. Symmetric: both directions are computed and returned with
    residual connections."""

    def __init__(self, dim=64, num_heads=4, window_size=3):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.pad = window_size // 2

        self.q_bf = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_bf = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_bf = nn.Conv2d(dim, dim, 1, bias=False)

        self.q_fl = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_fl = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_fl = nn.Conv2d(dim, dim, 1, bias=False)

        self.out_bf = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_fl = nn.Conv2d(dim, dim, 1, bias=False)

        self.rel_pos_bias = nn.Parameter(
            torch.zeros(num_heads, window_size * window_size)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def _to_heads(self, x, B, H, W):
        return x.view(B, self.num_heads, self.head_dim, H, W)

    def _unfold_local(self, x, B, H, W):
        """x: [B, heads, head_dim, H, W] -> [B, heads, head_dim, win*win, H, W]"""
        Bh = B * self.num_heads
        x_flat = x.reshape(Bh, self.head_dim, H, W)
        x_pad = F.pad(x_flat, [self.pad] * 4, mode='reflect')
        x_unf = x_pad.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        x_unf = x_unf.reshape(Bh, self.head_dim, H, W, -1)
        x_unf = x_unf.permute(0, 1, 4, 2, 3)
        return x_unf.reshape(B, self.num_heads, self.head_dim, -1, H, W)

    def _local_cross_attn(self, q, k_unf, v_unf):
        """q: [B, heads, head_dim, H, W]; k_unf/v_unf: [B, heads, head_dim, win*win, H, W]"""
        q = q.unsqueeze(3)
        attn = (q * k_unf).sum(dim=2) * self.scale            # [B, heads, win*win, H, W]
        attn = attn + self.rel_pos_bias.view(1, self.num_heads, -1, 1, 1)
        attn = attn.softmax(dim=2)
        out = (attn.unsqueeze(2) * v_unf).sum(dim=3)          # [B, heads, head_dim, H, W]
        return out

    def forward(self, f_bf, f_fl):
        B, C, H, W = f_bf.shape

        q_bf = self._to_heads(self.q_bf(f_bf), B, H, W)
        k_bf = self._to_heads(self.k_bf(f_bf), B, H, W)
        v_bf = self._to_heads(self.v_bf(f_bf), B, H, W)

        q_fl = self._to_heads(self.q_fl(f_fl), B, H, W)
        k_fl = self._to_heads(self.k_fl(f_fl), B, H, W)
        v_fl = self._to_heads(self.v_fl(f_fl), B, H, W)

        k_fl_unf = self._unfold_local(k_fl, B, H, W)
        v_fl_unf = self._unfold_local(v_fl, B, H, W)
        k_bf_unf = self._unfold_local(k_bf, B, H, W)
        v_bf_unf = self._unfold_local(v_bf, B, H, W)

        a_bf_to_fl = self._local_cross_attn(q_bf, k_fl_unf, v_fl_unf)
        a_fl_to_bf = self._local_cross_attn(q_fl, k_bf_unf, v_bf_unf)

        a_bf_to_fl = a_bf_to_fl.reshape(B, C, H, W)
        a_fl_to_bf = a_fl_to_bf.reshape(B, C, H, W)

        f_bf_enhanced = f_bf + self.out_bf(a_fl_to_bf)
        f_fl_enhanced = f_fl + self.out_fl(a_bf_to_fl)
        return f_bf_enhanced, f_fl_enhanced


class CrossAttentionFusionModel(nn.Module):
    """Single-backbone early-fusion model with pixel-level cross-attention.

    Pipeline: per-modality 3->`proj_dim` projection -> local cross-attention ->
    concat+1x1 conv -> stem-bypass projection (kernel=4, stride=4) ->
    ConvNeXt V2 backbone (stem replaced) -> head.
    """

    def __init__(self, backbone_name='convnextv2_tiny.fcmae_ft_in22k_in1k',
                 pretrained=True, proj_dim=64, num_heads=4, window_size=3):
        super().__init__()

        self.bf_proj = ModalityProjection(in_ch=3, out_ch=proj_dim)
        self.fl_proj = ModalityProjection(in_ch=3, out_ch=proj_dim)

        self.cross_attn = LocalCrossAttention(
            dim=proj_dim, num_heads=num_heads, window_size=window_size,
        )

        self.fusion_proj = nn.Sequential(
            nn.Conv2d(proj_dim * 2, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
        )

        self.stem_bypass = nn.Sequential(
            nn.Conv2d(proj_dim, 96, kernel_size=4, stride=4, bias=False),
            LayerNorm2d(96),
        )

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0,
        )
        self.backbone.stem = nn.Identity()
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.4), #higher dropout due to larger risk of overfitting
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3), #lower dropout, lower OF risk
            nn.Linear(256, 1),
        )

    def forward(self, bf, fl):
        f_bf = self.bf_proj(bf)
        f_fl = self.fl_proj(fl)

        f_bf_enh, f_fl_enh = self.cross_attn(f_bf, f_fl)

        fused = self.fusion_proj(torch.cat([f_bf_enh, f_fl_enh], dim=1))
        x = self.stem_bypass(fused)
        x = self.backbone(x)
        return self.head(x)

    def set_freeze_strategy(self, strategy: str):
        """Backbone freeze control. The fusion module + head are always trainable.
        - 'warmup':  backbone fully frozen
        - 'partial': only backbone stages 2-3 + norms trainable
        - 'none':    backbone fully unfrozen
        """
        for module in [self.bf_proj, self.fl_proj, self.cross_attn,
                       self.fusion_proj, self.stem_bypass, self.head]:
            for p in module.parameters():
                p.requires_grad = True

        for name, p in self.backbone.named_parameters():
            if strategy == 'warmup':
                p.requires_grad = False
            elif strategy == 'partial':
                p.requires_grad = any(k in name for k in ['stages.2', 'stages.3', 'norm'])
            elif strategy == 'none':
                p.requires_grad = True
            else:
                raise ValueError(f"unknown freeze strategy: {strategy!r}")