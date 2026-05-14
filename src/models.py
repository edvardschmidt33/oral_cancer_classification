import torch
import torch.nn as nn
import timm


def create_early_fusion_model(model_name='convnextv2_nano.fcmae_ft_in22k_in1k',
                              pretrained=True, num_classes=1):
    """ConvNeXt V2 with a 6-channel stem for paired BF+FL early fusion.

    The first conv is replaced with a 6-channel conv whose first 3 input
    channels copy the pretrained RGB weights and whose last 3 copy them again
    so both modalities start from the same ImageNet prior.
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    old_conv = model.stem[0]
    old_weight = old_conv.weight.data  # [out_ch, 3, kH, kW]

    new_conv = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_weight
        new_conv.weight[:, 3:] = old_weight.clone()
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.stem[0] = new_conv
    return model


def set_backbone_frozen(model, frozen: bool):
    """Freeze/unfreeze everything except the classifier head."""
    head = model.get_classifier()
    head_params = {id(p) for p in head.parameters()}
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = not frozen
