# Copyright (c) Facebook, Inc. and its affiliates.

# Mostly copy-pasted from
# https://github.com/facebookresearch/detr/blob/master/models/backbone.py
"""
Backbone modules.
"""
from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from .misc import NestedTensor
from .position_encoding import build_position_encoding


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": 0}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        out = OrderedDict()
        for name, x in xs.items():
            mask = F.interpolate(
                tensor_list.mask[None].float(), size=x.shape[-2:]
            ).bool()[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True,
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        xs = self[0](tensor_list)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(
        args.backbone, train_backbone, return_interm_layers, args.dilation
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
