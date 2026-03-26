# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "MyC2f",
    "MySPPF",
    "DHAF",
    "DAFM",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """
        Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """
        Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """
        Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
            self,
            c1: int,
            cm: int,
            c2: int,
            k: int = 3,
            n: int = 6,
            lightconv: bool = False,
            shortcut: bool = False,
            act: nn.Module = nn.ReLU(),
    ):
        """
        Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: Tuple[int, ...] = (5, 9, 13)):
        """
        Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """
        Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """
        Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """
        Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
            self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """
        Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """
        Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """
        Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc ** 0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
            self,
            c1: int,
            c2: int,
            n: int = 1,
            ec: int = 128,
            nh: int = 1,
            gc: int = 512,
            shortcut: bool = False,
            g: int = 1,
            e: float = 0.5,
    ):
        """
        Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
            self, ec: int = 256, ch: Tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """
        Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: List[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ImagePoolingAttn.

        Args:
            x (List[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k ** 2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc ** 0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """
        Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes input out unchanged."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
            self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """
        Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """
        Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """
        Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """
        Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """
        Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: List[int], k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1):
        """
        Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (List[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: List[int]):
        """
        Initialize CBFuse module.

        Args:
            idx (List[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through CBFuse layer.

        Args:
            xs (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
            self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """
        Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """
        Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
            self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """
        Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """
        Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """
        Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(
            self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """
        Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | List[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """
        Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
            self,
            c1: int,
            c2: int,
            n: int = 1,
            a2: bool = True,
            area: int = 1,
            residual: bool = False,
            mlp_ratio: float = 2.0,
            e: float = 0.5,
            g: int = 1,
            shortcut: bool = True,
    ):
        """
        Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """
        Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """
        Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: List[int], c3: int, embed: int):
        """
        Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (List[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: List[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min

        score = F.softmax(score, dim=-1, dtype=torch.float).to(score.dtype)

        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class MyC2f(nn.Module):
    """
    A C2f module with fully shared weights to extract common features
    from two parallel input streams (e.g., RGB and IR).

    This design ensures that the exact same set of transformations (cv1, bottlenecks, cv2)
    is applied to both modalities, forcing the network to learn common,
    modality-invariant features.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initializes the shared-weight C2f module.

        Args:
            c1 (int): Total input channels (must be even, e.g., RGB_channels + IR_channels).
            c2 (int): Total output channels (must be even).
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Use shortcut connections in Bottlenecks.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        # Validate that channels can be split evenly
        if c1 % 2 != 0 or c2 % 2 != 0:
            raise ValueError("Input and output channels (c1, c2) must be even for parallel stream processing.")

        self.c1_half = c1 // 2
        c2_half = c2 // 2
        # --- Define the shared layers ---
        self.c = int(c2_half * e)  # Hidden channels for a single stream
        # Shared convolution layers
        self.cv1 = Conv(self.c1_half, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2_half, 1)
        # A single, shared ModuleList for Bottlenecks
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using `chunk()`. Splits the input into two streams,
        processes them through the shared weights, and concatenates the results.
        """
        # Process each stream using the same shared path with chunk()
        rgb_out = self._process(x[:, :self.c1_half, :, :], use_chunk=True)
        ir_out = self._process(x[:, self.c1_half:, :, :], use_chunk=True)
        # Concatenate the final outputs
        return torch.cat([rgb_out, ir_out], dim=1)

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using `split()`. The logic is identical to `forward()`
        but uses `split()` internally for tensor splitting.
        """
        # Process each stream using the same shared path with split()
        rgb_out = self._process(x[:, :self.c1_half, :, :], use_chunk=False)
        ir_out = self._process(x[:, self.c1_half:, :, :], use_chunk=False)
        # Concatenate the final outputs
        return torch.cat([rgb_out, ir_out], dim=1)

    def _process(self, x: torch.Tensor, use_chunk: bool = True) -> torch.Tensor:
        """
        A private helper to process a single data stream through the shared C2f path.
        This avoids code duplication and allows switching between chunk and split.

        Args:
            x (torch.Tensor): The input tensor for a single stream.
            use_chunk (bool): If True, uses `chunk()`. Otherwise, uses `split()`.
        """
        # 1. Apply the first shared convolution and split
        if use_chunk:
            y = list(self.cv1(x).chunk(2, 1))
        else:
            y = list(self.cv1(x).split((self.c, self.c), 1))
        # 2. Apply the shared bottleneck blocks
        y.extend(m(y[-1]) for m in self.m)
        # 3. Concatenate and apply the final shared convolution
        return self.cv2(torch.cat(y, 1))


class MySPPF(nn.Module):
    """
    An SPPF (Spatial Pyramid Pooling - Fast) module with fully shared weights
    to extract common multi-scale features from two parallel input streams
    (e.g., RGB and IR).

    This design applies the exact same set of transformations (cv1, sequential
    max-pooling, cv2) to both modalities, forcing the network to learn common,
    modality-invariant spatial features at different scales.
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initializes the shared-weight SPPF module.

        Args:
            c1 (int): Total input channels (must be even, e.g., RGB_channels + IR_channels).
            c2 (int): Total output channels.
            k (int): The kernel size for the MaxPool2d layer.
        """
        super().__init__()
        # Validate that channels can be split evenly
        if c1 % 2 != 0 or c2 % 2 != 0:
            raise ValueError("Input and output channels (c1, c2) must be even for parallel stream processing.")

        self.c1_half = c1 // 2
        c2_half = c2 // 2
        # --- Define the shared layers ---
        c_ = self.c1_half // 2  # Hidden channels for a single stream
        self.cv1 = Conv(self.c1_half, c_, 1, 1)  # Shared initial convolution to reduce feature channels
        self.cv2 = Conv(c_ * 4, c2_half, 1, 1)  # Shared final convolution to fuse pyramid features
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # A single, shared MaxPool2d layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the shared-weight SPPF.

        It splits the input into two streams, processes each through the shared
        `_process` method, and then concatenates the results.
        """
        # Split the input tensor into two parallel streams (e.g., RGB and IR)
        x_rgb, x_ir = x.split(self.c1_half, dim=1)
        # Process each stream through the same shared SPPF layers
        rgb_out = self._process(x_rgb)
        ir_out = self._process(x_ir)
        # Concatenate the final outputs from both streams along the channel dimension
        return torch.cat((rgb_out, ir_out), dim=1)

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        """
        A private helper method to process a single data stream through the shared SPPF path.
        This encapsulates the core SPPF logic and ensures weight sharing.

        Args:
            x (torch.Tensor): The input tensor for a single stream.
        """
        # 1. Apply the initial shared convolution
        y = [self.cv1(x)]
        # 2. Apply the shared MaxPool layer three times sequentially. This efficiently
        #    creates a feature pyramid by reusing the same pooling layer.
        y.extend(self.m(y[-1]) for _ in range(3))
        # 3. Concatenate the initial features and all pooled features, then fuse them
        #    with the final convolution layer.
        return self.cv2(torch.cat(y, 1))


# TODO: 代码审查（Code Review）

class FFN(nn.Module):
    """一个简单的前馈网络 (Feed-Forward Network)"""

    def __init__(self, d_model, d_ffn=2048, dropout=0.1):
        super().__init__()
        # 通常 FFN 的中间层维度会远大于 d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),  # 比ReLU效果更好但计算量稍大
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        self.d_model = d_model
        # 确保 d_model 是偶数，因为要平分给X和Y
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be an even number, but got {d_model}")

        # 1. 创建一个空的“位置编码”模板，形状为 (通道数, 最大高度, 最大宽度)
        pe = torch.zeros(d_model, max_h, max_w)

        # 2. 准备计算参数
        d_model_half = d_model // 2
        # 'div_term' 是逐渐减小的频率项，公式是 1 / (10000^(2i / d_model_half))
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))

        # 3. 创建坐标轴向量
        pos_w = torch.arange(0., max_w).unsqueeze(1)  # [0, 1, ..., max_w-1]
        pos_h = torch.arange(0., max_h).unsqueeze(1)  # [0, 1, ..., max_h-1]

        # --- 核心计算部分 ---
        # 4. 填充【前半部分通道】，用于编码 Y 轴 (高度)
        # 4.1 填充偶数通道，计算每个高度位置在不同频率下的sin值
        pe[0:d_model_half:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        # 4.2 填充奇数通道，计算每个高度位置在不同频率下的cos值
        pe[1:d_model_half:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        # 5. 填充【后半部分通道】，用于编码 X 轴 (宽度)
        # 5.1 填充偶数通道，计算每个宽度位置在不同频率下的sin值
        pe[d_model_half::2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        # 5.2 填充奇数通道，计算每个宽度位置在不同频率下的cos值
        pe[d_model_half + 1::2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)

        # 6. 注册为 buffer，这样它会随模型移动 (e.g., to(device)) 但不被视为模型参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 是输入的特征图，形状为 (B, C, H, W)
        B, C, H, W = x.shape
        # 从预先计算好的pe中，裁剪出与当前输入匹配的尺寸，并调整形状以匹配批次大小
        # return self.pe[:, :H, :W].unsqueeze(0).repeat(B, 1, 1, 1)
        return self.pe[:, :H, :W].expand(B, -1, -1, -1)


class GatedFusion(nn.Module):
    """
    简单的门控融合模块
    """

    def __init__(self, in_channels):
        super().__init__()
        # 使用一个简单的卷积层来学习门控权重。先用 3x3 卷积提取局部上下文，再用 1x1 卷积生成最终的“注意力分数”或“logit”
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=True)
        )

    def forward(self, x_rgb, x_ir, x_common):
        # 1. 拼接所有输入特征
        gate_input = torch.cat([x_rgb, x_ir, x_common], dim=1)
        # 2. 通过卷积网络生成门控 logits
        gate_logits = self.gate_conv(gate_input)
        # 3. 将 logits 分割为三部分
        gate_rgb_logits, gate_ir_logits, gate_common_logits = torch.chunk(gate_logits, 3, dim=1)
        # 4. 使用 Softmax 生成权重
        gates_stacked = torch.stack([gate_rgb_logits, gate_ir_logits, gate_common_logits], dim=-1)
        gates_softmax = F.softmax(gates_stacked, dim=-1)
        # 5. 分离出三个权重图
        g_rgb, g_ir, g_common = torch.unbind(gates_softmax, dim=-1)  # 每个 Shape: (B, C, H, W)
        # 6. 对输入进行加权求和
        fused_output = g_rgb * x_rgb + g_ir * x_ir + g_common * x_common
        return fused_output


class AdaptiveCoGatedFusion(nn.Module):
    """
    自适应协同门控融合模块 (Direction 4)
    同时利用局部卷积上下文与全局特征协同关系，动态生成融合权重。
    """

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        inter_channels = max(in_channels // reduction, 8)

        # 局部空间上下文编码
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # 全局协同注意力：提取三个特征的全局语义
        self.global_fc = nn.Sequential(
            nn.Linear(in_channels * 3, inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, in_channels * 3)
        )

        # 生成最终门控权重
        self.gate_conv = nn.Conv2d(inter_channels, in_channels * 3, kernel_size=1, bias=True)

    def forward(self, x_rgb, x_ir, x_common):
        B, C, H, W = x_rgb.shape

        # ---- Step 1: 局部特征融合 ----
        local_feat = torch.cat([x_rgb, x_ir, x_common], dim=1)  # (B, 3C, H, W)
        local_context = self.local_conv(local_feat)  # (B, C', H, W)

        # ---- Step 2: 全局特征协同 ----
        rgb_vec = F.adaptive_avg_pool2d(x_rgb, 1).view(B, C)
        ir_vec = F.adaptive_avg_pool2d(x_ir, 1).view(B, C)
        com_vec = F.adaptive_avg_pool2d(x_common, 1).view(B, C)
        global_vec = torch.cat([rgb_vec, ir_vec, com_vec], dim=1)  # (B, 3C)
        global_inter = self.global_fc(global_vec).view(B, 3 * C, 1, 1)

        # ---- Step 3: 融合局部与全局信息 ----
        combined = self.gate_conv(local_context) + global_inter  # (B, 3C, H, W)

        # ---- Step 4: Softmax门控 ----
        gate_rgb, gate_ir, gate_common = torch.chunk(combined, 3, dim=1)
        stacked = torch.stack([gate_rgb, gate_ir, gate_common], dim=-1)
        weights = F.softmax(stacked, dim=-1)
        g_rgb, g_ir, g_common = torch.unbind(weights, dim=-1)

        # ---- Step 5: 加权融合输出 ----
        fused = g_rgb * x_rgb + g_ir * x_ir + g_common * x_common
        return fused


class TransformerWrapper(nn.Module):
    """
    一个完整的 Transformer Block，包含 MHA 和 FFN (Pre-Norm 结构)。
    支持自注意力和跨模态注意力。
    输入特征图形状: (B, C, H, W)，并且应该在外部添加位置编码。
    """

    def __init__(self, d_model, nhead, d_ffn=2048, dropout=0.1, attn_norm='single'):
        super().__init__()
        # Multi-Head Attention 部分
        # TODO: 为query和key/value创建独立的LayerNorm层，但会增加参数量
        self.attn_norm = attn_norm  # 'single' or 'separate'
        if self.attn_norm == 'single':
            self.norm1 = nn.LayerNorm(d_model)
        elif self.attn_norm == 'separate':
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)
        else:
            raise ValueError("attn_norm must be 'single' or 'separate'.")

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        # Feed-Forward Network 部分
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ffn, dropout)

    def forward(self, query, key=None, value=None):
        B, C, H, W = query.shape
        # 默认是自注意力
        is_self_attn = (key is None and value is None)
        if is_self_attn:
            key, value = query, query

        # Reshape for MHA: (B, C, H, W) -> (B, H*W, C) <-> (B, SeqLen, Dim)
        query_seq = query.flatten(2).transpose(1, 2)
        key_seq = key.flatten(2).transpose(1, 2)
        value_seq = value.flatten(2).transpose(1, 2)

        # 1. Multi-Head Attention (with Pre-Norm)
        q = self.norm1(query_seq)
        k = self.norm1(key_seq) if not is_self_attn else q
        v = self.norm1(value_seq) if not is_self_attn else q
        attn_output, _ = self.mha(q, k, v)
        x = query_seq + attn_output  # 第一个残差连接

        # 2. Feed-Forward Network (with Pre-Norm)
        x_ffn = self.ffn(self.norm2(x))
        x = x + x_ffn  # 第二个残差连接

        # 3. Reshape back: (B, H*W, C) -> (B, C, H, W)
        out = x.transpose(1, 2).reshape(B, C, H, W)
        return out


class DHAF(nn.Module):
    def __init__(self, c1, c2,
                 d_model=64,
                 target_size=(20, 20),
                 nhead=8,
                 num_blocks=4,  # 8
                 d_ffn_ratio=4,
                 dropout=0.1):
        """
        解耦与分层注意力融合网络 (Decoupled and Hierarchical Attention Fusion Network, DHAF-Net)

        新增参数:
        num_blocks (int): 每个注意力路径要堆叠的 TransformerWrapper 数量。
        """
        super().__init__()

        # TODO: 1×1 Conv + max/avg pool 降维
        # self.spatial_pool = nn.AdaptiveMaxPool2d(target_size)
        self.spatial_pool = nn.AdaptiveAvgPool2d(target_size)
        in_channels = c1 // 4
        # 1x1 卷积用于将每个流的通道数统一到 d_model
        # self.projection = nn.Conv2d(in_channels, d_model, kernel_size=1)
        # TODO: 使用独立的projection。也可以使用一个共享的projection降低参数量，但会降低灵活性
        self.proj_rgb_spec = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.proj_ir_spec = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.proj_rgb_comm = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.proj_ir_comm = nn.Conv2d(in_channels, d_model, kernel_size=1)
        # 位置编码模块
        self.pos_encoder = PositionalEncoding2D(d_model=d_model, max_h=target_size[0], max_w=target_size[1])

        # 使用 nn.ModuleList 来创建 TransformerWrapper 的深度堆叠
        # 每个路径都是一个独立的序列，包含 num_blocks 个 TransformerWrapper
        self.self_attn_rgb_blocks = nn.ModuleList([
            TransformerWrapper(d_model, nhead, d_model * d_ffn_ratio, dropout)
            for _ in range(num_blocks)
        ])
        self.self_attn_ir_blocks = nn.ModuleList([
            TransformerWrapper(d_model, nhead, d_model * d_ffn_ratio, dropout)
            for _ in range(num_blocks)
        ])
        self.cross_attn_rgb_ir_blocks = nn.ModuleList([
            TransformerWrapper(d_model, nhead, d_model * d_ffn_ratio, dropout)
            for _ in range(num_blocks)
        ])
        self.cross_attn_ir_rgb_blocks = nn.ModuleList([
            TransformerWrapper(d_model, nhead, d_model * d_ffn_ratio, dropout)
            for _ in range(num_blocks)
        ])

        # 融合 cross-attention 后的共性特征，生成一个通用的共性特征
        self.common_feature_fusion = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        # 门控融合模块
        # self.final_gate_fusion = GatedFusion(in_channels=d_model)
        self.final_gate_fusion = AdaptiveCoGatedFusion(in_channels=d_model)

        # 将融合后的低分辨率特征恢复到原始通道数
        self.proj_up_channel = nn.Conv2d(d_model, c2, kernel_size=1)
        # Skip Connection，定义一个1x1卷积，用于将原始输入x的通道数直接调整到最终输出的c2
        self.bypass_conv = nn.Conv2d(c1, c2, kernel_size=1)
        # 主路与旁路融合后的处理
        self.final_conv = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )

    def forward(self, x):
        # 从输入 x 获取原始的高度 H 和宽度 W
        _, _, H, W = x.shape
        # 旁路（Skip Connection），直接对原始输入x进行处理，保留高分辨率信息
        bypass_out = self.bypass_conv(x)  # 形状: (B, c2, H, W)
        # 主路（Attention Path）
        f_rgb_specific, f_ir_specific, f_rgb_common, f_ir_common = torch.chunk(x, 4, dim=1)
        # --- Step 1: 预处理: 统一空间和通道维度 ---
        # 空间池化, 将H和W降维到target_size
        f_rgb_specific_pooled = self.spatial_pool(f_rgb_specific)
        f_ir_specific_pooled = self.spatial_pool(f_ir_specific)
        f_rgb_common_pooled = self.spatial_pool(f_rgb_common)
        f_ir_common_pooled = self.spatial_pool(f_ir_common)
        # 通道投影 (Projection)
        f_rgb_specific_proj = self.proj_rgb_spec(f_rgb_specific_pooled)
        f_ir_specific_proj = self.proj_ir_spec(f_ir_specific_pooled)
        f_rgb_common_proj = self.proj_rgb_comm(f_rgb_common_pooled)
        f_ir_common_proj = self.proj_ir_comm(f_ir_common_pooled)
        # 现在所有特征图的形状都是 (B, d_model, target_size[0], target_size[1])

        # 为所有输入添加位置编码。这里所有流共享相同的位置编码，这是合理的，因为它们有相同的空间尺寸
        pos_embed = self.pos_encoder(f_rgb_specific_proj)
        f_rgb_specific_pos = f_rgb_specific_proj + pos_embed
        f_ir_specific_pos = f_ir_specific_proj + pos_embed
        f_rgb_common_pos = f_rgb_common_proj + pos_embed
        f_ir_common_pos = f_ir_common_proj + pos_embed

        # --- Step 2: 通过堆叠的 TransformerWrapper 进行深度特征增强 ---
        # Self-attention
        f_rgb_specific_enhanced = f_rgb_specific_pos
        for block in self.self_attn_rgb_blocks:
            f_rgb_specific_enhanced = block(f_rgb_specific_enhanced)
        f_ir_specific_enhanced = f_ir_specific_pos
        for block in self.self_attn_ir_blocks:
            f_ir_specific_enhanced = block(f_ir_specific_enhanced)
        # Cross-attention
        # 注意: 在跨注意力中，query 在循环中更新，而 key/value 保持不变
        f_rgb_fused_common = f_rgb_common_pos
        for block in self.cross_attn_rgb_ir_blocks:
            f_rgb_fused_common = block(query=f_rgb_fused_common, key=f_ir_common_pos, value=f_ir_common_pos)
        f_ir_fused_common = f_ir_common_pos
        for block in self.cross_attn_ir_rgb_blocks:
            f_ir_fused_common = block(query=f_ir_fused_common, key=f_rgb_common_pos, value=f_rgb_common_pos)

        # --- Step 3: 创建通用共性特征 ---
        f_common_concatenated = torch.cat([f_rgb_fused_common, f_ir_fused_common], dim=1)
        f_universal_common = self.common_feature_fusion(f_common_concatenated)

        # --- Step 4: 三路特征的最终门控融合 ---
        fused_low_res = self.final_gate_fusion(
            f_rgb_specific_enhanced,
            f_ir_specific_enhanced,
            f_universal_common
        )  # (B, d_model, target_h, target_w)

        # --- Step 5: 还原通道和空间维度 ---
        # 使用 1x1 卷积恢复通道数
        fused_proj = self.proj_up_channel(fused_low_res)  # 形状: (B, c2, target_h, target_w)
        # 使用插值法还原空间尺寸 (H, W)
        main_out = F.interpolate(fused_proj, size=(H, W), mode='bilinear', align_corners=False)  # 形状: (B, c2, H, W)

        # 使用残差连接融合主路与旁路
        output = main_out + bypass_out
        # 通过一个最终的卷积块进行精炼
        output = self.final_conv(output)
        return output


# -------------------------------------------------------------


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def log_sinkhorn_iterations(log_alpha, n_iters=3):
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return torch.exp(log_alpha)


def topk_attention_renorm(attn, keep_ratio=0.8):
    B, num_heads, N, _ = attn.shape
    k = int(N * keep_ratio)
    if k >= N or k <= 0: return attn
    topk_vals, topk_idx = torch.topk(attn, k=k, dim=-1)
    mask = torch.zeros_like(attn)
    mask.scatter_(-1, topk_idx, 1.0)
    attn_masked = attn * mask
    attn_renorm = attn_masked / (attn_masked.sum(dim=-1, keepdim=True) + 1e-8)
    return attn_renorm


# ------------------------------------------------------------------------------
# 2. 核心模块 (Core Modules)
# ------------------------------------------------------------------------------

class SRUE(nn.Module):
    """保持不变"""

    def __init__(self, in_channels, out_channels, spatial_scale=2, channel_scale=1, act_layer=nn.GELU):
        super(SRUE, self).__init__()
        mid_channels = in_channels // channel_scale

        self.main_branch = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=spatial_scale),
            nn.Conv2d(in_channels * (spatial_scale ** 2), mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            act_layer()
        )

        res_stride = spatial_scale
        res_kernel = 2 * spatial_scale - 1
        res_padding = spatial_scale - 1
        res_layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=res_kernel, stride=res_stride, padding=res_padding,
                      groups=in_channels, bias=False)
        ]
        if in_channels != mid_channels:
            res_layers.append(nn.BatchNorm2d(in_channels))
            res_layers.append(act_layer())
            res_layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1))
        res_layers.append(nn.BatchNorm2d(mid_channels))
        res_layers.append(act_layer())
        self.residual_branch = nn.Sequential(*res_layers)

        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            act_layer()
        )

    def forward(self, x):
        x_main = self.main_branch(x)
        x_res = self.residual_branch(x)
        x_cat = torch.cat([x_main, x_res], dim=1)
        return self.fusion(x_cat)


class DNAA(nn.Module):

    def __init__(self, dim, num_iters=3):
        super().__init__()
        self.dim = dim
        self.num_iters = num_iters
        self.ln = nn.GroupNorm(1, dim)
        self.qkv_conv = nn.Conv2d(dim * 2, dim * 6, kernel_size=1, bias=False)
        self.out_rgb = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_ir = nn.Conv2d(dim, dim, kernel_size=1)
        self.scale_spatial = dim ** -0.5

    def forward(self, F_rgb, F_ir):
        B, C, H, W = F_rgb.shape
        N = H * W
        scale_channel = N ** -0.5
        rgb_feat = self.ln(F_rgb)
        ir_feat = self.ln(F_ir)
        qkv = self.qkv_conv(torch.cat([rgb_feat, ir_feat], dim=1))
        qkv_rgb, qkv_ir = torch.chunk(qkv, 2, dim=1)
        Q_rgb, K_rgb, V_rgb = map(lambda t: t.flatten(2), torch.chunk(qkv_rgb, 3, dim=1))
        Q_ir, K_ir, V_ir = map(lambda t: t.flatten(2), torch.chunk(qkv_ir, 3, dim=1))

        log_A_c = torch.bmm(Q_rgb, K_ir.transpose(1, 2)) * scale_channel
        A_c = log_sinkhorn_iterations(log_A_c, self.num_iters)

        A_hw = torch.bmm(K_rgb.transpose(1, 2), Q_ir) * self.scale_spatial
        A_hw = F.softmax(A_hw, dim=-1)

        rgb_enhance = torch.bmm(torch.bmm(A_c.transpose(1, 2), V_rgb), A_hw)
        ir_enhance = torch.bmm(torch.bmm(A_c, V_ir), A_hw.transpose(1, 2))

        F_rgb_out = F_rgb + self.out_rgb(rgb_enhance.view(B, C, H, W))
        F_ir_out = F_ir + self.out_ir(ir_enhance.view(B, C, H, W))
        return F_rgb_out, F_ir_out


class SPA(nn.Module):

    def __init__(self, dim, num_heads=4, keep_ratio=0.8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.keep_ratio = keep_ratio
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(1, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q.view(B, self.num_heads, self.head_dim, N).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, N).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, N).transpose(2, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = topk_attention_renorm(attn, self.keep_ratio)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        out = self.proj(out)
        return x + out



class MDFFN(nn.Module):
    """
    Multi-Dimensional Feed-Forward Network (MD-FFN)
    """

    def __init__(self, in_channels, expansion_ratio=2, act_layer=nn.GELU):
        super(MDFFN, self).__init__()

        # 移除了 self.H 和 self.W 的参数，不再强制检查尺寸

        assert in_channels % 4 == 0, "Input channels must be divisible by 4."
        self.split_channels = in_channels // 4

        self.norm = LayerNorm2d(in_channels)

        # --- 1. C-Branch: Channel Mixing (Pointwise Conv) ---
        hidden_c = int(self.split_channels * expansion_ratio)
        self.branch_c = nn.Sequential(
            nn.Conv2d(self.split_channels, hidden_c, kernel_size=1),
            act_layer(),
            nn.Conv2d(hidden_c, self.split_channels, kernel_size=1)
        )

        # --- 2. H-Branch: Height Mixing (Vertical Strip Conv) ---
        # 使用 (K, 1) 的卷积核来模拟沿 H 维度的混合
        # padding=(K//2, 0) 保持尺寸不变
        k_h = 7
        self.branch_h = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(k_h, 1),
                      padding=(k_h // 2, 0), groups=self.split_channels, bias=False),  # DWConv
            nn.BatchNorm2d(self.split_channels),
            act_layer(),
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=1)  # PWConv
        )

        # --- 3. W-Branch: Width Mixing (Horizontal Strip Conv) ---
        # 使用 (1, K) 的卷积核来模拟沿 W 维度的混合
        k_w = 7
        self.branch_w = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(1, k_w),
                      padding=(0, k_w // 2), groups=self.split_channels, bias=False),  # DWConv
            nn.BatchNorm2d(self.split_channels),
            act_layer(),
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=1)  # PWConv
        )

        # 4. Identity Branch (无参数)

        # --- Fusion ---
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W) - H和W现在可以是任意值
        shortcut = x

        x = self.norm(x)
        x_splits = torch.chunk(x, chunks=4, dim=1)

        # Branch C
        out_c = self.branch_c(x_splits[0])
        # Branch H (No permute needed anymore, handled by Strip Conv)
        out_h = self.branch_h(x_splits[1])
        # Branch W (No permute needed anymore, handled by Strip Conv)
        out_w = self.branch_w(x_splits[2])
        # Branch ID
        out_id = x_splits[3]

        # Fusion
        x_cat = torch.cat([out_c, out_h, out_w, out_id], dim=1)
        x_out = self.fusion_conv(x_cat)

        return shortcut + x_out


class ACFM(nn.Module):
    """保持不变"""

    def __init__(self, channels):
        super(ACFM, self).__init__()
        self.common_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.gate_generator = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.rgb_u_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.ir_u_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_unique, rgb_common, ir_common, ir_unique):
        cat_common = torch.cat([rgb_common, ir_common], dim=1)
        f_base = self.common_fusion(cat_common)
        cat_all = torch.cat([f_base, rgb_unique, ir_unique], dim=1)
        gates = self.gate_generator(cat_all)
        w_rgb, w_ir = torch.chunk(gates, chunks=2, dim=1)
        feat_rgb_aligned = self.rgb_u_transform(rgb_unique)
        feat_ir_aligned = self.ir_u_transform(ir_unique)
        out = f_base + (w_rgb * feat_rgb_aligned) + (w_ir * feat_ir_aligned)
        return out


# ------------------------------------------------------------------------------
# 4. 整合模块 (移除对 h, w 参数的传递)
# ------------------------------------------------------------------------------

class DiffBlock(nn.Module):
    """
    Updated DiffBlock: Removed h, w args
    """

    def __init__(self, dim, num_heads=4, keep_ratio=0.8, expansion_ratio=2, act_layer=nn.GELU):
        super(DiffBlock, self).__init__()

        self.spa_rgb = SPA(dim, num_heads=num_heads, keep_ratio=keep_ratio)
        self.spa_ir = SPA(dim, num_heads=num_heads, keep_ratio=keep_ratio)
        self.dnaa = DNAA(dim, num_iters=3)

        # FFNs initialized without fixed height/width
        self.ffn_rgb_s = MDFFN(dim, expansion_ratio=expansion_ratio, act_layer=act_layer)
        self.ffn_ir_s = MDFFN(dim, expansion_ratio=expansion_ratio, act_layer=act_layer)
        self.ffn_rgb_c = MDFFN(dim, expansion_ratio=expansion_ratio, act_layer=act_layer)
        self.ffn_ir_c = MDFFN(dim, expansion_ratio=expansion_ratio, act_layer=act_layer)

    def forward(self, x_rgb_s, x_ir_s, x_rgb_c, x_ir_c):
        # Attention
        x_rgb_s = self.spa_rgb(x_rgb_s)
        x_ir_s = self.spa_ir(x_ir_s)
        x_rgb_c, x_ir_c = self.dnaa(x_rgb_c, x_ir_c)

        # FFN
        x_rgb_s = self.ffn_rgb_s(x_rgb_s)
        x_ir_s = self.ffn_ir_s(x_ir_s)
        x_rgb_c = self.ffn_rgb_c(x_rgb_c)
        x_ir_c = self.ffn_ir_c(x_ir_c)

        return x_rgb_s, x_ir_s, x_rgb_c, x_ir_c


class DAFM(nn.Module):
    """
    Updated DAFM: Removed target_size arg
    """

    def __init__(self, c1, c2,
                 spatial_scale=2,
                 channel_scale=1,
                 num_blocks=2,
                 # num_blocks=4,
                 # num_blocks=8,
                 num_heads=4,
                 target_channels=64):  # Removed target_size
        super(DAFM, self).__init__()

        assert c1 % 4 == 0, "Input channels c1 must be divisible by 4"
        in_channels = c1 // 4

        self.bypass_conv = nn.Conv2d(c1, c2, kernel_size=1)

        self.proj_rgb_s = SRUE(in_channels, target_channels, spatial_scale, channel_scale)
        self.proj_ir_s = SRUE(in_channels, target_channels, spatial_scale, channel_scale)
        self.proj_rgb_c = SRUE(in_channels, target_channels, spatial_scale, channel_scale)
        self.proj_ir_c = SRUE(in_channels, target_channels, spatial_scale, channel_scale)

        self.blocks = nn.ModuleList([
            DiffBlock(
                dim=target_channels,
                num_heads=num_heads,
                keep_ratio=0.8
            ) for _ in range(num_blocks)
        ])

        self.fusion_module = ACFM(channels=target_channels)

        self.upsample = nn.Sequential(
            nn.Conv2d(target_channels, c2 * (spatial_scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(spatial_scale),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1)
        )

        self.final_act = nn.GELU()

    def forward(self, x):
        # x: (B, c1, H, W)
        bypass = self.bypass_conv(x)
        x_rgb_s, x_ir_s, x_rgb_c, x_ir_c = torch.chunk(x, 4, dim=1)

        f_rgb_s = self.proj_rgb_s(x_rgb_s)
        f_ir_s = self.proj_ir_s(x_ir_s)
        f_rgb_c = self.proj_rgb_c(x_rgb_c)
        f_ir_c = self.proj_ir_c(x_ir_c)

        for block in self.blocks:
            f_rgb_s, f_ir_s, f_rgb_c, f_ir_c = block(f_rgb_s, f_ir_s, f_rgb_c, f_ir_c)

        f_fused = self.fusion_module(f_rgb_s, f_rgb_c, f_ir_c, f_ir_s)
        out = self.upsample(f_fused)

        return self.final_act(out + bypass)

