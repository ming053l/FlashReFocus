"""Layers shared by the decoder refiner (Stage I) and the renderer (Stage II): the NAFNet block and its parts."""
from math import floor

import torch
from torch import nn
from torch.autograd import Function


class LayerNormFunction(Function):
    """Channel LayerNorm of NCHW tensors with an explicit backward (as in NAFNet)."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _, C, _, _ = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        return weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        _, C, _, _ = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        # reduce over W, H, then N, in this order (it fixes the floating-point summation order)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class InvertedConvolution(nn.Module):
    """1x1 convolution followed by a depthwise k x k convolution."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding="same", groups=out_channels)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class NAFBlock(nn.Module):
    """NAFNet block (GELU, channel attention, inverted first convolution). With cond_channels > 0 the first convolution
    also reads condition maps (the renderer's M_C, resized to the block's resolution)."""

    def __init__(self, channels, cond_channels=0, ffn_expand=2.0, kernel_size=3):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = InvertedConvolution(channels + cond_channels, channels, kernel_size)
        self.activation = nn.GELU()
        self.attention = ChannelAttention(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        ffn_channels = floor(ffn_expand * channels)
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv4 = nn.Conv2d(ffn_channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x, cond=None):
        y = self.norm1(x) if cond is None else torch.cat((self.norm1(x), cond), dim=1)
        y = self.activation(self.conv1(y))
        y = x + self.conv2(y * self.attention(y)) * self.beta
        return y + self.conv4(self.activation(self.conv3(self.norm2(y)))) * self.gamma
