import torch
import torch.nn as nn
from torch import Tensor, sqrt, ones, zeros
from torch.autograd import Function
from typing import Literal
from math import floor


Activation_Function = Literal[
    "GELU", "SG", "Identity", "ReLU", "LReLU", "Tanh", "Sigmoid"
]
Attention_Mechanism = Literal["CA", "SCA"]


def get_activation(activation_type: Activation_Function = "Identity"):
    """
    Get the activation function from the activation type.
    :param activation_type: Activation type
    :return: Activation function
    """
    if activation_type == "GELU":
        return nn.GELU()
    elif activation_type == "SG":
        return SimpleGate()
    elif activation_type == "Identity":
        return nn.Identity()
    elif activation_type == "ReLU":
        return nn.ReLU()
    elif activation_type == "LReLU":
        return nn.LeakyReLU()
    elif activation_type == "Tanh":
        return nn.Tanh()
    elif activation_type == "Sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError(
            f"Activation type {activation_type} is not implemented."
        )


def get_cnn_attention(attention_type: Attention_Mechanism = None):
    if attention_type == "CA":
        return ChannelAttention
    elif attention_type == "SCA":
        return SimplifiedChannelAttention
    else:
        raise NotImplementedError(
            f"Attention type {attention_type} is not implemented."
        )


class SimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ChannelAttention(nn.Module):
    def __init__(self, num_channel: int):
        super(ChannelAttention, self).__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=num_channel,
                out_channels=num_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=num_channel // 2,
                out_channels=num_channel,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, num_channel: int):
        super(SimplifiedChannelAttention, self).__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=num_channel,
                out_channels=num_channel,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ConcatTensors(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.cat((x, y), dim=1)


class IdentityMod(nn.Module):
    def __init__(self):
        super(IdentityMod, self).__init__()

    def forward(self, x=None, *args, **kwargs) -> Tensor:
        return x


class SkipConnection(nn.Module):
    def __init__(
        self,
    ):
        super(SkipConnection, self).__init__()

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return x + skip


class LayerNormFunction(Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.channels = channels
        self.register_parameter("weight", nn.Parameter(ones(channels)))
        self.register_parameter("bias", nn.Parameter(zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class InvertedConvolution(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding="same", bias=True
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=bias,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            groups=out_channels,
            bias=bias,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BlockMod(nn.Module):
    def __init__(
        self,
        channels: int,
        dw_expand: float = 1.0,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
        attention_type: Literal["CA", "SCA"] = "CA",
        activation_type: Literal[
            "GELU", "SG", "Identity", "ReLU", "LReLU", "Tanh", "Sigmoid"
        ] = "GELU",
        inverted_conv: bool = True,
        kernel_size: int = 3,
        use_pos_map: bool = False,
        depth: int = None,
    ):
        """
        Modified block from NAFNet with optional CoordConv on th first convolution.
        Args:
            channels: Number of input channels
            dw_expand: Expansion factor for sub block 1
            ffn_expand: Expansion factor for sub block 2
            drop_out_rate: Dropout rate
            attention_type: Type of attention to use
            activation_type: Type of activation to use
            inverted_conv: Whether to use inverted convolution
            kernel_size: size of all convolution kernels
            use_pos_map: Whether coordconv is used
            depth: Depth of the block in the network, used for positional map scaling
        """
        super().__init__()
        dw_channel = (
            int(channels * dw_expand)
            if int(channels * dw_expand) % 2 == 0
            else int(channels * dw_expand) + 1
        )

        self.cat = ConcatTensors() if use_pos_map else IdentityMod()

        if inverted_conv:
            self.conv1 = InvertedConvolution(
                in_channels=channels + 2 if use_pos_map else channels,
                out_channels=dw_channel,
                kernel_size=kernel_size,
                padding="same",
                bias=True,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels=channels + 2 if use_pos_map else channels,
                out_channels=dw_channel,
                kernel_size=kernel_size,
                padding="same",
                stride=1,
                bias=True,
            )

        # Activation
        self.activation = get_activation(activation_type)

        # Channel Attention
        self.attention = get_cnn_attention(attention_type)(
            dw_channel // 2 if activation_type == "SG" else dw_channel
        )

        self.conv2 = nn.Conv2d(
            in_channels=dw_channel // 2 if activation_type == "SG" else dw_channel,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # sub block 1 done
        ffn_channel = floor(ffn_expand * channels)
        self.conv3 = nn.Conv2d(
            in_channels=channels,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        # second activation call
        self.conv4 = nn.Conv2d(
            in_channels=ffn_channel // 2 if activation_type == "SG" else ffn_channel,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # sub block 2 done

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

        # save use_pos_map and depth metadata for forward
        self.use_pos_map = use_pos_map
        self.depth = depth

    def forward(self, source: Tensor, pos_map: Tensor = None) -> Tensor:
        x = source

        x = self.norm1(x)

        x = self.cat(x, pos_map)
        x = self.conv1(x)
        x = self.activation(x)

        att = self.attention(x)

        x = x * att
        x = self.conv2(x)

        x = self.dropout1(x)

        y = source + x * self.beta

        x = self.conv3(self.norm2(y))
        x = self.activation(x)
        x = self.conv4(x)

        x = self.dropout2(x)

        ret = y + x * self.gamma

        return ret
