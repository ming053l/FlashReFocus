import warnings
from typing import Tuple, Optional

import torch
from timm.layers import to_2tuple
from torch import nn, Tensor, cat, sqrt, ones, zeros
from torch.autograd import Function

from torch.nn.init import trunc_normal_


class IdentityMod(nn.Module):
    def __init__(self):
        super(IdentityMod, self).__init__()

    def forward(self, x=None, *args, **kwargs) -> Tensor:
        return x


class ConcatTensors(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return cat((x, y), dim=1)


class SkipConnection(nn.Module):
    def __init__(
        self,
    ):
        super(SkipConnection, self).__init__()

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return x + skip


class ApplyVectorWeights(nn.Module):
    def __init__(self):
        super(ApplyVectorWeights, self).__init__()

    def forward(self, x: Tensor, weights: Tensor) -> Tensor:
        return x * weights


class ChannelEmbeddingCompression(nn.Module):
    def __init__(self, embed_dim, embed_dim_next):
        super().__init__()
        self.patch_unembed = PatchUnEmbedIR(embed_dim=embed_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim_next, 1, 1, 0)
        self.patch_embed = PatchEmbedIR(embed_dim=embed_dim_next)

    def forward(self, x, x_size):
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x)
        return x


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


class ContextFiLM(nn.Module):
    def __init__(self, C, hidden=64, use_tanh=True, scale_limit=0.5):
        super().__init__()
        self.use_tanh = use_tanh
        self.scale_limit = scale_limit

        self.conv1 = nn.Conv2d(1, hidden, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, 2 * C, 1)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, context):
        # x: (B,C,H,W), context: (B,1,H,W)
        s, t = self.conv2(self.act(self.conv1(context))).chunk(2, dim=1)
        if self.use_tanh:
            s = torch.tanh(s) * self.scale_limit
            t = torch.tanh(t) * self.scale_limit
        return x * (1.0 + s) + t


class InvContextFiLM(nn.Module):
    def __init__(self, C, hidden=64, use_tanh=True, scale_limit=0.5):
        super().__init__()
        self.use_tanh = use_tanh
        self.scale_limit = scale_limit

        self.conv1 = nn.Conv2d(C, hidden, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, 2, 1)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, context):
        # x: (B,1,H,W), context: (B,C,H,W)
        s, t = self.conv2(self.act(self.conv1(context))).chunk(2, dim=1)
        if self.use_tanh:
            s = torch.tanh(s) * self.scale_limit
            t = torch.tanh(t) * self.scale_limit
        return x * (1.0 + s) + t


class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        x: (b h w c)
        """
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x


class ChannelAttention(nn.Module):
    """
    Channel attention module with optional attention weights.
    """

    def __init__(self, num_channel: int):
        """

        :param num_channel: Number of channels in the input tensor
        :param apply_att_weights: Should attention weights be applied in the forward pass?
        """
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
        x = self.model(x)
        return x


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, num_channel, apply_att_weights=False):
        """

        :param num_channel: Number of channels in the input tensor
        :param apply_att_weights: Should attention weights be applied in the forward pass?
        """
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

        self.apply1 = ApplyVectorWeights() if apply_att_weights else IdentityMod()

    def forward(self, x: Tensor, att_weights: Tensor = None) -> Tensor:
        x = self.model(x)
        return self.apply1(x=x, weights=att_weights)


class DiffApertureAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, layer_idx, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = (self.embed_dim * self.factor) // num_heads
        self.key_dim = self.embed_dim // (num_heads * 2)
        self.scaling = self.key_dim**-0.5

        # Projections for Q and K (remaining the same size but will be split)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)

        self.lepe = DWConv2d(embed_dim * self.factor, 5, 1, 2)

        # Learnable lambda parameters
        self.lambda_init = 0.8 - 0.6 * torch.exp(torch.tensor(-0.3 * (layer_idx - 1)))
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.key_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.key_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.key_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.key_dim).normal_(mean=0, std=0.1)
        )

        # Head-wise normalization
        self.sub_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, rel_pos):
        bsz, h, w, _ = x.size()
        mask_h, mask_w = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        # Split Q and K into two groups for differential attention
        # q1, q2 shape: (bsz, h, w, num_heads, key_dim)
        q1, q2 = q.view(bsz, h, w, self.num_heads, 2, self.key_dim).unbind(4)
        k1, k2 = k.view(bsz, h, w, self.num_heads, 2, self.key_dim).unbind(4)

        k1, k2 = k1 * self.scaling, k2 * self.scaling
        v = v.view(bsz, h, w, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        # Calculate learnable lambda
        lam = (
            torch.exp(self.lambda_q1 @ self.lambda_k1)
            - torch.exp(self.lambda_q2 @ self.lambda_k2)
            + self.lambda_init
        )

        # --- Horizontal Attention (Width) ---
        # qr shape: (bsz, num_heads, h, w, key_dim)
        qr1, qr2 = q1.permute(0, 3, 1, 2, 4), q2.permute(0, 3, 1, 2, 4)
        kr1, kr2 = k1.permute(0, 3, 1, 2, 4), k2.permute(0, 3, 1, 2, 4)
        q1_w, q2_w = qr1.transpose(1, 2), qr2.transpose(1, 2)
        k1_w, k2_w = kr1.transpose(1, 2), kr2.transpose(1, 2)

        # attn_w = softmax(q1k1) - lambda * softmax(q2k2)
        mask_w = mask_w.unsqueeze(1).expand(-1, h, -1, -1, -1)
        attn1_w = torch.softmax(q1_w @ k1_w.transpose(-1, -2) + mask_w, dim=-1)
        attn2_w = torch.softmax(q2_w @ k2_w.transpose(-1, -2) + mask_w, dim=-1)
        v = (attn1_w - lam * attn2_w) @ v

        # --- Vertical Attention (Height) ---
        q1_h, q2_h = qr1.permute(0, 3, 1, 2, 4), qr2.permute(0, 3, 1, 2, 4)
        k1_h, k2_h = kr1.permute(0, 3, 1, 2, 4), kr2.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)

        mask_h = mask_h.unsqueeze(1).expand(-1, w, -1, -1, -1)
        attn1_h = torch.softmax(q1_h @ k1_h.transpose(-1, -2) + mask_h, dim=-1)
        attn2_h = torch.softmax(q2_h @ k2_h.transpose(-1, -2) + mask_h, dim=-1)
        output = (attn1_h - lam * attn2_h) @ v

        # --- Final Processing ---
        # Apply head-wise GroupNorm and scale by (1 - lambda_init)
        output = output.permute(0, 3, 1, 2, 4)  # (bsz, h, w, num_heads, head_dim)
        output = self.sub_norm(output) * (1 - self.lambda_init)

        output = output.flatten(-2, -1) + lepe
        return self.out_proj(output)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


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


class DynRelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        """
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        """
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        self.register_buffer("angle", angle)

    def generate_1d_decay(self, l: int, range_factor: Tensor):
        """
        generate 1d decay mask, the result is l*l
        """

        range_factor = abs(range_factor)

        bs = range_factor.size(0)
        ###print(f"Generating 1D decay mask for batch size: {bs} and length: {l}")

        heads_ranges = (
            self.heads_range
            * torch.arange(self.num_heads, dtype=torch.float)
            / self.num_heads
        )
        heads_ranges = heads_ranges.to(range_factor.device)
        ###print("Heads Ranges:")
        ###print(heads_ranges)
        ###print(f"Heads Ranges shape: {heads_ranges.shape}")

        range_factor = torch.sqrt(
            torch.sqrt(range_factor)
        )  # (n) # give extra weight to smaller values
        range_factor = range_factor[:, None]  # (n 1)
        ###print("Range Factor:")
        ###print(range_factor)
        ###print(f"Range Factor shape: {range_factor.shape}")

        ranges = -self.initial_value - heads_ranges.repeat(bs, 1) * range_factor
        ###print("Ranges:")
        decay = torch.log(1 - 2**ranges)  # (b n)
        ###print("Decay:")
        ###print(decay)
        ###print(f"Decay shape: {decay.shape}")

        index = torch.arange(l).to(decay)
        ###print("Index:")
        ###print(index)
        ###print(f"Index shape: {index.shape}")
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        # extend mask to batch size with one channel for each
        mask = mask[None, None, :, :]  # (1 1 l l)
        ###print("Mask before decay application:")
        ###print(mask)
        ###print(f"Mask shape : {mask.shape}")

        mask = mask * decay[:, :, None, None]  # (b n l l)
        ###print("Mask after decay application:")
        ###print(mask)
        ###print(f"Mask shape: {mask.shape}")

        return mask

    def forward(self, slen: Tuple[int], range_factor: Tensor):
        mask_h = self.generate_1d_decay(
            slen[0], range_factor=range_factor
        )  # x axis decay
        mask_w = self.generate_1d_decay(
            slen[1], range_factor=range_factor
        )  # y axis decay
        return mask_h, mask_w


class PatchEmbedIR(nn.Module):
    r"""Image to Patch Embedding

    Args:
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self, embed_dim=96, norm_layer=None):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (b c h w) -> (b h w c)
        if self.norm is not None:
            # print("Using norm layer")
            x = self.norm(x)
        return x


class PatchUnEmbedIR(nn.Module):
    r"""Image to Patch Unembedding

    Args:
        embed_dim (int): Number of linear projection output channels.
    """

    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x
