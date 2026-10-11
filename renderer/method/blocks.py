"""Building blocks of the Stage II renderer: NAF blocks with condition maps for the U-Net, and the defocus-bias
attention of the transformer bottleneck (axial attention biased by defocus differences, scaled by the strength K)."""
from math import floor

import torch
import torch.nn.functional as F
from timm.layers import DropPath
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
    """NAFNet block whose first convolution also reads the condition maps M_C (resized to the block's resolution)."""

    def __init__(self, channels, cond_channels, ffn_expand=2.0, kernel_size=3):
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

    def forward(self, x, cond):
        y = self.activation(self.conv1(torch.cat((self.norm1(x), cond), dim=1)))
        y = x + self.conv2(y * self.attention(y)) * self.beta
        return y + self.conv4(self.activation(self.conv3(self.norm2(y)))) * self.gamma


class DWConv2d(nn.Module):
    """Depthwise convolution on channels-last (b, h, w, c) tensors."""

    def __init__(self, dim, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, padding, groups=dim)

    def forward(self, x):
        return self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DefocusBiasAttention(nn.Module):
    """Defocus-bias attention (Sec. 3.2, Appendix C.1), axial: along each row, then along each column.

    A_ij = softmax_j(q_i^T k_j / sqrt(d) + gamma_b * alpha * B_ij),  B_ij = -|T_D,i - T_D,j|,  T_D = phi_D(M),
    where phi_D maps the defocus magnitude M to fixed log features at several scales and then, with a learned linear
    layer (defocus_head_proj), to one value per head; alpha (defocus_alpha) is learned per head and
    gamma_b = exp(clip(w_b K + c_b, -4, 4)) (aperture_scale_proj); gamma_K = gamma_b * alpha in the main paper.
    The output is W_o(sum_j A_ij v_j + LePE(V)_i), with LePE a depthwise 5x5 convolution of the values."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = embed_dim // num_heads                                 # d
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)                         # W_q
        self.k_proj = nn.Linear(embed_dim, embed_dim)                         # W_k
        self.v_proj = nn.Linear(embed_dim, embed_dim)                         # W_v
        self.lepe = DWConv2d(embed_dim, 5, 2)
        self.out_proj = nn.Linear(embed_dim, embed_dim)                       # W_o
        self.num_log_features = max(embed_dim // 4, 1)
        self.defocus_head_proj = nn.Linear(self.num_log_features, num_heads)  # linear layer of phi_D
        self.defocus_alpha = nn.Parameter(torch.ones(num_heads))              # alpha
        self.aperture_scale_proj = nn.Linear(1, num_heads)                    # w_b, c_b

    def _log_features(self, defocus):
        """M (b, 1, h, w) -> (b, h, w, n) features log(1 + M / s) / log(1 + 1 / s), scales s from 1/n to 1."""
        n = self.num_log_features
        min_scale = 1.0 / max(float(n), 16.0)
        scales = torch.logspace(torch.log10(defocus.new_tensor(min_scale)), torch.log10(defocus.new_tensor(1.0)),
                                steps=n, dtype=defocus.dtype, device=defocus.device).view(1, n, 1, 1)
        features = torch.log1p(defocus.clamp(0.0, 1.0) / scales) / torch.log1p(1.0 / scales)
        return features.permute(0, 2, 3, 1).contiguous()

    def _defocus_bias(self, defocus):
        """B along rows (b, h, heads, w, w) and along columns (b, w, heads, h, h)."""
        t_d = self.defocus_head_proj(self._log_features(defocus))           # T_D = phi_D(M): b, h, w, heads
        t_rows = t_d.permute(0, 1, 3, 2).contiguous()                         # b, h, heads, w
        t_cols = t_d.permute(0, 2, 3, 1).contiguous()                         # b, w, heads, h
        return (-torch.abs(t_rows.unsqueeze(-1) - t_rows.unsqueeze(-2)),
                -torch.abs(t_cols.unsqueeze(-1) - t_cols.unsqueeze(-2)))

    def _gamma_b(self, x, strength):
        raw = self.aperture_scale_proj(strength.to(device=x.device, dtype=x.dtype))
        return torch.exp(torch.clamp(raw, min=-4.0, max=4.0)).view(x.size(0), 1, self.num_heads, 1, 1)

    def forward(self, x, defocus, strength):
        b, h, w, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling
        v = self.v_proj(x)
        lepe = self.lepe(v)
        q = q.view(b, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)   # b, heads, h, w, d
        k = k.view(b, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

        bias_rows, bias_cols = self._defocus_bias(defocus)
        scale = self._gamma_b(x, strength) * self.defocus_alpha.view(1, 1, self.num_heads, 1, 1)   # gamma_b * alpha

        v = v.reshape(b, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)            # b, h, heads, w, d
        attn = torch.softmax(q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) + scale * bias_rows, dim=-1)
        v = (attn @ v).permute(0, 3, 2, 1, 4)                                         # b, w, heads, h, d
        attn = torch.softmax(q.permute(0, 3, 1, 2, 4) @ k.permute(0, 3, 1, 2, 4).transpose(-1, -2)
                             + scale * bias_cols, dim=-1)
        out = (attn @ v).permute(0, 3, 1, 2, 4).flatten(-2, -1)                       # b, h, w, c
        return self.out_proj(out + lepe)


class DefocusAwareAttentionBlock(nn.Module):
    """Defocus-aware attention block (DAAB, Fig. 3C): depthwise positional convolution, defocus-bias attention, FFN."""

    def __init__(self, embed_dim, num_heads, ffn_dim, drop_path=0.0):
        super().__init__()
        self.pos = DWConv2d(embed_dim, 3, 1)
        self.attention_layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = DefocusBiasAttention(embed_dim, num_heads)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, defocus, strength):
        x = x + self.pos(x)
        x = x + self.drop_path(self.attention(self.attention_layer_norm(x), defocus, strength))
        return x + self.drop_path(self.ffn(self.final_layer_norm(x)))


class DAABStack(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, ffn_dim, drop_path):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DefocusAwareAttentionBlock(embed_dim, num_heads, ffn_dim, drop_path[i]) for i in range(depth)])

    def forward(self, x, defocus, strength):
        defocus = F.interpolate(defocus.to(x.dtype), size=x.shape[1:3], mode="bilinear", align_corners=False)
        for daab in self.blocks:
            x = daab(x, defocus, strength)
        return x


class DefocusResidualGroup(nn.Module):
    """Defocus residual group (DRG, Fig. 3B): N DAABs, then a 3x3 convolution over the features and M_C, with a
    residual connection."""

    def __init__(self, embed_dim, cond_channels, depth, num_heads, ffn_dim, drop_path):
        super().__init__()
        self.residual_group = DAABStack(embed_dim, depth, num_heads, ffn_dim, drop_path)
        self.conv = nn.Conv2d(embed_dim + cond_channels, embed_dim, 3, 1, 1)

    def forward(self, x, cond, defocus, strength):
        y = self.residual_group(x, defocus, strength).permute(0, 3, 1, 2)
        return self.conv(torch.cat((y, cond), dim=1)).permute(0, 2, 3, 1) + x
