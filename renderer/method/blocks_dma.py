from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import checkpoint
from timm.layers import DropPath
from torchvision.ops import deform_conv2d

from method.blocks import FeedForwardNetwork
from method.nn_util import (
    ChannelEmbeddingCompression,
    DWConv2d,
    DynRelPos2d,
    IdentityMod,
    PatchEmbedIR,
    PatchUnEmbedIR,
)


class DeformableLEPE(nn.Module):
    def __init__(self, dim, kernel_size=5, stride=1, padding=2, max_offset=8.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_offset = float(max_offset)

        self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.offset_conv = nn.Conv2d(
            dim,
            2 * kernel_size * kernel_size,
            3,
            1,
            1,
        )
        self.mask_conv = nn.Conv2d(dim, kernel_size * kernel_size, 3, 1, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="linear")
        nn.init.constant_(self.bias, 0.0)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        nn.init.constant_(self.mask_conv.weight, 0.0)
        nn.init.constant_(self.mask_conv.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        x: (b h w c)
        """
        x = x.permute(0, 3, 1, 2).contiguous()
        offset = self.max_offset * torch.tanh(self.offset_conv(x))
        mask = 2.0 * torch.sigmoid(self.mask_conv(x))
        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask,
        )
        return x.permute(0, 2, 3, 1).contiguous()


class DMApertureAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1, defocus_token_dim=None):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim * self.factor, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)

        if defocus_token_dim is None:
            defocus_token_dim = max(embed_dim // 4, 1)
        self.defocus_token_dim = defocus_token_dim
        self.defocus_head_proj = nn.Linear(defocus_token_dim, num_heads, bias=True)
        self.defocus_alpha = nn.Parameter(torch.ones(num_heads))
        self.aperture_scale_proj = nn.Linear(1, num_heads, bias=True)
        self.rel_gate_scale = nn.Parameter(torch.ones(num_heads))
        self.rel_gate_bias = nn.Parameter(torch.full((num_heads,), 4.0))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_normal_(self.defocus_head_proj.weight)
        nn.init.constant_(self.defocus_head_proj.bias, 0.0)
        nn.init.constant_(self.aperture_scale_proj.weight, 0.0)
        nn.init.constant_(self.aperture_scale_proj.bias, 0.0)

    def _defocus_to_tokens(self, defocus_map: torch.Tensor):
        if defocus_map.ndim != 4 or defocus_map.size(1) != 1:
            raise ValueError(
                "defocus_map must have shape [B, 1, H, W], got "
                f"{tuple(defocus_map.shape)}"
            )

        scales = torch.linspace(
            0.1,
            1.0,
            steps=self.defocus_token_dim,
            dtype=defocus_map.dtype,
            device=defocus_map.device,
        ).view(1, self.defocus_token_dim, 1, 1)
        d_feat = defocus_map.repeat(1, self.defocus_token_dim, 1, 1) * scales
        return d_feat.permute(0, 2, 3, 1).contiguous()

    def _build_defocus_bias(self, defocus_tokens: torch.Tensor):
        d_head = self.defocus_head_proj(defocus_tokens)

        d_head_w = d_head.permute(0, 1, 3, 2).contiguous()
        defocus_bias_w = -torch.abs(d_head_w.unsqueeze(-1) - d_head_w.unsqueeze(-2))

        d_head_h = d_head.permute(0, 2, 3, 1).contiguous()
        defocus_bias_h = -torch.abs(d_head_h.unsqueeze(-1) - d_head_h.unsqueeze(-2))

        return defocus_bias_w, defocus_bias_h

    def _aperture_to_head_scale(
        self, x: torch.Tensor, bokeh_strength: torch.Tensor | None
    ):
        bsz = x.size(0)
        if bokeh_strength is None:
            return x.new_ones((bsz, 1, self.num_heads, 1, 1))

        if bokeh_strength.dim() == 0:
            bokeh_strength = bokeh_strength.view(1)
        if bokeh_strength.dim() == 1:
            bokeh_strength = bokeh_strength[:, None]
        elif bokeh_strength.dim() == 4:
            bokeh_strength = bokeh_strength.mean(dim=(2, 3))

        if bokeh_strength.dim() != 2 or bokeh_strength.size(1) != 1:
            raise ValueError(
                "bokeh_strength must have shape [B], [B, 1], or [B, 1, H, W], got "
                f"{tuple(bokeh_strength.shape)}"
            )
        if bokeh_strength.size(0) != bsz:
            raise ValueError(
                "bokeh_strength batch size does not match feature batch size: "
                f"{bokeh_strength.size(0)} vs {bsz}"
            )

        raw_scale = self.aperture_scale_proj(
            bokeh_strength.to(device=x.device, dtype=x.dtype)
        )
        head_scale = torch.exp(torch.clamp(raw_scale, min=-4.0, max=4.0))
        return head_scale.view(bsz, 1, self.num_heads, 1, 1)

    def _refine_defocus_bias(
        self,
        rel_pos: torch.Tensor,
        defocus_bias: torch.Tensor,
        aperture_scale: torch.Tensor,
        alpha: torch.Tensor,
        axis_size: int,
    ):
        rel_pos = rel_pos.unsqueeze(1).expand(-1, axis_size, -1, -1, -1)
        gate_scale = self.rel_gate_scale.view(1, 1, self.num_heads, 1, 1)
        gate_bias = self.rel_gate_bias.view(1, 1, self.num_heads, 1, 1)
        spatial_gate = torch.sigmoid(rel_pos * gate_scale + gate_bias)
        return spatial_gate * aperture_scale * alpha * defocus_bias

    def forward(
        self,
        x: torch.Tensor,
        defocus_map: torch.Tensor,
        bokeh_strength: torch.Tensor | None = None,
        attention_rel_pos=None,
    ):
        bsz, h, w, _ = x.size()
        mask_h, mask_w = attention_rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

        defocus_tokens = self._defocus_to_tokens(defocus_map)
        defocus_bias_w, defocus_bias_h = self._build_defocus_bias(defocus_tokens)
        alpha = self.defocus_alpha.view(1, 1, self.num_heads, 1, 1)
        aperture_scale = self._aperture_to_head_scale(x, bokeh_strength)

        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        refined_defocus_bias_w = self._refine_defocus_bias(
            rel_pos=mask_w,
            defocus_bias=defocus_bias_w,
            aperture_scale=aperture_scale,
            alpha=alpha,
            axis_size=h,
        )
        mask_w = mask_w.unsqueeze(1).expand(-1, h, -1, -1, -1)
        qk_mat_w = qk_mat_w + mask_w + refined_defocus_bias_w
        qk_mat_w = torch.softmax(qk_mat_w, dim=-1)
        v = qk_mat_w @ v

        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        refined_defocus_bias_h = self._refine_defocus_bias(
            rel_pos=mask_h,
            defocus_bias=defocus_bias_h,
            aperture_scale=aperture_scale,
            alpha=alpha,
            axis_size=w,
        )
        mask_h = mask_h.unsqueeze(1).expand(-1, w, -1, -1, -1)
        qk_mat_h = qk_mat_h + mask_h + refined_defocus_bias_h
        qk_mat_h = torch.softmax(qk_mat_h, dim=-1)
        output = qk_mat_h @ v

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        return self.out_proj(output)


class DMApertureAwareAttentionV2(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1, defocus_token_dim=None):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim * self.factor, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)

        if defocus_token_dim is None:
            defocus_token_dim = max(embed_dim // 4, 1)
        self.defocus_token_dim = defocus_token_dim
        self.defocus_head_proj = nn.Linear(defocus_token_dim, num_heads, bias=True)
        self.aperture_scale_proj = nn.Linear(1, num_heads, bias=True)
        self.rel_gate_scale = nn.Parameter(torch.ones(num_heads))
        self.rel_gate_bias = nn.Parameter(torch.full((num_heads,), 4.0))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_normal_(self.defocus_head_proj.weight)
        nn.init.constant_(self.defocus_head_proj.bias, 0.0)
        nn.init.constant_(self.aperture_scale_proj.weight, 0.0)
        nn.init.constant_(self.aperture_scale_proj.bias, 0.0)

    def _defocus_to_tokens(self, defocus_map: torch.Tensor):
        if defocus_map.ndim != 4 or defocus_map.size(1) != 1:
            raise ValueError(
                "defocus_map must have shape [B, 1, H, W], got "
                f"{tuple(defocus_map.shape)}"
            )

        min_scale = 1.0 / max(float(self.defocus_token_dim), 16.0)
        scales = torch.logspace(
            torch.log10(defocus_map.new_tensor(min_scale)),
            torch.log10(defocus_map.new_tensor(1.0)),
            steps=self.defocus_token_dim,
            dtype=defocus_map.dtype,
            device=defocus_map.device,
        ).view(1, self.defocus_token_dim, 1, 1)
        defocus_map = defocus_map.clamp(0.0, 1.0)
        d_feat = torch.log1p(defocus_map / scales) / torch.log1p(1.0 / scales)
        return d_feat.permute(0, 2, 3, 1).contiguous()

    #     return defocus_bias_w, defocus_bias_h
    def _build_defocus_bias(self, defocus_tokens: torch.Tensor):
        d_head = self.defocus_head_proj(defocus_tokens)

        d_head_w = d_head.permute(0, 1, 3, 2).contiguous()
        defocus_bias_w = -torch.abs(d_head_w.unsqueeze(-1) - d_head_w.unsqueeze(-2))

        d_head_h = d_head.permute(0, 2, 3, 1).contiguous()
        defocus_bias_h = -torch.abs(d_head_h.unsqueeze(-1) - d_head_h.unsqueeze(-2))

        return defocus_bias_w, defocus_bias_h

    def _aperture_to_head_scale(
        self, x: torch.Tensor, bokeh_strength: torch.Tensor | None
    ):
        bsz = x.size(0)
        if bokeh_strength is None:
            return x.new_ones((bsz, 1, self.num_heads, 1, 1))

        if bokeh_strength.dim() == 0:
            bokeh_strength = bokeh_strength.view(1)
        if bokeh_strength.dim() == 1:
            bokeh_strength = bokeh_strength[:, None]
        elif bokeh_strength.dim() == 4:
            bokeh_strength = bokeh_strength.mean(dim=(2, 3))

        if bokeh_strength.dim() != 2 or bokeh_strength.size(1) != 1:
            raise ValueError(
                "bokeh_strength must have shape [B], [B, 1], or [B, 1, H, W], got "
                f"{tuple(bokeh_strength.shape)}"
            )
        if bokeh_strength.size(0) != bsz:
            raise ValueError(
                "bokeh_strength batch size does not match feature batch size: "
                f"{bokeh_strength.size(0)} vs {bsz}"
            )

        raw_scale = self.aperture_scale_proj(
            bokeh_strength.to(device=x.device, dtype=x.dtype)
        )
        head_scale = torch.exp(raw_scale)
        return head_scale.view(bsz, 1, self.num_heads, 1, 1)

    def _refine_defocus_bias(
        self,
        rel_pos: torch.Tensor,
        defocus_bias: torch.Tensor,
        aperture_scale: torch.Tensor,
        axis_size: int,
    ):
        rel_pos = rel_pos.unsqueeze(1).expand(-1, axis_size, -1, -1, -1)

        gate_scale = self.rel_gate_scale.view(1, 1, self.num_heads, 1, 1)
        gate_bias = self.rel_gate_bias.view(1, 1, self.num_heads, 1, 1)

        spatial_gate = torch.sigmoid(rel_pos * gate_scale + gate_bias)

        defocus_bias = (aperture_scale + 1e-3) * defocus_bias * spatial_gate

        return defocus_bias

    def forward(
        self,
        x: torch.Tensor,
        defocus_map: torch.Tensor,
        bokeh_strength: torch.Tensor | None = None,
        attention_rel_pos=None,
    ):
        bsz, h, w, _ = x.size()
        mask_h, mask_w = attention_rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

        defocus_tokens = self._defocus_to_tokens(defocus_map)
        defocus_bias_w, defocus_bias_h = self._build_defocus_bias(defocus_tokens)
        aperture_scale = self._aperture_to_head_scale(x, bokeh_strength)

        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        refined_defocus_bias_w = self._refine_defocus_bias(
            rel_pos=mask_w,
            defocus_bias=defocus_bias_w,
            aperture_scale=aperture_scale,
            axis_size=h,
        )
        mask_w = mask_w.unsqueeze(1).expand(-1, h, -1, -1, -1)
        qk_mat_w = qk_mat_w + mask_w + refined_defocus_bias_w  
        qk_mat_w = torch.softmax(qk_mat_w, dim=-1)
        v = qk_mat_w @ v

        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        refined_defocus_bias_h = self._refine_defocus_bias(
            rel_pos=mask_h,
            defocus_bias=defocus_bias_h,
            aperture_scale=aperture_scale,
            axis_size=w,
        )
        mask_h = mask_h.unsqueeze(1).expand(-1, w, -1, -1, -1)
        qk_mat_h = qk_mat_h + mask_h + refined_defocus_bias_h
        qk_mat_h = torch.softmax(qk_mat_h, dim=-1)
        output = qk_mat_h @ v

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        return self.out_proj(output)


class DMARelPosAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim * self.factor, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        defocus_map: torch.Tensor | None = None,
        bokeh_strength: torch.Tensor | None = None,
        attention_rel_pos=None,
    ):
        bsz, h, w, _ = x.size()
        mask_h, mask_w = attention_rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        mask_w = mask_w.unsqueeze(1).expand(-1, h, -1, -1, -1)
        qk_mat_w = qk_mat_w + mask_w
        qk_mat_w = torch.softmax(qk_mat_w, dim=-1)
        v = qk_mat_w @ v

        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        mask_h = mask_h.unsqueeze(1).expand(-1, w, -1, -1, -1)
        qk_mat_h = qk_mat_h + mask_h
        qk_mat_h = torch.softmax(qk_mat_h, dim=-1)
        output = qk_mat_h @ v

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        return self.out_proj(output)


class DMADefocusBiasAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1, defocus_token_dim=None):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim * self.factor, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)

        if defocus_token_dim is None:
            defocus_token_dim = max(embed_dim // 4, 1)
        self.defocus_token_dim = defocus_token_dim
        self.defocus_head_proj = nn.Linear(defocus_token_dim, num_heads, bias=True)
        self.defocus_alpha = nn.Parameter(torch.ones(num_heads))
        self.aperture_scale_proj = nn.Linear(1, num_heads, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_normal_(self.defocus_head_proj.weight)
        nn.init.constant_(self.defocus_head_proj.bias, 0.0)
        nn.init.constant_(self.aperture_scale_proj.weight, 0.0)
        nn.init.constant_(self.aperture_scale_proj.bias, 0.0)

    def _defocus_to_tokens(self, defocus_map: torch.Tensor):
        if defocus_map.ndim != 4 or defocus_map.size(1) != 1:
            raise ValueError(
                "defocus_map must have shape [B, 1, H, W], got "
                f"{tuple(defocus_map.shape)}"
            )

        min_scale = 1.0 / max(float(self.defocus_token_dim), 16.0)
        scales = torch.logspace(
            torch.log10(defocus_map.new_tensor(min_scale)),
            torch.log10(defocus_map.new_tensor(1.0)),
            steps=self.defocus_token_dim,
            dtype=defocus_map.dtype,
            device=defocus_map.device,
        ).view(1, self.defocus_token_dim, 1, 1)
        defocus_map = defocus_map.clamp(0.0, 1.0)
        d_feat = torch.log1p(defocus_map / scales) / torch.log1p(1.0 / scales)
        return d_feat.permute(0, 2, 3, 1).contiguous()

    def _build_defocus_bias(self, defocus_tokens: torch.Tensor):
        d_head = self.defocus_head_proj(defocus_tokens)

        d_head_w = d_head.permute(0, 1, 3, 2).contiguous()
        defocus_bias_w = -torch.abs(d_head_w.unsqueeze(-1) - d_head_w.unsqueeze(-2))

        d_head_h = d_head.permute(0, 2, 3, 1).contiguous()
        defocus_bias_h = -torch.abs(d_head_h.unsqueeze(-1) - d_head_h.unsqueeze(-2))

        return defocus_bias_w, defocus_bias_h

    def _aperture_to_head_scale(
        self, x: torch.Tensor, bokeh_strength: torch.Tensor | None
    ):
        bsz = x.size(0)
        if bokeh_strength is None:
            return x.new_ones((bsz, 1, self.num_heads, 1, 1))

        if bokeh_strength.dim() == 0:
            bokeh_strength = bokeh_strength.view(1)
        if bokeh_strength.dim() == 1:
            bokeh_strength = bokeh_strength[:, None]
        elif bokeh_strength.dim() == 4:
            bokeh_strength = bokeh_strength.mean(dim=(2, 3))

        if bokeh_strength.dim() != 2 or bokeh_strength.size(1) != 1:
            raise ValueError(
                "bokeh_strength must have shape [B], [B, 1], or [B, 1, H, W], got "
                f"{tuple(bokeh_strength.shape)}"
            )
        if bokeh_strength.size(0) != bsz:
            raise ValueError(
                "bokeh_strength batch size does not match feature batch size: "
                f"{bokeh_strength.size(0)} vs {bsz}"
            )

        raw_scale = self.aperture_scale_proj(
            bokeh_strength.to(device=x.device, dtype=x.dtype)
        )
        head_scale = torch.exp(torch.clamp(raw_scale, min=-4.0, max=4.0))
        return head_scale.view(bsz, 1, self.num_heads, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        defocus_map: torch.Tensor,
        bokeh_strength: torch.Tensor | None = None,
        attention_rel_pos=None,
    ):
        bsz, h, w, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

        defocus_tokens = self._defocus_to_tokens(defocus_map)
        defocus_bias_w, defocus_bias_h = self._build_defocus_bias(defocus_tokens)
        alpha = self.defocus_alpha.view(1, 1, self.num_heads, 1, 1)
        aperture_scale = self._aperture_to_head_scale(x, bokeh_strength)

        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = qk_mat_w + aperture_scale * alpha * defocus_bias_w
        qk_mat_w = torch.softmax(qk_mat_w, dim=-1)
        v = qk_mat_w @ v

        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + aperture_scale * alpha * defocus_bias_h
        qk_mat_h = torch.softmax(qk_mat_h, dim=-1)
        output = qk_mat_h @ v

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        return self.out_proj(output)


class DMApertureAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        drop_path=0.0,
        layerscale=False,
        norm_layer=nn.LayerNorm,
        layer_init_values=1e-5,
        attn_version="v1",
    ):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.attention_layer_norm = norm_layer(self.embed_dim, eps=1e-6)
        if attn_version == "v1":
            self.attention = DMApertureAwareAttention(embed_dim, num_heads)
        elif attn_version == "v2":
            self.attention = DMApertureAwareAttentionV2(embed_dim, num_heads)
        elif attn_version in ("rel_pos", "relpos"):
            self.attention = DMARelPosAttention(embed_dim, num_heads)
        elif attn_version in ("defocus", "defocus_bias"):
            self.attention = DMADefocusBiasAttention(embed_dim, num_heads)
        else:
            raise ValueError(
                f'Unknown attn_version "{attn_version}". Use "v1", "v2", '
                '"rel_pos", or "defocus".'
            )
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = norm_layer(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(
                layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True
            )

    def forward(
        self,
        x: torch.Tensor,
        defocus_map: torch.Tensor,
        bokeh_strength: torch.Tensor | None = None,
        attention_rel_pos=None,
    ):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(
                self.gamma_1
                * self.attention(
                    self.attention_layer_norm(x),
                    defocus_map,
                    bokeh_strength,
                    attention_rel_pos,
                )
            )
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(
                self.attention(
                    self.attention_layer_norm(x),
                    defocus_map,
                    bokeh_strength,
                    attention_rel_pos,
                )
            )
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x


class DMARelPosAttentionBlock(DMApertureAttentionBlock):
    def __init__(self, *args, **kwargs):
        kwargs["attn_version"] = "rel_pos"
        super().__init__(*args, **kwargs)


class DMADefocusBiasAttentionBlock(DMApertureAttentionBlock):
    def __init__(self, *args, **kwargs):
        kwargs["attn_version"] = "defocus"
        super().__init__(*args, **kwargs)


class BasicLayerDMA(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        ffn_dim=96,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        layerscale=False,
        layer_init_values=1e-5,
        init_value=None,
        heads_range=None,
        attn_version="v1",
        **unused_kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.Relpos = DynRelPos2d(
            embed_dim,
            num_heads,
            2 if init_value is None else init_value,
            9 if heads_range is None else heads_range,
        )

        if attn_version in ("rel_then_defocus", "rel_defocus", "rel_dma"):
            split = depth // 2
            attn_versions = [
                "rel_pos" if i < split else "defocus" for i in range(depth)
            ]
        elif isinstance(attn_version, (list, tuple)):
            if len(attn_version) != depth:
                raise ValueError(
                    "attn_version list length must match depth: "
                    f"{len(attn_version)} vs {depth}"
                )
            attn_versions = list(attn_version)
        else:
            attn_versions = [attn_version for _ in range(depth)]

        self.blocks = nn.ModuleList(
            [
                DMApertureAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    layerscale=layerscale,
                    norm_layer=norm_layer,
                    layer_init_values=layer_init_values,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    attn_version=attn_versions[i],
                )
                for i in range(depth)
            ]
        )

    @staticmethod
    def _make_range_factor(x: torch.Tensor, bokeh_strength: torch.Tensor | None):
        bsz = x.size(0)
        if bokeh_strength is None:
            return x.new_ones((bsz,))
        if bokeh_strength.dim() == 0:
            bokeh_strength = bokeh_strength.view(1)
        if bokeh_strength.dim() == 2 and bokeh_strength.size(1) == 1:
            bokeh_strength = bokeh_strength[:, 0]
        elif bokeh_strength.dim() == 4:
            bokeh_strength = bokeh_strength.mean(dim=(1, 2, 3))
        if bokeh_strength.dim() != 1:
            raise ValueError(
                "bokeh_strength must have shape [B], [B, 1], or [B, 1, H, W], got "
                f"{tuple(bokeh_strength.shape)}"
            )
        if bokeh_strength.size(0) != bsz:
            raise ValueError(
                "bokeh_strength batch size does not match feature batch size: "
                f"{bokeh_strength.size(0)} vs {bsz}"
            )
        return bokeh_strength.to(device=x.device, dtype=x.dtype)

    @staticmethod
    def _make_defocus_map(x: torch.Tensor, defocus_map: torch.Tensor | None):
        bsz, h, w, _ = x.size()
        if defocus_map is None:
            return x.new_zeros((bsz, 1, h, w))
        if defocus_map.dim() == 1:
            return (
                defocus_map.to(device=x.device, dtype=x.dtype)
                .view(bsz, 1, 1, 1)
                .expand(-1, -1, h, w)
            )
        if defocus_map.dim() != 4 or defocus_map.size(1) != 1:
            raise ValueError(
                "defocus_map must be [B], [B, 1, H, W], or None, got "
                f"{tuple(defocus_map.shape)}"
            )
        return F.interpolate(
            defocus_map.to(device=x.device, dtype=x.dtype),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x, defocus_map=None, bokeh_strength=None):
        _, h, w, _ = x.size()
        defocus_map = self._make_defocus_map(x, defocus_map)
        range_factor = self._make_range_factor(x, bokeh_strength)
        rel_pos = self.Relpos((h, w), range_factor=range_factor)
        for blk in self.blocks:
            if self.use_checkpoint:
                tmp_blk = partial(
                    blk,
                    defocus_map=defocus_map,
                    bokeh_strength=bokeh_strength,
                    attention_rel_pos=rel_pos,
                )
                x = checkpoint.checkpoint(tmp_blk, x)
            else:
                x = blk(
                    x,
                    defocus_map=defocus_map,
                    bokeh_strength=bokeh_strength,
                    attention_rel_pos=rel_pos,
                )
        return x


class ResidualBlockDMA(nn.Module):
    def __init__(
        self,
        embed_dim,
        embed_dim_next,
        depth,
        num_heads,
        heads_range=None,
        init_value=None,
        ffn_dim=96,
        layerscale=False,
        layer_init_values=1e-5,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        resi_connection="1conv",
        use_pos_map=False,
        pos_map_channels=2,
        attn_version="v1",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.residual_group = BasicLayerDMA(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            init_value=init_value,
            heads_range=heads_range,
            drop_path=drop_path,
            ffn_dim=ffn_dim,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            layerscale=layerscale,
            layer_init_values=layer_init_values,
            attn_version=attn_version,
        )

        embed_dim_conv_in = embed_dim + pos_map_channels if use_pos_map else embed_dim
        if resi_connection == "1conv":
            self.conv = nn.Conv2d(embed_dim_conv_in, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(embed_dim_conv_in, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

        self.patch_embed = PatchEmbedIR(embed_dim=embed_dim)
        self.patch_unembed = PatchUnEmbedIR(embed_dim=embed_dim)
        self.final_op = (
            ChannelEmbeddingCompression(embed_dim, embed_dim_next)
            if embed_dim != embed_dim_next
            else IdentityMod()
        )
        self.use_pos_map = use_pos_map
        self.pos_map_channels = pos_map_channels

    def forward(
        self,
        x,
        x_size,
        pos_map=None,
        defocus_map=None,
        bokeh_strength=None,
        **kwargs,
    ):
        rb_stl_out = self.residual_group(
            x, defocus_map=defocus_map, bokeh_strength=bokeh_strength
        )
        rb_unembed = self.patch_unembed(rb_stl_out, x_size)
        rb_unembed = (
            torch.cat((rb_unembed, pos_map), dim=1) if self.use_pos_map else rb_unembed
        )
        rb_last_conv = self.conv(rb_unembed)
        rb_re_embed = self.patch_embed(rb_last_conv)
        rb_res = rb_re_embed + x
        return self.final_op(rb_res, x_size)


#############################################################################
class NormalResidualBlock(nn.Module):
    def __init__(self, channels):
        super(NormalResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class CMMResidualBlock(nn.Module):
    def __init__(self, channels):
        super(CMMResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.cmm = CMM(channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.cmm(res)
        res = self.gelu(res)
        res = self.conv2(res)
        return x + res


def res_group_fn(channels, num_blocks=4):
    layers = []
    for _ in range(num_blocks):
        layers.append(NormalResidualBlock(channels))

    layers.append(CMMResidualBlock(channels))

    return nn.Sequential(*layers)


class CMM(nn.Module):
    def __init__(self, channels):
        super(CMM, self).__init__()
        self.A_R = nn.Parameter(torch.ones(1, 1, channels))
        self.A_I = nn.Parameter(torch.ones(1, 1, channels))

    def forward(self, x):
        x_reshaped = x.permute(0, 2, 3, 1)

        x_fft = torch.fft.fft(x_reshaped, dim=-1)
        real = x_fft.real
        imag = x_fft.imag

        real = real * self.A_R
        imag = imag * self.A_I

        x_complex = torch.complex(real, imag)
        x_ifft = torch.fft.ifft(x_complex, dim=-1).real

        out = x_ifft.permute(0, 3, 1, 2).contiguous()

        return out


class MResGroup(nn.Module):
    def __init__(self, channels, num_blocks=2):
        super(MResGroup, self).__init__()
        self.split_chns = channels // 2

        self.res_group1 = res_group_fn(self.split_chns, num_blocks)
        self.res_group2 = res_group_fn(self.split_chns, num_blocks)

        self.down = nn.Conv2d(
            self.split_chns, self.split_chns, kernel_size=3, stride=2, padding=1
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        f1, f2 = torch.split(x, self.split_chns, dim=1)

        out1 = self.res_group1(f1)

        out2 = self.down(f2)
        out2 = self.res_group2(out2)
        out2 = self.up(out2)

        return torch.cat([out1, out2], dim=1)
