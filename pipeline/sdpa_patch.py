"""Route the Stage II attention through torch.nn.functional.scaled_dot_product_attention (SDPA).

apply() replaces at runtime, without editing the submodules:
  * Depth Anything V2 (DINOv2) Attention.forward: plain softmax attention -> SDPA (Flash / memory-efficient kernels);
  * DMADefocusBiasAttention.forward (renderer): both axial passes -> SDPA with the additive defocus bias as a float
    attn_mask (memory-efficient kernel; Flash does not take a float bias) and scale = 1, since k is pre-scaled.
Both compute the same result as the originals up to floating-point reordering."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402


def _da2_forward(self, x, attn_bias=None):
    assert attn_bias is None
    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    x = F.scaled_dot_product_attention(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(x))


def _dma_forward(self, x, defocus_map, bokeh_strength=None, attention_rel_pos=None):
    bsz, h, w, _ = x.size()
    q = self.q_proj(x)
    k = self.k_proj(x) * self.scaling
    v = self.v_proj(x)
    lepe = self.lepe(v)
    qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
    kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

    bias_w, bias_h = self._build_defocus_bias(self._defocus_to_tokens(defocus_map))
    scale = self._aperture_to_head_scale(x, bokeh_strength) * self.defocus_alpha.view(1, 1, self.num_heads, 1, 1)

    v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)                  # b, h, heads, w, d
    v = F.scaled_dot_product_attention(qr.transpose(1, 2), kr.transpose(1, 2), v,
                                       attn_mask=(scale * bias_w).to(q.dtype), scale=1.0)
    v = v.permute(0, 3, 2, 1, 4)                                                          # b, w, heads, h, d
    out = F.scaled_dot_product_attention(qr.permute(0, 3, 1, 2, 4), kr.permute(0, 3, 1, 2, 4), v,
                                         attn_mask=(scale * bias_h).to(q.dtype), scale=1.0)
    out = out.permute(0, 3, 1, 2, 4).flatten(-2, -1)
    return self.out_proj(out + lepe)


def apply(da2=True, renderer=True):
    """renderer=False keeps the renderer's own attention: its short axial rows with a float bias run slower through SDPA
    (0.31 s against 0.26 s at 1536x1536 in our measurement), while DA2 runs 4.2x faster."""
    paths.add_to_path()
    if da2:
        from depth_anything_v2.dinov2_layers import attention as da2_attn
        da2_attn.Attention.forward = _da2_forward
        da2_attn.MemEffAttention.forward = _da2_forward
    if renderer:
        from method import blocks_dma
        blocks_dma.DMADefocusBiasAttention.forward = _dma_forward
