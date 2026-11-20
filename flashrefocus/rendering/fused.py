"""Fused inference path for the renderer (deployment only; the released weights and outputs are unchanged).

The defocus-bias attention (defocus_bias_attention.py) builds, for every row and then every column of the bottleneck
token grid, the bias B_ij = -|T_D,i - T_D,j| and the full attention matrix in memory: O(L^2) per row or column, which
dominates both time and peak memory at high resolution. Here one Triton kernel does both axial passes as a
FlashAttention-style loop with an online softmax, and computes gamma_b * alpha * B_ij on the fly from the per-token
values T_D, so neither the bias nor the attention matrix is ever stored. It reads q, k, v and T_D in their original
channels-last layout through strides (no transposed copies) and writes the row pass in the layout the column pass reads.

  from flashrefocus.rendering.fused import enable_fused
  enable_fused(renderer)          # every DefocusBiasAttention of the renderer uses the fused kernel on CUDA

Without Triton or on CPU the reference implementation is used.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                      # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:
    @triton.autotune(
        configs=[triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
                 for bm, bn, nw, ns in [(128, 32, 4, 3), (128, 64, 4, 2), (128, 32, 2, 3), (64, 64, 4, 2),
                                        (32, 64, 2, 3), (64, 128, 4, 2)]],
        key=["L", "D"])
    @triton.jit
    def _axial_bias_attention_kernel(
            Q, K, V, T, S, O,
            sq_b, sq_i, sq_l, sq_h,                      # q strides: batch, inner (the other axis), sequence, head
            sk_b, sk_i, sk_l, sk_h,
            sv_b, sv_i, sv_l, sv_h,
            so_b, so_i, so_l, so_h,
            st_b, st_i, st_l, st_h,                      # T_D strides (no channel dimension)
            inner, heads, L,
            D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        z = tl.program_id(1)
        hd = z % heads
        zi = (z // heads) % inner
        zb = z // (heads * inner)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        q_base = Q + zb * sq_b + zi * sq_i + hd * sq_h
        k_base = K + zb * sk_b + zi * sk_i + hd * sk_h
        v_base = V + zb * sv_b + zi * sv_i + hd * sv_h
        t_base = T + zb * st_b + zi * st_i + hd * st_h
        mask_m = offs_m < L
        q = tl.load(q_base + offs_m[:, None] * sq_l + offs_d[None, :], mask=mask_m[:, None], other=0.0)
        t_q = tl.load(t_base + offs_m * st_l, mask=mask_m, other=0.0).to(tl.float32)
        s = tl.load(S + zb * heads + hd).to(tl.float32)  # gamma_b * alpha of this image and head
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        for start_n in range(0, L, BLOCK_N):
            cols = start_n + offs_n
            mask_n = cols < L
            k = tl.load(k_base + cols[:, None] * sk_l + offs_d[None, :], mask=mask_n[:, None], other=0.0)
            t_k = tl.load(t_base + cols * st_l, mask=mask_n, other=0.0).to(tl.float32)
            qk = tl.dot(q, tl.trans(k))                  # fp32 accumulate
            qk = qk - s * tl.abs(t_q[:, None] - t_k[None, :])
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_base + cols[:, None] * sv_l + offs_d[None, :], mask=mask_n[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        acc = acc / l_i[:, None]
        o_base = O + zb * so_b + zi * so_i + hd * so_h
        tl.store(o_base + offs_m[:, None] * so_l + offs_d[None, :], acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def _axial_pass(q, k, v, t, scale, axis):
    """One axial pass. q, k, v: (b, h, w, heads, d) views; t: (b, h, w, heads); scale: (b, heads).
    axis = 2 attends along w (each row), axis = 1 along h (each column). Returns (b, h, w, heads, d) contiguous."""
    b, h, w, heads, d = q.shape
    out = torch.empty((b, h, w, heads, d), device=q.device, dtype=q.dtype)
    other = 1 if axis == 2 else 2
    inner, L = q.shape[other], q.shape[axis]

    def st5(x):
        return x.stride(0), x.stride(other), x.stride(axis), x.stride(3)

    st_t = (t.stride(0), t.stride(other), t.stride(axis), t.stride(3))
    for x in (q, k, v, out):
        assert x.stride(4) == 1
    grid = lambda meta: (triton.cdiv(L, meta["BLOCK_M"]), b * inner * heads)  # noqa: E731
    _axial_bias_attention_kernel[grid](q, k, v, t, scale, out, *st5(q), *st5(k), *st5(v), *st5(out), *st_t,
                                       inner, heads, L, D=d)
    return out


def fused_attention_forward(self, x, defocus, strength):
    """Drop-in forward of DefocusBiasAttention using the fused kernel (same parameters, same output)."""
    if not (HAVE_TRITON and x.is_cuda) or self.training or self.key_dim not in (16, 32, 64, 128):
        return type(self).forward(self, x, defocus, strength)
    b, h, w, c = x.shape
    q = self.q_proj(x)
    k = self.k_proj(x) * self.scaling
    v = self.v_proj(x)
    lepe = self.lepe(v)
    dt = q.dtype
    q5 = q.view(b, h, w, self.num_heads, self.key_dim)
    k5 = k.to(dt).view(b, h, w, self.num_heads, self.key_dim)
    v5 = v.view(b, h, w, self.num_heads, self.key_dim)
    t_d = self.defocus_head_proj(self._log_features(defocus)).float().contiguous()   # T_D: b, h, w, heads
    scale = (self._gamma_b(x, strength).view(b, self.num_heads).float()
             * self.defocus_alpha.float().view(1, self.num_heads)).contiguous()      # gamma_b * alpha: b, heads
    rows = _axial_pass(q5, k5, v5, t_d, scale, axis=2)
    out = _axial_pass(q5, k5, rows, t_d, scale, axis=1)
    return self.out_proj(out.view(b, h, w, c) + lepe)


def enable_fused(renderer):
    """Switch every DefocusBiasAttention in `renderer` to the fused kernel. Returns the renderer."""
    from .defocus_bias_attention import DefocusBiasAttention
    for m in renderer.modules():
        if isinstance(m, DefocusBiasAttention):
            m.forward = fused_attention_forward.__get__(m)
    return renderer
