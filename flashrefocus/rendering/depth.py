"""Depth D for the renderer from Depth Anything V2 (DA-V2): D = 1 - minmax(DA-V2 output), in [0, 1], 1 = far.

The DA-V2 attention runs through scaled_dot_product_attention (same result up to floating-point reordering, 4.2x faster
at 1536x1536); pre- and post-processing run on the GPU."""
import torch
import torch.nn.functional as F
from torch import nn

from .. import paths

DA2_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _sdpa_attention_forward(self, x, attn_bias=None):
    assert attn_bias is None
    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    x = F.scaled_dot_product_attention(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(x))


def use_sdpa_attention():
    """Route the DA-V2 (DINOv2) attention through torch's scaled_dot_product_attention."""
    paths.use_third_party(paths.DEPTH_ANYTHING_V2_CODE)
    from depth_anything_v2.dinov2_layers import attention
    attention.Attention.forward = _sdpa_attention_forward
    attention.MemEffAttention.forward = _sdpa_attention_forward


def load_depth_anything_v2(encoder="vitl", checkpoint=paths.DEPTH_ANYTHING_V2, device="cuda"):
    paths.use_third_party(paths.DEPTH_ANYTHING_V2_CODE)
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(**DA2_CONFIGS[encoder])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def disparity_to_depth(disparity):
    """Relative inverse depth -> D = 1 - minmax(disparity), finite values only (zeros if the map is constant)."""
    d = torch.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = d.min(), d.max()
    d = (d - lo) / (hi - lo) if float(hi - lo) > 1e-12 else torch.zeros_like(d)
    return (1.0 - d).clamp(0, 1)


class DepthEstimator(nn.Module):
    """image (1, 3, H, W) in [0, 1] -> D (1, H, W). DA-V2 runs at `input_size` on the short side (a multiple of 14),
    in fp16 autocast."""

    def __init__(self, model, input_size=1024):
        super().__init__()
        self.model, self.input_size = model, input_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @classmethod
    def from_pretrained(cls, encoder="vitl", checkpoint=paths.DEPTH_ANYTHING_V2, input_size=1024, device="cuda",
                        sdpa=True):
        if sdpa:
            use_sdpa_attention()
        return cls(load_depth_anything_v2(encoder, checkpoint, device), input_size).to(device)

    def _network_size(self, h, w):
        from depth_anything_v2.util.transform import Resize
        s = self.input_size
        tw, th = Resize(width=s, height=s, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                        resize_method="lower_bound").get_size(w, h)
        return th, tw

    @torch.no_grad()
    def forward(self, image):
        h, w = image.shape[-2:]
        x = F.interpolate(image, self._network_size(h, w), mode="bicubic", align_corners=False)
        x = (x - self.mean.to(x)) / self.std.to(x)
        with torch.autocast(image.device.type, dtype=torch.float16):
            disparity = self.model(x)
        disparity = F.interpolate(disparity[:, None].float(), (h, w), mode="bilinear", align_corners=True)[0]
        return disparity_to_depth(disparity)
