"""Stage II renderer R_psi (Sec. 3.2, Fig. 3): a NAF U-Net whose bottleneck stacks defocus residual groups (DRGs).

Inputs, with the notation of the paper: the all-in-focus image I_AiF, the blur strength K in [0, 1], the depth
D in [0, 1] (1 = far; inverted, min-max normalized Depth Anything V2 output) and the focal depth D_f = D(p_f).
The condition maps M_C (Eq. 6) are the signed defocus M^s = D - D_f (Eq. 5), the mask M^fg = 1[M^s <= 0] of the focal
plane and what lies before it (dilated by a 5x5 max filter) and the constant map K; every NAF block and every DRG
reads M_C at its resolution, and the defocus-bias attention reads the magnitude M = |M^s|. The output is
I_bokeh = I_AiF + Delta I."""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

from method.blocks import NAFBlock, DefocusResidualGroup

COND_CHANNELS = 3   # M_C = [M^s, M^fg, K]
RGB_MEAN = (0.4488, 0.4371, 0.4040)


class Renderer(nn.Module):
    def __init__(self, u_width=16, u_depth=2, embed_dim=96, num_groups=3, group_depth=3, num_heads=3, mlp_ratio=2,
                 drop_path_rate=0.1, noise_level=0.05):
        super().__init__()
        self.u_depth = u_depth
        self.noise_level = noise_level
        self.register_buffer("mean", torch.tensor(RGB_MEAN).view(1, 3, 1, 1), persistent=False)

        # input: I_AiF, M_C and the K map once more
        self.in_stage = nn.Conv2d(3 + COND_CHANNELS + 1, u_width, 3, 1, 1)
        self.in_stage_2 = nn.Conv2d(u_width, u_width, 3, 1, 1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        chan = u_width
        for _ in range(u_depth):
            self.encoders.append(nn.ModuleList([NAFBlock(chan, COND_CHANNELS)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        # bottleneck: DRGs on channels-last tokens (parameter names follow the released checkpoint)
        self.conv_prep = nn.Conv2d(chan, embed_dim, 3, 1, 1)
        self.defocus_fuse = nn.Conv2d(embed_dim + COND_CHANNELS, embed_dim, 3, 1, 1)
        self.patch_embed = nn.Module()
        self.patch_embed.norm = nn.LayerNorm(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_groups * group_depth)]
        self.blocks = nn.ModuleList([                                       # the DRGs
            DefocusResidualGroup(embed_dim, COND_CHANNELS, group_depth, num_heads, mlp_ratio * embed_dim,
                                 dpr[i * group_depth:(i + 1) * group_depth]) for i in range(num_groups)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim + COND_CHANNELS, chan, 3, 1, 1)

        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for level in range(u_depth):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, 2 * chan, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            n = 4 if level == u_depth - 1 else 2
            self.decoders.append(nn.ModuleList([NAFBlock(chan, COND_CHANNELS) for _ in range(n)]))
        self.out_stage = nn.Conv2d(u_width + COND_CHANNELS, 3, 3, 1, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _noise(self, x):
        return x + torch.randn_like(x) * self.noise_level if self.training and self.noise_level > 0 else x

    @staticmethod
    def _pad(x, multiple):
        h, w = x.shape[-2:]
        ph, pw = (multiple - h % multiple) % multiple, (multiple - w % multiple) % multiple
        if ph == 0 and pw == 0:
            return x
        return F.pad(x, (0, pw, 0, ph), "reflect" if ph < h and pw < w else "replicate")

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _constant_map(x, strength):
        """K as a constant map of the size of x."""
        return strength.to(device=x.device, dtype=x.dtype).view(x.size(0), 1, 1, 1).expand(-1, -1, *x.shape[-2:])

    @staticmethod
    def _cond_at(cond, level):
        """M_C at U-Net level `level` (resolution 1 / 2^level)."""
        return F.interpolate(cond, scale_factor=1 / 2 ** level, mode="bilinear", align_corners=False)

    def forward(self, image, strength, depth, focal_depth):
        """image I_AiF (b, 3, h, w) in [0, 1]; strength K (b, 1); depth D (b, 1, h, w); focal_depth D_f (b, 1).
        Returns I_bokeh (b, 3, h, w)."""
        h, w = image.shape[-2:]
        strength = strength.view(-1, 1)
        multiple = 2 ** self.u_depth
        image, depth = self._pad(image, multiple), self._pad(depth.to(image), multiple)
        x = image - self.mean

        signed_defocus = depth - focal_depth.to(image).view(-1, 1, 1, 1)                                  # M^s
        front_mask = F.max_pool2d((signed_defocus < 1e-3).to(signed_defocus.dtype), 5, stride=1, padding=2)  # M^fg
        cond = torch.cat((self._resize(signed_defocus, x.shape[-2:]), front_mask, self._constant_map(x, strength)),
                         dim=1)                                                                            # M_C

        x = self.in_stage_2(self.in_stage(torch.cat((x, cond, self._constant_map(x, strength)), dim=1)))
        skips = []
        for level, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            for block in encoder:
                x = block(x, self._cond_at(cond, level))
            skips.append(x)
            x = down(x)

        feat = self.conv_prep(self._noise(x))
        defocus = self._resize(signed_defocus.abs().to(feat.dtype), feat.shape[-2:]).clamp(0.0, 1.0)      # M
        cond_bottleneck = self._cond_at(cond, self.u_depth)
        feat = self.defocus_fuse(torch.cat((feat, cond_bottleneck), dim=1))
        tokens = self.patch_embed.norm(feat.permute(0, 2, 3, 1))                                           # b, h, w, c
        for drg in self.blocks:
            tokens = drg(tokens, cond_bottleneck, defocus, strength)
        feat = self.conv_after_body(self.norm(tokens).permute(0, 3, 1, 2)) + feat
        x = self._noise(x + self.conv_last(torch.cat((feat, cond_bottleneck), dim=1)))

        for i, (decoder, up, skip) in enumerate(zip(self.decoders, self.ups, skips[::-1])):
            x = up(x) + skip
            for block in decoder:
                x = block(x, self._cond_at(cond, self.u_depth - 1 - i))
        residual = self.out_stage(torch.cat((x, cond), dim=1)) + self.mean                                # Delta I
        return (residual + image)[:, :, :h, :w]


def load_state(model, path):
    """Load a training checkpoint ({"model": state_dict}) or a bare state_dict strictly. Older checkpoints also store
    the table of a relative-position bias ("Relpos") that the defocus-bias attention does not use."""
    state = torch.load(path, map_location="cpu")
    state = state.get("model", state)
    state = {k.removeprefix("module."): v for k, v in state.items() if ".Relpos." not in k}
    model.load_state_dict(state, strict=True)
    return model


def load_renderer(path, device="cuda"):
    return load_state(Renderer(), path).to(device).eval()
