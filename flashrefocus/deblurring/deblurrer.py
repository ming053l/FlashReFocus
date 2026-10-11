"""Stage I end to end: blurred image -> restored all-in-focus image (one-step diffusion, then the decoder refiner or the
SD2.1 VAE decoder). Images are 1 x 3 x H x W in [-1, 1], H and W multiples of 8."""
import torch

from .. import paths
from .decoder_refiner import load_decoder_refiner
from .one_step_diffusion import load_one_step_diffusion


class Deblurrer:
    def __init__(self, model, refiner=None):
        self.model, self.refiner = model, refiner
        self.scaling_factor = model.vae.config.scaling_factor
        self.prompt_embedding = model.empty_prompt_embedding()

    @classmethod
    def from_pretrained(cls, checkpoint=paths.STAGE1, refiner=paths.REFINER, detail_weight=1.25,
                        refiner_dtype=torch.bfloat16, merge_vae_lora=False):
        """The released Stage I; refiner=None decodes with the SD2.1 VAE decoder instead."""
        model = load_one_step_diffusion(checkpoint, detail_weight=detail_weight, merge_vae_lora=merge_vae_lora)
        return cls(model, load_decoder_refiner(refiner, dtype=refiner_dtype) if refiner is not None else None)

    @property
    def detail_weight(self):
        return self.model.detail_weight

    @detail_weight.setter
    def detail_weight(self, omega):
        self.model.detail_weight = omega

    @torch.no_grad()
    def restored_latent(self, x):
        """z_hat / scaling_factor, as the VAE decoder takes it."""
        z = self.model.vae.encode(x.to(self.model.weight_dtype)).latent_dist.mode() * self.scaling_factor
        emb = self.prompt_embedding.expand(z.shape[0], -1, -1)
        return (z - self.model.predict_noise(z, emb)) / self.scaling_factor

    @torch.no_grad()
    def __call__(self, x, refiner_tile=0, refiner_overlap=64):
        """Restored image in [-1, 1] (not clamped); refiner_tile > 0 runs the refiner on overlapping tiles."""
        latent = self.restored_latent(x)
        if self.refiner is None:
            return self.model.vae.decode(latent).sample
        dtype = next(self.refiner.parameters()).dtype
        x, latent = x.to(dtype), latent.to(dtype)
        if refiner_tile > 0:
            return refine_in_tiles(self.refiner, x, latent, refiner_tile, refiner_overlap)
        return self.refiner(x, latent)


@torch.no_grad()
def refine_in_tiles(refiner, x, latent, tile, overlap):
    """The refiner on overlapping pixel tiles (multiples of 8), blended with linear ramps; bounds memory."""
    _, _, H, W = x.shape
    tile, overlap = tile - tile % 8, overlap - overlap % 8
    starts = lambda n: sorted({min(p, max(0, n - tile)) for p in range(0, max(1, n - overlap), tile - overlap)})
    out = torch.zeros_like(x, dtype=torch.float32)
    weight_sum = torch.zeros((1, 1, H, W), device=x.device)
    ramp = torch.linspace(0, 1, overlap + 2, device=x.device)[1:-1]
    for t in starts(H):
        for l in starts(W):
            b, r = min(t + tile, H), min(l + tile, W)
            tile_out = refiner(x[:, :, t:b, l:r], latent[:, :, t // 8:b // 8, l // 8:r // 8]).float()
            wy, wx = torch.ones(b - t, device=x.device), torch.ones(r - l, device=x.device)
            if overlap:
                if t > 0: wy[:overlap] = ramp
                if b < H: wy[-overlap:] = ramp.flip(0)
                if l > 0: wx[:overlap] = ramp
                if r < W: wx[-overlap:] = ramp.flip(0)
            w = (wy[:, None] * wx[None, :])[None, None]
            out[:, :, t:b, l:r] += tile_out * w
            weight_sum[:, :, t:b, l:r] += w
    return out / weight_sum.clamp_min(1e-6)
