"""Decoder refiner (Sec. 3.1, Appendix B.2): the Stable Diffusion 1.5 VAE decoder with fusion modules that inject
features of the blurred input into its up-blocks, recovering structure that VAE decoding loses.

SD1.5 and SD2.1 share the VAE encoder and quant_conv, so the refiner decodes the latents of the SD2.1-based Stage I.
The input enters as (a) the VAE-encoder features at the resolutions of the decoder stages (conv_in, down-blocks 0 and 1,
matched by spatial size) and (b) a guidance map computed from the image and its high-frequency magnitude
(Laplacian-pyramid residual), shared by every fusion module. Each fusion module adds a zero-initialised residual to the
decoder feature, so training starts from the plain decoder. Parameter names follow the released checkpoint."""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import gaussian_blur

from .. import paths
from ..layers import NAFBlock


def high_frequency_residual(image, kernel_size=5, sigma=1.0):
    """image minus its blurred 2x down-up-sampled version (one Laplacian-pyramid level)."""
    low = gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)[:, :, ::2, ::2]
    up = gaussian_blur(F.interpolate(low, size=image.shape[-2:], mode="bilinear", align_corners=False),
                       kernel_size=kernel_size, sigma=sigma)
    return image - up


class FeatureFusion(nn.Module):
    """Decoder feature + fusion of (encoder feature, decoder feature, guidance map)."""

    def __init__(self, decoder_channels, encoder_channels, guidance_channels, depth):
        super().__init__()
        self.in_conv = nn.Conv2d(encoder_channels + decoder_channels, decoder_channels, 3, 1, 1)
        self.dm_blocks = nn.ModuleList([NAFBlock(decoder_channels + guidance_channels) for _ in range(depth)])
        self.out_conv = nn.Conv2d(decoder_channels + guidance_channels, decoder_channels, 3, 1, 1)

    def forward(self, encoder_feat, decoder_feat, guidance):
        if guidance.shape[-2:] != decoder_feat.shape[-2:]:
            guidance = F.interpolate(guidance, size=decoder_feat.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat((self.in_conv(torch.cat((encoder_feat, decoder_feat), dim=1)), guidance), dim=1)
        for block in self.dm_blocks:
            fused = block(fused)
        return self.out_conv(fused) + decoder_feat


def _last_resnet_channels(block):
    return int(block.resnets[-1].out_channels)


class DecoderRefiner(nn.Module):
    ENCODER_STAGES = 2   # down-blocks whose features the decoder can match (the decoder runs at H/4, H/2, H, H)

    def __init__(self, vae, depth=2, guidance_channels=32):
        super().__init__()
        self.vae = vae
        decoder_channels = [_last_resnet_channels(b) for b in vae.decoder.up_blocks]
        encoder_channels = [int(vae.encoder.conv_in.out_channels)] + [_last_resnet_channels(b) for b in vae.encoder.down_blocks]
        n = len(decoder_channels)
        self.fusion_blocks = nn.ModuleList([
            FeatureFusion(dec, encoder_channels[max(0, n - 2 - i)], guidance_channels, depth)
            for i, dec in enumerate(decoder_channels)])
        self.map_init = nn.Sequential(nn.Conv2d(3 + 1, guidance_channels, 3, 1, 1),        # image + high frequency
                                      *[NAFBlock(guidance_channels) for _ in range(depth)])
        for block in self.fusion_blocks:
            nn.init.zeros_(block.out_conv.weight)
            nn.init.zeros_(block.out_conv.bias)

    def encoder_features(self, image):
        """VAE-encoder features at the resolutions of the decoder stages: conv_in and the first down-blocks."""
        with torch.no_grad():
            h = self.vae.encoder.conv_in(image)
            feats = [h.detach()]
            for block in self.vae.encoder.down_blocks[:self.ENCODER_STAGES]:
                h = block(h)
                feats.append(h.detach())
        return feats

    @staticmethod
    def _matching_feature(decoder_feat, encoder_features):
        for feat in reversed(encoder_features):
            if feat.shape[0] == decoder_feat.shape[0] and feat.shape[-2:] == decoder_feat.shape[-2:]:
                return feat
        return None

    def forward(self, image, latent):
        """image: the blurred input in [-1, 1]; latent: the restored latent (unscaled). Returns the image in [-1, 1]."""
        encoder_features = self.encoder_features(image)
        high_freq = high_frequency_residual(image).abs().mean(dim=1, keepdim=True)
        guidance = self.map_init(torch.cat((image, high_freq), dim=1))
        h = self.vae.decoder.mid_block(self.vae.decoder.conv_in(self.vae.post_quant_conv(latent)))
        for block, fusion in zip(self.vae.decoder.up_blocks, self.fusion_blocks):
            h = block(h)
            feat = self._matching_feature(h, encoder_features)
            if feat is not None:
                h = fusion(feat, h, guidance)
        return self.vae.decoder.conv_out(self.vae.decoder.conv_act(self.vae.decoder.conv_norm_out(h)))


def build_decoder_refiner(sd15=paths.SD15):
    from diffusers import AutoencoderKL
    return DecoderRefiner(AutoencoderKL.from_pretrained(str(sd15), subfolder="vae"))


def load_decoder_refiner(path=paths.REFINER, sd15=paths.SD15, device="cuda", dtype=torch.bfloat16):
    """The released refiner, in bf16 as trained. Its checkpoint stores every parameter except the VAE encoder, which
    comes from SD1.5 (only its first two down-blocks run)."""
    refiner = build_decoder_refiner(sd15)
    res = refiner.load_state_dict(torch.load(str(path), map_location="cpu"), strict=False)
    missing = [k for k in res.missing_keys if not k.startswith(("vae.encoder", "vae.quant_conv", "vae.post_quant_conv"))]
    assert not missing and not res.unexpected_keys, (missing[:5], res.unexpected_keys[:5])
    return refiner.to(device, dtype=dtype).eval()
