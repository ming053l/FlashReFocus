import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from .modules import BlockMod, LayerNorm2d
from torchvision.transforms.functional import gaussian_blur


def laplacian_pyramid(img, kernel_size=5, sigma=1.0):
    # Gaussian blur before downsampling
    blurred = gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

    # Downsample
    low = blurred[:, :, ::2, ::2]

    # Upsample
    up = F.interpolate(low, size=img.shape[-2:], mode="bilinear", align_corners=False)

    # Smooth after upsampling
    up = gaussian_blur(up, kernel_size=kernel_size, sigma=sigma)

    # Laplacian residual
    high = img - up

    return high


class VAEFeatureFusion(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        encoder_dim: int | None = None,
        map_hidden_channels: int = 48,
        depth=2,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.encoder_dim = int(encoder_dim if encoder_dim is not None else embed_dim)
        self.in_conv = nn.Conv2d(
            self.encoder_dim + self.embed_dim,
            self.embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # self.gate_conv = nn.Sequential(
        #     nn.Conv2d(map_hidden_channels, embed_dim, 1, 1, 0),
        #     nn.GroupNorm(16, embed_dim),
        #     nn.GELU(),
        #     nn.Conv2d(embed_dim, embed_dim * 2, 3, 1, 1),
        # )
        # nn.init.zeros_(self.gate_conv[-1].weight)
        # nn.init.zeros_(self.gate_conv[-1].bias)
        self.dm_blocks = nn.ModuleList(
            [
                BlockMod(
                    self.embed_dim + map_hidden_channels,
                )
                for _ in range(depth)
            ]
        )
        self.out_conv = nn.Conv2d(
            self.embed_dim + map_hidden_channels,
            self.embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # self.norm = LayerNorm2d(self.embed_dim)
        # self.last_conv = BlockMod(self.embed_dim)

    def forward(
        self,
        encoder_feat: torch.Tensor,
        decoder_feat: torch.Tensor,
        defocus_prior: torch.Tensor,
    ) -> torch.Tensor:
        if defocus_prior.shape[-2:] != decoder_feat.shape[-2:]:
            defocus_prior = F.interpolate(
                defocus_prior,
                size=decoder_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused = torch.cat((encoder_feat, decoder_feat), dim=1)
        fused = self.in_conv(fused)
        fused = torch.cat((fused, defocus_prior), dim=1)
        for blk in self.dm_blocks:
            fused = blk(fused)
        fused = self.out_conv(fused)
        # fused = self.norm(fused)
        # a, b = torch.tanh(self.gate_conv(defocus_prior)).chunk(2, dim=1)
        # fused = fused * (1 + a) + b
        # fused = self.last_conv(fused)
        return fused + decoder_feat


class VAEMultiScaleFusionDecoder(nn.Module):
    """
    Wrapper around a diffusers AutoencoderKL that:
    1. manually runs the VAE encoder to collect multi-scale encoder features
    2. manually runs the decoder
    3. updates decoder features with encoder features of the same spatial size
       while allowing different channel counts
    4. returns the final reconstruction

    The updated decoder feature of each stage is fed back into the next decoder stage.
    """

    def __init__(
        self,
        vae: nn.Module,
        depth=2,
        map_hidden_channels=32,
        use_pixel_weighting=False,
        use_high_freq_info=False,
    ):
        super().__init__()
        self.vae = vae

        decoder_stage_channels = [self._infer_mid_channels()]
        decoder_stage_channels.extend(
            self._infer_block_channels(block) for block in self.vae.decoder.up_blocks
        )
        encoder_stage_channels = self._infer_encoder_stage_channels()
        num_up_blocks = len(self.vae.decoder.up_blocks)
        encoder_channels_for_decoder = [
            encoder_stage_channels[max(0, num_up_blocks - 2 - idx)]
            for idx in range(num_up_blocks)
        ]
        if len(decoder_stage_channels[1:]) != len(encoder_channels_for_decoder):
            raise ValueError(
                "Could not align VAE decoder up-blocks with encoder feature stages: "
                f"{len(decoder_stage_channels[1:])} decoder stages vs "
                f"{len(encoder_channels_for_decoder)} encoder stages."
            )

        # Skip the smallest (mid) stage; start updates from the first up-block stage.
        self.fusion_blocks = nn.ModuleList(
            [
                VAEFeatureFusion(
                    embed_dim=channels,
                    encoder_dim=encoder_channels,
                    map_hidden_channels=map_hidden_channels,
                    depth=depth,
                )
                for channels, encoder_channels in zip(
                    decoder_stage_channels[1:],
                    encoder_channels_for_decoder,
                )
            ]
        )

        self.map_init = nn.Sequential(
            nn.Conv2d(3 + int(use_high_freq_info), map_hidden_channels, 3, 1, 1),
            *[BlockMod(map_hidden_channels) for _ in range(depth)],
        )
        self.use_high_freq_info = use_high_freq_info
        self.zero_init_refiner()
        if use_pixel_weighting:
            self.pixel_weight_predictor = SMDHead(map_hidden_channels)
        self.use_pixel_weighting = use_pixel_weighting

    def zero_init_refiner(self) -> None:
        """Start from the frozen VAE path and make the learned residual initially zero."""
        for fusion_block in self.fusion_blocks:
            nn.init.zeros_(fusion_block.out_conv.weight)
            if fusion_block.out_conv.bias is not None:
                nn.init.zeros_(fusion_block.out_conv.bias)

    def _infer_mid_channels(self) -> int:
        if (
            hasattr(self.vae.decoder.mid_block, "resnets")
            and len(self.vae.decoder.mid_block.resnets) > 0
        ):
            mid_block = self.vae.decoder.mid_block.resnets[-1]
            if hasattr(mid_block, "out_channels"):
                return int(mid_block.out_channels)
            if hasattr(mid_block, "conv2") and hasattr(mid_block.conv2, "out_channels"):
                return int(mid_block.conv2.out_channels)
        return int(self.vae.config.block_out_channels[-1])

    @staticmethod
    def _infer_block_channels(block: nn.Module) -> int:
        if hasattr(block, "resnets") and len(block.resnets) > 0:
            last_resnet = block.resnets[-1]
            if hasattr(last_resnet, "out_channels"):
                return int(last_resnet.out_channels)
            if hasattr(last_resnet, "conv2") and hasattr(
                last_resnet.conv2, "out_channels"
            ):
                return int(last_resnet.conv2.out_channels)
        raise AttributeError(
            "Cannot infer decoder block channels from the provided VAE block."
        )

    def _infer_encoder_stage_channels(self) -> list[int]:
        conv_in = self.vae.encoder.conv_in
        if not hasattr(conv_in, "out_channels"):
            raise AttributeError("Cannot infer encoder conv_in output channels.")

        channels = [int(conv_in.out_channels)]
        for block in self.vae.encoder.down_blocks:
            channels.append(self._infer_block_channels(block))
        return channels

    @staticmethod
    def _find_matching_encoder_feature(
        decoder_feat: torch.Tensor, encoder_features: list[torch.Tensor]
    ) -> torch.Tensor | None:
        for encoder_feat in reversed(encoder_features):
            if (
                encoder_feat.shape[0] == decoder_feat.shape[0]
                and encoder_feat.shape[-2:] == decoder_feat.shape[-2:]
            ):
                return encoder_feat
        return None

    def encode_with_features(
        self, x: torch.Tensor, sample_posterior: bool = True
    ) -> tuple[torch.Tensor, list[torch.Tensor], DiagonalGaussianDistribution]:
        encoder_features: list[torch.Tensor] = []

        with torch.no_grad():
            h = self.vae.encoder.conv_in(x)
            encoder_features.append(h.detach())

            for block in self.vae.encoder.down_blocks:
                h = block(h)
                encoder_features.append(h.detach())

            h = self.vae.encoder.mid_block(h)
            encoder_features.append(h.detach())

            h = self.vae.encoder.conv_norm_out(h)
            h = self.vae.encoder.conv_act(h)
            h = self.vae.encoder.conv_out(h)

            moments = self.vae.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            latent = posterior.sample() if sample_posterior else posterior.mode()
            latent = latent.detach()
        return latent, encoder_features, posterior

    def encode_decoder_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encoder features at the resolutions of the decoder stages only (conv_in, down0, down1)."""
        with torch.no_grad():
            h = self.vae.encoder.conv_in(x)
            feats = [h.detach()]
            for block in self.vae.encoder.down_blocks[:2]:
                h = block(h)
                feats.append(h.detach())
        return feats

    def decode_with_fusions(
        self,
        latent: torch.Tensor,
        encoder_features: list[torch.Tensor],
        defocus_prior: torch.Tensor,
    ) -> torch.Tensor:

        h = self.vae.post_quant_conv(latent)
        h = self.vae.decoder.conv_in(h)
        h = self.vae.decoder.mid_block(h)

        # Do not update the smallest (mid) stage.

        for idx, block in enumerate(self.vae.decoder.up_blocks, start=1):
            h = block(h)
            matched_encoder_feat = self._find_matching_encoder_feature(
                h, encoder_features
            )
            if matched_encoder_feat is not None:
                h = self.fusion_blocks[idx - 1](
                    matched_encoder_feat,
                    h,
                    defocus_prior,
                )
        h = self.vae.decoder.conv_norm_out(h)
        h = self.vae.decoder.conv_act(h)
        h = self.vae.decoder.conv_out(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor | None = None,
        defocus_map: torch.Tensor | None = None,
        sample_posterior: bool = False,
        return_dict: bool = False,
    ):
        if latent is None:
            latent, encoder_features, posterior = self.encode_with_features(
                x, sample_posterior=sample_posterior
            )
        else:
            # With the latent given, only the encoder features the decoder can match are needed: the decoder stages
            # run at H/4, H/2, H, H, matched by spatial size to conv_in, down0, down1. The deeper stages (down2, down3,
            # mid block with its full attention, conv_out, quant_conv) would be computed and discarded; skipping them
            # leaves the output unchanged.
            encoder_features, posterior = self.encode_decoder_features(x), None
            if latent.shape[0] != encoder_features[0].shape[0]:
                raise ValueError(
                    "latent batch size does not match encoder features: "
                    f"{latent.shape[0]} vs {encoder_features[0].shape[0]}"
                )
        if self.use_high_freq_info:
            high_freq_info = laplacian_pyramid(x).abs().mean(dim=1, keepdim=True)
            x_input = torch.cat((x, high_freq_info), dim=1)
        else:
            x_input = x
        defocus_prior = self.map_init(x_input)

        if self.use_pixel_weighting:
            w_pixel = self.pixel_weight_predictor(defocus_prior)
            reconstruction = self.decode_with_fusions(
                latent, encoder_features, defocus_prior
            ) * w_pixel + x * (1 - w_pixel)
        else:
            reconstruction = self.decode_with_fusions(
                latent, encoder_features, defocus_prior
            )

        if return_dict:
            return {
                "reconstruction": reconstruction,
                "latent": latent,
                "posterior": posterior,
                "w_pixel": w_pixel if self.use_pixel_weighting else None,
            }
        return reconstruction


class SMDHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.activation = nn.Sigmoid()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            BlockMod(in_channels),
            BlockMod(in_channels),
            nn.Conv2d(in_channels, 5, 1),
        )

        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, feat):
        pred = self.head(feat)

        eps = 1e-3

        mu0 = self.activation(pred[:, 0:1, :, :])
        mu1 = self.activation(pred[:, 1:2, :, :])

        sigma0 = torch.clamp(self.activation(pred[:, 2:3, :, :]), eps, 1.0)
        sigma1 = torch.clamp(self.activation(pred[:, 3:4, :, :]), eps, 1.0)

        pi0 = self.activation(pred[:, 4:5, :, :])
        pi1 = 1.0 - pi0

        mask = (pi0 / sigma0 > pi1 / sigma1).float()
        disp = mu0 * mask + mu1 * (1.0 - mask)

        return disp
