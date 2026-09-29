import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms.functional import gaussian_blur

from method.blocks_dma import ResidualBlockDMA, MResGroup
from method.model import Bokehlicious as _BaseBokehlicious


class RadianceVirtualization(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=32,
        alpha=3.0,
        beta=5.0,
        bright_threshold=0.95,
        bright_softness=0.02,
        eps=1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bright_threshold = bright_threshold
        self.bright_softness = bright_softness
        self.eps = eps

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, 1, 1),
        )

    def forward(self, x):
        x = x.clamp(0.0, 1.0)
        base_radiance = F.softplus(self.net(x)) + self.eps
        max_rgb = x.amax(dim=1, keepdim=True)
        bright_radiance = self.alpha * torch.pow(max_rgb + self.eps, self.beta)
        bright_gate = torch.sigmoid(
            (max_rgb - self.bright_threshold) / self.bright_softness
        )
        return base_radiance * (1.0 - bright_gate) + bright_radiance * bright_gate


class Bokehlicious(_BaseBokehlicious):
    """
    Bokehlicious variant whose deep feature extractor uses DMA attention.

    The DMA attention defocus map comes from bokeh_strength_map. depth_map is
    still used to estimate a relative focal plane for defocus-map refinement.
    """

    def __init__(
        self,
        *args,
        use_mres_after_in_stage2: bool = False,
        use_mres_before_out_stage: bool = False,
        mres_num_blocks: int = 2,
        attn_version: str = "v1",
        out_stage_use_pos_map: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            out_stage_use_pos_map=out_stage_use_pos_map,
            **kwargs,
        )
        self.attn_version = attn_version
        self.after_in_stage2_mres = (
            MResGroup(self.u_width, mres_num_blocks)
            if use_mres_after_in_stage2
            else nn.Identity()
        )
        self.before_out_stage_mres = (
            MResGroup(self.u_width, mres_num_blocks)
            if use_mres_before_out_stage
            else nn.Identity()
        )
        self.defocus_fuse = nn.Conv2d(
            self.embed_dims[0] + 2, self.embed_dims[0], 3, 1, 1
        )
        self.gen_defocus_scale = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.SELU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.gen_defocus_scale[-2].weight)
        nn.init.zeros_(self.gen_defocus_scale[-2].bias)
        self.rad_viz = RadianceVirtualization()

        self._replace_dfe_blocks_with_dma()
        self.noise_level = 0.01

    def _replace_dfe_blocks_with_dma(self):
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))
        ]

        self.blocks = nn.ModuleList()
        drop_offset = 0
        for i_block in range(self.num_blocks):
            depth = self.depths[i_block]
            embed_dim = self.embed_dims[i_block]
            embed_dim_next = (
                self.embed_dims[i_block + 1]
                if i_block < self.num_blocks - 1
                else embed_dim
            )
            self.blocks.append(
                ResidualBlockDMA(
                    embed_dim=embed_dim,
                    embed_dim_next=embed_dim_next,
                    depth=depth,
                    num_heads=self.num_heads[i_block],
                    init_value=self.init_values[i_block],
                    heads_range=self.heads_ranges[i_block],
                    ffn_dim=int(self.mlp_ratios[i_block] * embed_dim),
                    drop_path=dpr[drop_offset : drop_offset + depth],
                    norm_layer=self.norm_layer,
                    use_checkpoint=self.use_checkpoints[i_block],
                    layerscale=self.layerscales[i_block],
                    layer_init_values=self.layer_init_values,
                    use_pos_map=self.positional_dfe,
                    attn_version=self.attn_version,
                )
            )
            drop_offset += depth

        self.blocks.apply(self._init_weights)

    @staticmethod
    def _resize_like(x: Tensor, value: Tensor, name: str) -> Tensor:
        if value is None:
            raise ValueError(f"{name} is required.")
        if value.dim() != 4 or value.size(1) != 1:
            raise ValueError(
                f"{name} must have shape [B, 1, H, W], got " f"{tuple(value.shape)}"
            )
        if value.size(0) != x.size(0):
            raise ValueError(
                f"{name} batch size does not match feature batch size: "
                f"{value.size(0)} vs {x.size(0)}"
            )
        return F.interpolate(
            value.to(device=x.device, dtype=x.dtype),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _as_batch_scalar(x: Tensor, value: Tensor, name: str) -> Tensor:
        if value is None:
            raise ValueError(f"{name} is required.")
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        value = value.to(device=x.device, dtype=x.dtype)
        if value.dim() == 0:
            value = value.view(1)
        if value.dim() == 1:
            value = value[:, None]
        elif value.dim() == 4:
            value = value.mean(dim=(2, 3))

        if value.dim() != 2 or value.size(1) != 1:
            raise ValueError(
                f"{name} must have shape [B], [B, 1], or [B, 1, H, W], got "
                f"{tuple(value.shape)}"
            )
        if value.size(0) != x.size(0):
            raise ValueError(
                f"{name} batch size does not match feature batch size: "
                f"{value.size(0)} vs {x.size(0)}"
            )
        return value

    @classmethod
    def _make_aperture_embedding_map(
        cls,
        x: Tensor,
        bokeh_strength: Tensor,
    ) -> Tensor:
        aperture = cls._as_batch_scalar(x, bokeh_strength, "bokeh_strength")
        aperture = aperture.view(x.size(0), 1, 1, 1)
        aperture = aperture.expand(-1, -1, x.shape[-2], x.shape[-1])
        return aperture

    @staticmethod
    def estimate_d_from_abs_map_batch(map1, map2, tol=1e-6):
        B = map1.shape[0]
        cand_plus = (map1 + map2).flatten(1)
        cand_minus = (map1 - map2).flatten(1)
        candidates = torch.cat([cand_plus, cand_minus], dim=1)
        q = torch.round(candidates / tol).long()
        mode_q = torch.mode(q, dim=1).values
        mask = q == mode_q[:, None]
        d_est = (candidates * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        return d_est.view(B, 1)

    @classmethod
    def _make_focal_distance(
        cls,
        x: Tensor,
        disp_focus: Tensor,
    ) -> Tensor:
        disp_focus = cls._as_batch_scalar(x, disp_focus, "disp_focus")
        return disp_focus.view(x.size(0), 1, 1, 1)

    @staticmethod
    def _resize_by_depth(value: Tensor, depth: int, enabled: bool = True) -> Tensor:
        if not enabled or value is None:
            return None
        return F.interpolate(
            value,
            scale_factor=1 / 2**depth,
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _run_blocks(blocks: nn.ModuleList, x: Tensor, pos_map: Tensor = None) -> Tensor:
        for block in blocks:
            x = block(x, pos_map=pos_map)
        return x

    def forward_features(
        self,
        x,
        pos_map=None,
        bokeh_strength=None,
        defocus_map: Tensor = None,
    ):
        x_size = (x.shape[2], x.shape[3])
        if defocus_map is None:
            raise ValueError("defocus_map is required for DMA forward_features.")

        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(
                x,
                x_size,
                pos_map=pos_map,
                defocus_map=defocus_map,
                bokeh_strength=bokeh_strength,
            )

        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def forward(
        self,
        source: Tensor,
        bokeh_strength: Tensor = None,
        pos_map: Tensor = None,
        bokeh_strength_map: Tensor = None,
        depth_map: Tensor = None,
        disp_focus: Tensor = None,
        **kwargs,
    ):
        source, pos_map, bokeh_strength_map, depth_map, padding = (
            self._pad_forward_inputs(
                source=source,
                pos_map=pos_map,
                bokeh_strength_map=bokeh_strength_map,
                depth_map=depth_map,
            )
        )

        self.mean = self.mean.type_as(source)
        source_mean = (source - self.mean) * self.img_range

        if disp_focus is not None:
            focal_d = self._make_focal_distance(source, disp_focus)
        else:
            focal_d = self.estimate_d_from_abs_map_batch(depth_map, bokeh_strength_map)
            focal_d = focal_d.view(-1, 1, 1, 1)
        signed_defocus = depth_map - focal_d
        defocus_scale = self.gen_defocus_scale(
            torch.cat((gaussian_blur(source_mean, 5), signed_defocus), dim=1)
        )
        defocus_pos_map = torch.cat((bokeh_strength_map, signed_defocus), dim=1)

        x = (
            torch.cat((source_mean, defocus_pos_map), dim=1)
            if self.in_stage_use_pos_map
            else source
        )
        x = (
            torch.cat((x, self._make_aperture_embedding_map(x, bokeh_strength)), dim=1)
            if self.in_stage_use_bokeh_strength_map
            else x
        )

        x = self.in_stage(x)
        x = self.in_stage_2(x)
        x = self.after_in_stage2_mres(x)

        encs = []
        for encoder, down, _, depth in zip(
            self.encoders, self.downs, self.enc_blks_use_pos_map, range(0, self.u_depth)
        ):
            pos_map_e = self._resize_by_depth(pos_map, depth)
            x = self._run_blocks(encoder, x, pos_map_e)
            encs.append(x)
            x = down(x)

        x_prep = self.conv_prep(self.add_white_noise(x))

        # focus_tolerance = 0.03
        # focus_softness = 0.01
        # soft_focus_mask = torch.sigmoid(
        #     (focus_tolerance - bokeh_strength_map) / focus_softness
        # )
        # soft_focus_mask = F.max_pool2d(
        #     soft_focus_mask,
        #     kernel_size=5,
        #     stride=1,
        #     padding=2,
        # )
        # dfe_focus_mask = F.interpolate(
        #     soft_focus_mask,
        #     size=x_prep.shape[-2:],
        #     mode="bilinear",
        # )

        dfe_defocus_map = self._resize_like(
            x_prep,
            bokeh_strength_map,
            "bokeh_strength_map",
        )

        defocus_scale = self._resize_like(x_prep, defocus_scale, "defocus_scale")
        dfe_defocus_map = ((1 + defocus_scale) * dfe_defocus_map).clamp(0.0, 1.0)
        # protect small focus regions
        # dfe_defocus_map = dfe_defocus_map * (1.0 - 0.8*dfe_focus_mask.clamp(0.0, 1.0))
        # dfe_defocus_map = dfe_defocus_map.clamp(0.0, 1.0)

        radiance = self.rad_viz(self._resize_by_depth(source, self.u_depth))
        # x_prep = self.defocus_fuse(
        #     torch.cat((x_prep, radiance, dfe_defocus_map), dim=1)
        # )
        apt_emb_l = self._make_aperture_embedding_map(radiance, bokeh_strength)
        x_prep = self.defocus_fuse(torch.cat((x_prep, radiance, apt_emb_l), dim=1))

        pos_map_e = self._resize_by_depth(pos_map, self.u_depth)
        x_after_body = self.forward_features(
            x_prep,
            bokeh_strength=bokeh_strength,
            defocus_map=dfe_defocus_map,
            pos_map=pos_map_e,
        )
        res = self.conv_after_body(x_after_body) + x_prep

        # res = (
        #     torch.cat(
        #         (res, radiance, dfe_defocus_map),
        #         dim=1,
        #     )
        #     if self.positional_conv_last
        #     else res
        # )
        res = (
            torch.cat((res, radiance, apt_emb_l), dim=1)
            if self.positional_conv_last
            else res
        )
        x = x + self.conv_last(res)

        x = self.add_white_noise(x)

        for decoder, up, skip, enc_skip, use_pos_map_d, depth in zip(
            self.decoders,
            self.ups,
            self.skips,
            encs[::-1],
            self.dec_blks_use_pos_map,
            range(0, self.u_depth).__reversed__(),
        ):
            x = up(x)
            x = skip(x, enc_skip)
            pos_map_d = self._resize_by_depth(pos_map, depth, use_pos_map_d)
            x = self._run_blocks(decoder, x, pos_map_d)

        x = self.before_out_stage_mres(x)

        if self.out_stage_use_pos_map:
            x = torch.cat((x, pos_map), dim=1)
        x = self.out_stage(x)
        x = x / self.img_range + self.mean
        out = x + source
        return self._crop_forward_output(out, padding)

    def add_white_noise(self, x):
        if self.training and self.noise_level > 0:
            return x + torch.randn_like(x) * self.noise_level
        return x


BokehliciousDMA = Bokehlicious
