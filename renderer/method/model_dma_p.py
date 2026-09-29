import torch
import torch.nn.functional as F
from torch import Tensor, nn

from method.blocks import BlockMod
from method.blocks_dma import ResidualBlockDMA
from method.model import Bokehlicious as _BaseBokehlicious


class RadianceVirtualization(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=32,
        alpha=3.0,
        beta=5.0,
        bright_threshold=0.95,
        eps=1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bright_threshold = bright_threshold
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

        R0 = F.softplus(self.net(x)) + self.eps

        max_rgb = x.amax(dim=1, keepdim=True)
        bright_mask = (max_rgb >= self.bright_threshold).float()

        R_bright = self.alpha * torch.pow(max_rgb + self.eps, self.beta)

        R = bright_mask * R_bright + (1.0 - bright_mask) * R0
        return R


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
        use_radiance_visualization: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            out_stage_use_pos_map=out_stage_use_pos_map,
            **kwargs,
        )
        self.use_radiance_visualization = use_radiance_visualization
        self.attn_version = attn_version
        self.conditioning_map_channels = 3
        v3_in_channels = self.in_chans
        v3_in_channels += (
            self.conditioning_map_channels if self.in_stage_use_pos_map else 0
        )
        v3_in_channels += 1 if self.in_stage_use_bokeh_strength_map else 0
        if self.in_stage.in_channels != v3_in_channels:
            self.in_stage = nn.Conv2d(
                in_channels=v3_in_channels,
                out_channels=self.u_width,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=1,
                bias=True,
            )
        self.in_stage_2 = nn.Conv2d(
            in_channels=self.u_width,
            out_channels=self.u_width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self._replace_unet_pos_blocks_with_conditioning_blocks()
        self.after_in_stage2_mres = nn.Identity()
        # self.before_out_stage_mres = (
        #     MResNet(3, self.u_width, 1, mres_num_blocks, True)
        #     if use_mres_before_out_stage
        #     else nn.Identity()
        # )
        self.before_out_stage_mres = (
            nn.Sequential(
                nn.Conv2d(4, self.u_width, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.u_width, self.u_width, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.u_width, 3, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            if use_mres_before_out_stage
            else nn.Identity()
        )
        self.use_mres_before_out_stage = use_mres_before_out_stage
        self.defocus_fuse = nn.Conv2d(
            (
                (self.embed_dims[0] + 2)
                if self.use_radiance_visualization
                else (self.embed_dims[0] + 3)
            ),
            self.embed_dims[0],
            3,
            1,
            1,
        )
        # self.gen_defocus_scale = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 1, 1),
        #     nn.SELU(),
        #     nn.Conv2d(16, 2, 3, 1, 1),
        #     nn.Tanh(),
        # )
        # nn.init.zeros_(self.gen_defocus_scale[-2].weight)
        # nn.init.zeros_(self.gen_defocus_scale[-2].bias)
        if self.use_radiance_visualization:
            self.rad_viz = RadianceVirtualization()

        conv_last_extra_channels = (
            self.conditioning_map_channels if self.positional_conv_last else 0
        )
        self.conv_last = nn.Conv2d(
            self.embed_dims[0] + conv_last_extra_channels,
            self.u_width * 2**self.u_depth,
            3,
            1,
            1,
        )

        extra_out_channels = (
            self.conditioning_map_channels if self.out_stage_use_pos_map else 0
        )
        print(f"Extra out channels: {extra_out_channels}")
        self.out_stage = nn.Conv2d(
            in_channels=self.u_width + extra_out_channels,
            out_channels=self.in_chans,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self._replace_dfe_blocks_with_dma()
        self.noise_level = 0.05

    def _replace_unet_pos_blocks_with_conditioning_blocks(self):
        chan = self.u_width
        for i, (num, depth, use_pos_map) in enumerate(
            zip(self.enc_blk_nums, range(0, self.u_depth), self.enc_blks_use_pos_map)
        ):
            self.encoders[i] = nn.ModuleList(
                [
                    BlockMod(
                        chan,
                        **self.u_block_config,
                        depth=depth,
                        use_pos_map=use_pos_map,
                        pos_map_channels=self.conditioning_map_channels,
                    )
                    for _ in range(num)
                ]
            )
            chan *= 2

        for i, (num, use_pos_map, depth) in enumerate(
            zip(
                self.dec_blk_nums,
                self.dec_blks_use_pos_map,
                range(0, self.u_depth).__reversed__(),
            )
        ):
            chan //= 2
            self.decoders[i] = nn.ModuleList(
                [
                    BlockMod(
                        chan,
                        **self.u_block_config,
                        use_pos_map=use_pos_map,
                        pos_map_channels=self.conditioning_map_channels,
                        depth=depth,
                    )
                    for _ in range(num)
                ]
            )

        self.encoders.apply(self._init_weights)
        self.decoders.apply(self._init_weights)

    def _replace_dfe_blocks_with_dma(self):
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))
        ]

        self.blocks = nn.ModuleList()
        for i_block in range(self.num_blocks):
            self.blocks.append(
                ResidualBlockDMA(
                    embed_dim=self.embed_dims[i_block],
                    embed_dim_next=(
                        self.embed_dims[i_block + 1]
                        if i_block < self.num_blocks - 1
                        else self.embed_dims[i_block]
                    ),
                    depth=self.depths[i_block],
                    num_heads=self.num_heads[i_block],
                    init_value=self.init_values[i_block],
                    heads_range=self.heads_ranges[i_block],
                    ffn_dim=int(self.mlp_ratios[i_block] * self.embed_dims[i_block]),
                    drop_path=dpr[
                        sum(self.depths[:i_block]) : sum(self.depths[: i_block + 1])
                    ],
                    norm_layer=self.norm_layer,
                    use_checkpoint=self.use_checkpoints[i_block],
                    layerscale=self.layerscales[i_block],
                    layer_init_values=self.layer_init_values,
                    use_pos_map=self.positional_dfe,
                    pos_map_channels=self.conditioning_map_channels,
                    attn_version=self.attn_version,
                )
            )

        self.blocks.apply(self._init_weights)

    @staticmethod
    def _resize_attention_defocus_map(
        x: Tensor,
        defocus_map: Tensor,
        name: str,
    ) -> Tensor:
        if defocus_map is None:
            raise ValueError(f"{name} is required.")
        if defocus_map.dim() != 4 or defocus_map.size(1) != 1:
            raise ValueError(
                f"{name} must have shape [B, 1, H, W], got "
                f"{tuple(defocus_map.shape)}"
            )
        if defocus_map.size(0) != x.size(0):
            raise ValueError(
                f"{name} batch size does not match feature batch size: "
                f"{defocus_map.size(0)} vs {x.size(0)}"
            )
        return F.interpolate(
            defocus_map.to(device=x.device, dtype=x.dtype),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _make_aperture_embedding_map(
        x: Tensor,
        bokeh_strength: Tensor,
    ) -> Tensor:
        if bokeh_strength is None:
            raise ValueError("bokeh_strength is required.")
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
        if bokeh_strength.size(0) != x.size(0):
            raise ValueError(
                "bokeh_strength batch size does not match feature batch size: "
                f"{bokeh_strength.size(0)} vs {x.size(0)}"
            )
        aperture = bokeh_strength.to(device=x.device, dtype=x.dtype).view(
            x.size(0), 1, 1, 1
        )

        aperture = aperture.expand(-1, -1, x.shape[-2], x.shape[-1])
        return aperture

    @classmethod
    def _make_conditioning_map(
        cls,
        x: Tensor,
        defocus_map: Tensor,
        bokeh_strength: Tensor,
    ) -> Tensor:
        foreground_mask = (defocus_map < 1e-3).to(dtype=defocus_map.dtype)
        defocus_map = cls._resize_attention_defocus_map(
            x,
            defocus_map,
            "bokeh_strength_map",
        )
        foreground_mask = F.interpolate(
            foreground_mask.to(device=x.device, dtype=x.dtype),
            size=x.shape[-2:],
            mode="nearest",
        )
        foreground_mask = F.max_pool2d(
            foreground_mask,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        aperture_map = cls._make_aperture_embedding_map(x, bokeh_strength)
        return torch.cat((defocus_map, foreground_mask, aperture_map), dim=1)

    @staticmethod
    def estimate_d_from_abs_map_batch(map1, map2, tol=1e-6):
        B = map1.shape[0]
        cand_plus = (map1 + map2).flatten(1)  # (B, H*W)
        cand_minus = (map1 - map2).flatten(1)  # (B, H*W)
        candidates = torch.cat([cand_plus, cand_minus], dim=1)  # (B, 2*H*W)
        q = torch.round(candidates / tol).long()  # (B, 2*H*W)
        mode_q = torch.mode(q, dim=1).values  # (B,)
        mask = q == mode_q[:, None]  # (B, 2*H*W)
        d_est = (candidates * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        return d_est.view(B, 1)

    @staticmethod
    def _make_focal_distance(
        x: Tensor,
        disp_focus: Tensor,
    ) -> Tensor:
        if disp_focus is None:
            raise ValueError("disp_focus is required.")
        if not torch.is_tensor(disp_focus):
            disp_focus = torch.tensor(disp_focus)
        disp_focus = disp_focus.to(device=x.device, dtype=x.dtype)
        if disp_focus.dim() == 0:
            disp_focus = disp_focus.view(1)
        if disp_focus.dim() == 1:
            disp_focus = disp_focus[:, None]
        elif disp_focus.dim() == 4:
            disp_focus = disp_focus.mean(dim=(2, 3))

        if disp_focus.dim() != 2 or disp_focus.size(1) != 1:
            raise ValueError(
                "disp_focus must have shape [B], [B, 1], or [B, 1, H, W], got "
                f"{tuple(disp_focus.shape)}"
            )
        if disp_focus.size(0) != x.size(0):
            raise ValueError(
                "disp_focus batch size does not match feature batch size: "
                f"{disp_focus.size(0)} vs {x.size(0)}"
            )
        return disp_focus.view(x.size(0), 1, 1, 1)

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
        focus_map: Tensor = None,
        depth_map: Tensor = None,
        disp_focus: Tensor = None,
        **kwargs,
    ):
        kwargs.pop("pos_map", None)
        kwargs.pop("bokeh_strength_map", None)
        source, _, _, depth_map, padding = self._pad_forward_inputs(
            source=source,
            pos_map=None,
            bokeh_strength_map=None,
            depth_map=depth_map,
        )

        self.mean = self.mean.type_as(source)
        source_mean = (source - self.mean) * self.img_range
        # source_mean = source
        if depth_map is None:
            raise ValueError("depth_map is required to build bokeh_strength_map.")
        if disp_focus is None:
            raise ValueError("disp_focus is required to build bokeh_strength_map.")

        depth_map = depth_map.to(device=source.device, dtype=source.dtype)
        focal_d = self._make_focal_distance(source, disp_focus)
        signed_defocus = depth_map - focal_d
        ##
        # a, b = self.gen_defocus_scale(source_mean).chunk(2, dim=1)
        # signed_defocus = a * signed_defocus + b
        ##
        bokeh_strength_map = signed_defocus.abs()
        conditioning_map = self._make_conditioning_map(
            source_mean,
            signed_defocus,
            bokeh_strength,
        )
        # focus_tolerance = 0.02
        # focus_softness = 0.015
        # soft_focus_mask = torch.sigmoid(
        #     (focus_tolerance - bokeh_strength_map) / focus_softness
        # )

        x = (
            torch.cat((source_mean, conditioning_map), dim=1)
            if self.in_stage_use_pos_map
            else source_mean
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
        for encoder, down, use_pos_map, depth in zip(
            self.encoders, self.downs, self.enc_blks_use_pos_map, range(0, self.u_depth)
        ):
            if use_pos_map:
                conditioning_map_e = F.interpolate(
                    conditioning_map,
                    scale_factor=1 / 2**depth,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                conditioning_map_e = None
            for blk in encoder:
                x = blk(x, pos_map=conditioning_map_e)
            encs.append(x)
            x = down(x)

        x_noise = self.add_white_noise(x)
        x_prep = self.conv_prep(x_noise)

        dfe_defocus_map = self._resize_attention_defocus_map(
            x_prep,
            bokeh_strength_map,
            "bokeh_strength_map",
        )
        dfe_defocus_map = dfe_defocus_map.clamp(0.0, 1.0)

        conditioning_map_t = (
            F.interpolate(
                conditioning_map,
                scale_factor=1 / 2**self.u_depth,
                mode="bilinear",
                align_corners=False,
            )
            if self.positional_dfe or self.positional_conv_last
            else None
        )
        bokeh_strength_e = self._make_aperture_embedding_map(x_prep, bokeh_strength)
        if self.use_radiance_visualization:
            radiance = self.rad_viz(
                F.interpolate(
                    source,
                    scale_factor=1 / 2**self.u_depth,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            x_prep = self.defocus_fuse(
                torch.cat((x_prep, radiance, bokeh_strength_e), dim=1)
            )
        else:
            x_prep = self.defocus_fuse(torch.cat((x_prep, conditioning_map_t), dim=1))

        x_after_body = self.forward_features(
            x_prep,
            bokeh_strength=bokeh_strength,
            defocus_map=dfe_defocus_map,
            pos_map=conditioning_map_t,
        )
        res = self.conv_after_body(x_after_body) + x_prep

        res = (
            torch.cat((res, conditioning_map_t), dim=1)
            if self.positional_conv_last
            else res
        )
        x = x + self.conv_last(res)

        x = self.add_white_noise(x)

        # conditioning_map_d = conditioning_map_t  #############
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
            conditioning_map_d = (
                F.interpolate(
                    conditioning_map,
                    scale_factor=1 / 2**depth,
                    mode="bilinear",
                    align_corners=False,
                )
                if use_pos_map_d
                else None
            )
            # conditioning_map_d = up_map(conditioning_map_d) if use_pos_map_d else None
            for blk in decoder:
                x = blk(x, pos_map=conditioning_map_d)

        if self.out_stage_use_pos_map:
            x = torch.cat((x, conditioning_map), dim=1)
        x = self.out_stage(x)
        if self.use_mres_before_out_stage:
            x = self.before_out_stage_mres(torch.cat((x, signed_defocus), dim=1)) + x
        x = x / self.img_range + self.mean
        output = x + source
        # output = output * (1 - soft_focus_mask) + source * soft_focus_mask
        output = self._crop_forward_output(output, padding)
        return output

    def add_white_noise(self, x):
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
        return x


BokehliciousDMA = Bokehlicious
