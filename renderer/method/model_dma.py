import torch
import torch.nn.functional as F
from torch import Tensor, nn

from method.blocks_dma import ResidualBlockDMA, MResGroup
from method.model import Bokehlicious as _BaseBokehlicious


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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
            self.embed_dims[0] + 1, self.embed_dims[0], 3, 1, 1
        )
        self.gen_defocus_scale = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.SELU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.gen_defocus_scale[-2].weight)
        nn.init.zeros_(self.gen_defocus_scale[-2].bias)
        self._replace_dfe_blocks_with_dma()
        self.noise_level = 0.05

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
        aperture_map=None,
    ):
        x_size = (x.shape[2], x.shape[3])
        if defocus_map is None:
            raise ValueError("defocus_map is required for DMA forward_features.")
        if aperture_map is None and self.positional_dfe:
            aperture_map = self._make_aperture_embedding_map(x, bokeh_strength)
            aperture_map = torch.cat((aperture_map, 1 - aperture_map), dim=1)

        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(
                x,
                x_size,
                pos_map=aperture_map if self.positional_dfe else pos_map,
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

        x = (
            torch.cat((source_mean, pos_map), dim=1)
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
        for encoder, down, use_pos_map, depth in zip(
            self.encoders, self.downs, self.enc_blks_use_pos_map, range(0, self.u_depth)
        ):
            if pos_map is not None:
                pos_map_e = F.interpolate(
                    pos_map,
                    scale_factor=1 / 2**depth,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                pos_map_e = None
            for blk in encoder:
                x = blk(x, pos_map=pos_map_e)
            encs.append(x)
            x = down(x)

        x_noise = self.add_white_noise(x)
        x_prep = self.conv_prep(x_noise)

        dfe_defocus_map = self._resize_attention_defocus_map(
            x_prep,
            bokeh_strength_map,
            "bokeh_strength_map",
        )
        if depth_map is None:
            raise ValueError("depth_map is required for DMA defocus refinement.")
        depth_map = F.interpolate(
            depth_map.to(device=x_prep.device, dtype=x_prep.dtype),
            size=x_prep.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if disp_focus is not None:
            focal_d = self._make_focal_distance(x_prep, disp_focus)
        else:
            focal_d = self.estimate_d_from_abs_map_batch(depth_map, dfe_defocus_map)
            focal_d = focal_d.view(-1, 1, 1, 1)
        defocus_scale = self.gen_defocus_scale(depth_map - focal_d)
        dfe_defocus_map = (1 + defocus_scale) * dfe_defocus_map
        dfe_defocus_map = dfe_defocus_map.clamp(0.0, 1.0)

        x_prep = self.defocus_fuse(torch.cat((x_prep, dfe_defocus_map), dim=1))

        aperture_map_t = (
            self._make_aperture_embedding_map(x_prep, bokeh_strength)
            if self.positional_dfe or self.positional_conv_last
            else None
        )
        aperture_map_t = (
            torch.cat((aperture_map_t, 1 - aperture_map_t), dim=1)
            if self.positional_dfe
            else aperture_map_t
        )

        x_after_body = self.forward_features(
            x_prep,
            bokeh_strength=bokeh_strength,
            defocus_map=dfe_defocus_map,
            aperture_map=aperture_map_t,
        )
        res = self.conv_after_body(x_after_body) + x_prep

        res = (
            torch.cat(
                (
                    res,
                    F.interpolate(
                        pos_map,
                        scale_factor=1 / 2**self.u_depth,
                        mode="bilinear",
                        align_corners=False,
                    ),
                ),
                dim=1,
            )
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
            pos_map_d = (
                F.interpolate(
                    pos_map,
                    scale_factor=1 / 2**depth,
                    mode="bilinear",
                    align_corners=False,
                )
                if use_pos_map_d
                else None
            )
            for blk in decoder:
                x = blk(x, pos_map=pos_map_d)

        x = self.before_out_stage_mres(x)
        if self.out_stage_use_pos_map:
            out_aperture_map = self._make_aperture_embedding_map(x, bokeh_strength)
            out_defocus_map = self._resize_attention_defocus_map(
                x,
                dfe_defocus_map,
                "dfe_defocus_map",
            )
            x = torch.cat((x, out_aperture_map, out_defocus_map), dim=1)
        x = self.out_stage(x)
        x = x / self.img_range + self.mean
        return self._crop_forward_output(x + source, padding)

    def add_white_noise(self, x):
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
        return x


BokehliciousDMA = Bokehlicious
