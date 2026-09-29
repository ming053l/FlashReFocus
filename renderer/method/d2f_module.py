import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthLayerKernelFusion(nn.Module):
    """
    Layered Fusion from D2F.

    The module implements the paper's fixed layered rendering core fl(I, D):
    D builds soft defocus-layer masks, each layer is rendered by a pre-defined
    soft disk kernel, and the rendered layers are composited in layer order.
    """

    def __init__(
        self,
        num_layers=15,
        gamma=100.0,
        sigma=0.25,
        phi=0.5,
        eps=1e-6,
        clamp_output=True,
        normalize_defocus=True,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2.")

        self.num_layers = int(num_layers)
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.phi = float(phi)
        self.eps = float(eps)
        self.clamp_output = bool(clamp_output)
        self.normalize_defocus = bool(normalize_defocus)
        self.bokeh_scale = nn.Parameter(torch.tensor(0.54132485), requires_grad=True)

        self.kernel_sizes = self._make_kernel_sizes(self.num_layers)
        self.kernels = nn.ParameterList(
            [
                nn.Parameter(
                    self._make_layer_kernel(layer_idx, kernel_size),
                    requires_grad=False,
                )
                for layer_idx, kernel_size in enumerate(self.kernel_sizes)
            ]
        )

    @staticmethod
    def _make_kernel_sizes(num_layers):
        sizes = []
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                size = 1
            elif layer_idx < 3:
                size = 2 * layer_idx + 1
            elif layer_idx < 8:
                size = 4 * (layer_idx - 3) + 7
            elif layer_idx < 11:
                size = 6 * (layer_idx - 8) + 27
            else:
                size = 8 * (layer_idx - 11) + 45
            sizes.append(size) 
        return sizes

    def _make_layer_kernel(self, layer_idx, kernel_size):
        if layer_idx == 0:
            return torch.ones(1, 1, 1, 1)

        center = kernel_size // 2
        radius = float(center)
        coords = torch.arange(kernel_size, dtype=torch.float32) - float(center)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        kernel = 0.5 + 0.5 * torch.tanh(
            self.sigma * (radius**2 - xx**2 - yy**2) + self.phi
        )
        kernel = kernel / kernel.sum().clamp_min(self.eps)
        return kernel.view(1, 1, kernel_size, kernel_size)

    def _prepare_single_channel(self, value, name, image_size):
        if value is None:
            raise ValueError(f"{name} is required.")
        if value.dim() == 3:
            value = value.unsqueeze(1)
        if value.dim() != 4:
            raise ValueError(f"{name} must have shape Bx1xHxW or BxHxW.")
        if value.size(1) != 1:
            value = value.mean(dim=1, keepdim=True)
        if value.shape[-2:] != image_size:
            value = F.interpolate(value, size=image_size, mode="bilinear", align_corners=False)
        return value

    def _normalize_defocus_map(self, defocus):
        if not self.normalize_defocus:
            return defocus.clamp(0.0, 1.0)

        max_value = defocus.amax(dim=(2, 3), keepdim=True)
        needs_scale = max_value > 1.0
        scaled = defocus / max_value.clamp_min(self.eps)
        defocus = torch.where(needs_scale, scaled, defocus)
        return defocus.clamp(0.0, 1.0)

    def _prepare_bokeh_strength(self, bokeh_strength, image):
        if bokeh_strength is None:
            return torch.ones(
                image.size(0),
                1,
                1,
                1,
                device=image.device,
                dtype=image.dtype,
            )
        if not torch.is_tensor(bokeh_strength):
            bokeh_strength = torch.tensor(bokeh_strength)
        bokeh_strength = bokeh_strength.to(device=image.device, dtype=image.dtype)
        if bokeh_strength.dim() == 0:
            bokeh_strength = bokeh_strength.view(1, 1, 1, 1)
        elif bokeh_strength.dim() == 1:
            bokeh_strength = bokeh_strength.view(-1, 1, 1, 1)
        elif bokeh_strength.dim() == 2:
            bokeh_strength = bokeh_strength.view(bokeh_strength.size(0), -1, 1, 1)
        elif bokeh_strength.dim() == 4:
            bokeh_strength = bokeh_strength.mean(dim=(2, 3), keepdim=True)
        else:
            raise ValueError(
                "bokeh_strength must have shape [], [B], [B, 1], or [B, 1, H, W]."
            )
        if bokeh_strength.size(1) != 1:
            bokeh_strength = bokeh_strength.mean(dim=1, keepdim=True)
        if bokeh_strength.size(0) == 1 and image.size(0) > 1:
            bokeh_strength = bokeh_strength.expand(image.size(0), -1, -1, -1)
        if bokeh_strength.size(0) != image.size(0):
            raise ValueError(
                "bokeh_strength batch size does not match image batch size: "
                f"{bokeh_strength.size(0)} vs {image.size(0)}"
            )
        return bokeh_strength

    def _make_layer_masks(self, defocus):
        centers = torch.arange(
            self.num_layers,
            device=defocus.device,
            dtype=defocus.dtype,
        ).view(1, self.num_layers, 1, 1)
        centers = centers / float(max(self.num_layers - 1, 1))

        masks = 0.5 + 0.5 * torch.tanh(
            self.gamma * ((1.0 / self.num_layers) - (defocus - centers).abs())
        )
        return masks.unsqueeze(2)

    def _blur(self, value, layer_idx):
        kernel = self.kernels[layer_idx].to(device=value.device, dtype=value.dtype)
        if kernel.size(-1) == 1:
            return value

        batch, channels, height, width = value.shape
        pad = kernel.size(-1) // 2
        pad_mode = "reflect" if min(height, width) > pad else "replicate"
        padded = F.pad(value, (pad, pad, pad, pad), mode=pad_mode)
        weight = kernel.repeat(channels, 1, 1, 1)
        return F.conv2d(padded, weight, groups=channels)

    def _transmittance_after_each_layer(self, masks):
        layer_masks = masks.squeeze(2)
        one_minus = 1.0 - layer_masks
        suffix = torch.ones_like(one_minus)

        running = torch.ones_like(one_minus[:, 0:1])
        for layer_idx in range(self.num_layers - 1, -1, -1):
            suffix[:, layer_idx : layer_idx + 1] = running
            running = running * one_minus[:, layer_idx : layer_idx + 1]

        return suffix.unsqueeze(2)

    def layered_render(self, value, defocus):
        masks = self._make_layer_masks(defocus)
        transmittance = self._transmittance_after_each_layer(masks)

        numerator = torch.zeros_like(value)
        denominator = torch.zeros(
            value.size(0),
            1,
            value.size(2),
            value.size(3),
            device=value.device,
            dtype=value.dtype,
        )

        for layer_idx in range(self.num_layers):
            mask = masks[:, layer_idx]
            transmission = transmittance[:, layer_idx]
            rendered_layer = self._blur(value * mask, layer_idx)
            rendered_mask = self._blur(mask, layer_idx)

            numerator = numerator + rendered_layer * transmission
            denominator = denominator + rendered_mask * transmission

        return numerator / denominator.clamp_min(self.eps)

    def forward(
        self,
        image,
        defocus,
        bokeh_strength=None,
        return_intermediates=False,
    ):
        if image.dim() != 4:
            raise ValueError("image must have shape BxCxHxW.")

        image = image.clamp(0.0, 1.0)
        defocus = self._prepare_single_channel(defocus, "defocus", image.shape[-2:])
        defocus = self._normalize_defocus_map(defocus.to(dtype=image.dtype, device=image.device))
        bokeh_strength = self._prepare_bokeh_strength(bokeh_strength, image)
        defocus = (
            defocus
            * bokeh_strength
            * F.softplus(self.bokeh_scale.to(device=image.device, dtype=image.dtype))
        ).clamp(0.0, 1.0)

        output = self.layered_render(image, defocus)

        if self.clamp_output:
            output = output.clamp(0.0, 1.0)

        if return_intermediates:
            return {
                "output": output,
                "defocus": defocus,
                "bokeh_strength": bokeh_strength,
            }

        return output
