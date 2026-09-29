from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


def tokenize_prompt_texts(tokenizer, prompt_texts, device):
    return tokenizer(
        prompt_texts,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device, non_blocking=True)


def metric_is_better(metric_name, value, best_value):
    if best_value is None:
        return True
    return value > best_value if metric_name in {"psnr", "ssim"} else value < best_value


def get_conditioning_tensor_from_sample(sample, conditioning_source):
    if conditioning_source == "source":
        return sample["source"]
    if conditioning_source in {"defocus_map", "depth_map"}:
        return sample[conditioning_source].repeat(3, 1, 1)
    raise ValueError(f"Unsupported conditioning_source: {conditioning_source}")


def is_bokeh_rendering_task(task_mode):
    return str(task_mode).lower() in {"bokeh", "bokeh_render", "bokeh_rendering"}


def is_realbokeh_bokeh_task(args):
    dataset_name = str(getattr(args, "dataset_name", "") or "").lower()
    return dataset_name in {"realbokeh", "real_bokeh"} and is_bokeh_rendering_task(
        getattr(args, "task_mode", "")
    )


def use_ram_tag_prompts(args):
    return bool(getattr(args, "enable_ram_caption", False)) and not is_realbokeh_bokeh_task(
        args
    )


def bokeh_defocus_scale(args):
    explicit = getattr(args, "bokeh_defocus_scale", None)
    if explicit is not None:
        return float(explicit)
    dataset_name = str(getattr(args, "dataset_name", "") or "").lower()
    if dataset_name in {"realbokeh", "real_bokeh"}:
        return float(getattr(args, "realbokeh_defocus_scale", 1.0))
    if dataset_name in {"hypersim", "hypersim_bokeh"}:
        return float(getattr(args, "hypersim_defocus_scale", 1.0))
    return 1.0


def refine_bokeh_defocus_map(
    source_image,
    defocus_map,
    aperture_embedding,
    args,
):
    if defocus_map.ndim == 3:
        defocus_map = defocus_map.unsqueeze(1)
    return defocus_map


def scaled_bokeh_defocus_map(defocus_map, args):
    if not is_bokeh_rendering_task(getattr(args, "task_mode", "")):
        return defocus_map
    return defocus_map * bokeh_defocus_scale(args)


def maybe_scatter_bokeh_conditioning(control_image, defocus_map, args, allow_grad=False):
    if not is_bokeh_rendering_task(getattr(args, "task_mode", "")):
        return control_image
    if getattr(args, "conditioning_source", "source") != "source":
        return control_image
    if not control_image.is_cuda or not defocus_map.is_cuda:
        raise RuntimeError(
            "Bokeh rendering scatter preprocessing requires CUDA tensors because "
            "scatter.py only implements the CUDA path."
        )

    from scatter import ModuleRenderScatter

    renderer = getattr(maybe_scatter_bokeh_conditioning, "_renderer", None)
    if renderer is None:
        renderer = ModuleRenderScatter()
        maybe_scatter_bokeh_conditioning._renderer = renderer

    original_dtype = control_image.dtype
    renderer = renderer.to(device=control_image.device)
    grad_context = torch.enable_grad() if allow_grad else torch.no_grad()
    with grad_context, torch.autocast("cuda", enabled=False):
        bokeh, _ = renderer(control_image.float(), defocus_map.float())
    return bokeh.clamp(0.0, 1.0).to(dtype=original_dtype)


def format_aperture_prompt(aperture_value):
    return f'A bokeh image with aperture size of "1/{float(aperture_value)}"'


def strength_map(d_map, bias_scale=0.0, blur_kernel_size=3, blur_sigma=1, target_size=None):
    if target_size is not None and tuple(d_map.shape[-2:]) != tuple(target_size):
        d_map = F.interpolate(d_map, size=target_size, mode="bilinear", align_corners=False)
    d_map = gaussian_blur(d_map, blur_kernel_size, blur_sigma)
    d_map = ((d_map - d_map.amin()) / (d_map.amax() - d_map.amin() + 1e-8)).clamp(0, 1)
    return d_map + bias_scale


class PALLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, eps: float = 1e-3):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                "`prediction` and `target` must have the same shape for PAL loss; "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(0)
            target = target.unsqueeze(0)
        if prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError(
                "PAL loss expects image tensors shaped [N, 3, H, W] or [3, H, W]."
            )

        batch_size, channels, height, width = prediction.shape
        device_type = prediction.device.type
        autocast_context = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast_context:
            prediction_float = prediction.float()
            target_float = target.float()
            prediction_pixels = prediction_float.permute(0, 2, 3, 1).reshape(
                batch_size, -1, channels
            )
            target_pixels = target_float.permute(0, 2, 3, 1).reshape(
                batch_size, -1, channels
            )
            ones = torch.ones(
                (batch_size, prediction_pixels.shape[1], 1),
                device=prediction.device,
                dtype=torch.float32,
            )
            design_matrix = torch.cat([prediction_pixels, ones], dim=-1)
            xtx = torch.bmm(design_matrix.transpose(1, 2), design_matrix).float()
            xtt = torch.bmm(design_matrix.transpose(1, 2), target_pixels).float()
            identity = torch.eye(
                channels + 1,
                device=prediction.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            transform = torch.linalg.solve(xtx + self.eps * identity, xtt)
            transform = transform.transpose(1, 2).detach()

            prediction_flat = prediction_float.reshape(batch_size, channels, -1)
            prediction_aug = torch.cat(
                [
                    prediction_flat,
                    torch.ones(
                        (batch_size, 1, height * width),
                        device=prediction.device,
                        dtype=torch.float32,
                    ),
                ],
                dim=1,
            )
            aligned = torch.bmm(transform, prediction_aug).reshape(
                batch_size, channels, height, width
            )
            return self.alpha * F.l1_loss(aligned, target_float)
