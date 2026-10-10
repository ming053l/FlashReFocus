"""Renderer pre-training on synthetic multi-focal Hypersim scenes (rendering targets from BokehMe), built on the
Bokehlicious training loop (train.py). See train_renderer.sh for the configuration of the released renderer."""
import os
import sys
import time

import torch
from tqdm import tqdm

from dataset import loader_hypersim_dma
from method import config_dma, model_dma_p


def _arg_value(name: str, default: str | None = None):
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


def _pop_arg_value(name: str, default: str | None = None):
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    value = default
    if idx + 1 < len(sys.argv):
        value = sys.argv[idx + 1]
        del sys.argv[idx : idx + 2]
    else:
        del sys.argv[idx]
    return value


crop_size = int(_arg_value("--crop-size", "0") or 0)
if crop_size <= 0 and "-h" not in sys.argv and "--help" not in sys.argv:
    raise ValueError(
        "train_hypersim_dma_p.py uses loader-side augmentation, so --crop-size "
        "must be > 0 to define the random crop size."
    )
os.environ["BOKEHLICIOUS_DMA_IMAGE_SIZE"] = str(crop_size)

DEPTH_MAP_DROPOUT_P = float(_pop_arg_value("--depth-map-dropout-p", "0.1") or 0.0)
if not 0.0 <= DEPTH_MAP_DROPOUT_P <= 1.0:
    raise ValueError("--depth-map-dropout-p must be in [0, 1].")
USE_K_IN_DEFOCUS = "--use-k-in-defocus" in sys.argv
if USE_K_IN_DEFOCUS:
    sys.argv.remove("--use-k-in-defocus")

if USE_K_IN_DEFOCUS:
    _BaseHypersimRealBokeh = loader_hypersim_dma.RealBokeh

    class RealBokehUseKInDefocus(_BaseHypersimRealBokeh):
        def __init__(self, *args, **kwargs):
            kwargs["use_k_in_defocus"] = True
            super().__init__(*args, **kwargs)

    loader_hypersim_dma.RealBokeh = RealBokehUseKInDefocus

sys.modules["dataset.loader"] = loader_hypersim_dma
sys.modules["method.config"] = config_dma
sys.modules["method.model"] = model_dma_p

import train  # noqa: E402


def _disable_train_script_crop(
    inputs,
    targets,
    pos_map,
    bokeh_strength_map,
    crop_size,
    depth_map=None,
):
    return inputs, targets, pos_map, bokeh_strength_map, depth_map


def _maybe_dropout_depth_map(depth_map: torch.Tensor | None, probability: float):
    if depth_map is None or probability <= 0.0:
        return depth_map
    batch_size = depth_map.size(0)
    if probability >= 1.0:
        drop_mask = torch.ones(batch_size, device=depth_map.device, dtype=torch.bool)
    else:
        drop_mask = torch.rand(batch_size, device=depth_map.device) < probability
    if not bool(drop_mask.any()):
        return depth_map
    drop_mask = drop_mask.view(batch_size, *([1] * (depth_map.dim() - 1)))
    return torch.where(drop_mask, torch.zeros_like(depth_map), depth_map)


def _defocus_laplacian_consistency_loss_p(
    model,
    inputs,
    bokeh_strength,
    focus_map,
    depth_map,
    disp_focus,
    focus_shift_max,
    focus_shift_min,
    strength_low_ratio,
    strength_margin,
):
    alt_focus, alt_defocus_map = train.sample_shifted_focus(
        depth_map=depth_map,
        disp_focus=disp_focus,
        max_shift=focus_shift_max,
        min_shift=focus_shift_min,
    )
    if alt_focus is None:
        zero = inputs.sum() * 0.0
        return zero, zero

    alt_outputs = model(
        source=inputs,
        bokeh_strength=bokeh_strength,
        focus_map=focus_map,
        depth_map=depth_map,
        disp_focus=alt_focus,
    )
    alt_outputs, _ = train.split_model_outputs(alt_outputs)

    source_lap = train.laplacian_magnitude(inputs.detach())
    alt_lap = train.laplacian_magnitude(alt_outputs.clamp(0.0, 1.0))
    blur_residual = (source_lap - alt_lap).relu()
    corr_target = alt_defocus_map * train.bokeh_strength_to_spatial_scale(
        bokeh_strength, alt_defocus_map
    )
    corr_loss = train.pearson_corr_loss(blur_residual, corr_target)

    strength_loss = inputs.sum() * 0.0
    if strength_low_ratio > 0.0:
        scale = torch.empty(
            inputs.size(0),
            device=inputs.device,
            dtype=inputs.dtype,
        ).uniform_(0.0, strength_low_ratio)
        low_strength = train.scale_bokeh_strength(bokeh_strength, scale)
        low_outputs = model(
            source=inputs,
            bokeh_strength=low_strength,
            focus_map=focus_map,
            depth_map=depth_map,
            disp_focus=alt_focus,
        )
        low_outputs, _ = train.split_model_outputs(low_outputs)
        low_lap = train.laplacian_magnitude(low_outputs.clamp(0.0, 1.0))
        low_residual = (source_lap - low_lap).relu()
        strength_delta = train.bokeh_strength_to_spatial_scale(
            bokeh_strength - low_strength, alt_defocus_map
        )
        margin = strength_margin * strength_delta.clamp_min(0.0) * alt_defocus_map
        strength_loss = (low_residual - blur_residual + margin).relu()
        strength_loss = (strength_loss * alt_defocus_map).mean()

    return corr_loss, strength_loss


def _train_one_epoch_p(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    criterion_weight,
    device,
    amp,
    log_interval,
    crop_size,
    zero_aperture_batch_p,
    defocus_corr_weight,
    defocus_strength_weight,
    focus_shift_max,
    focus_shift_min,
    strength_low_ratio,
    strength_margin,
):
    model.train()
    running_loss = 0.0
    start = time.time()
    step = 0

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), 1):
        inputs = batch["source"].to(device, non_blocking=True).float()
        targets = batch["target"].to(device, non_blocking=True).float()
        bokeh_strength = batch["bokeh_strength"].to(device, non_blocking=True).float()
        focus_map = batch["focus_map"].to(device, non_blocking=True).float()
        depth_map = batch["depth_map"].to(device, non_blocking=True).float()
        disp_focus = batch["disp_focus"].to(device, non_blocking=True).float()

        depth_map = _maybe_dropout_depth_map(depth_map, DEPTH_MAP_DROPOUT_P)
        targets, bokeh_strength = train.maybe_make_zero_aperture_batch(
            inputs=inputs,
            targets=targets,
            bokeh_strength=bokeh_strength,
            probability=zero_aperture_batch_p,
        )

        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0

        if amp:
            with train.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                outputs = model(
                    source=inputs,
                    bokeh_strength=bokeh_strength,
                    focus_map=focus_map,
                    depth_map=depth_map,
                    disp_focus=disp_focus,
                )
                primary_outputs, _ = train.split_model_outputs(outputs)
                loss = 0.0
                for crit, weight in zip(criterion, criterion_weight):
                    if isinstance(crit, train.LearnedPerceptualImagePatchSimilarity):
                        lpips_loss = crit(
                            primary_outputs.clamp(0.0, 1.0), targets.clamp(0.0, 1.0)
                        )
                        crit.reset()
                        loss = loss + weight * lpips_loss
                    else:
                        loss = loss + weight * crit(primary_outputs, targets)
                loss = train.add_prior_sobel_gradient_loss(loss, outputs, targets)
            loss_value += float(loss.detach())
            scaler.scale(loss).backward()
            del outputs, loss

            if defocus_corr_weight > 0.0 or defocus_strength_weight > 0.0:
                with train.autocast(
                    device_type="cuda", enabled=True, dtype=torch.bfloat16
                ):
                    corr_loss, strength_loss = _defocus_laplacian_consistency_loss_p(
                        model=model,
                        inputs=inputs,
                        bokeh_strength=bokeh_strength,
                        focus_map=focus_map,
                        depth_map=depth_map,
                        disp_focus=disp_focus,
                        focus_shift_max=focus_shift_max,
                        focus_shift_min=focus_shift_min,
                        strength_low_ratio=(
                            strength_low_ratio if defocus_strength_weight > 0.0 else 0.0
                        ),
                        strength_margin=strength_margin,
                    )
                    aux_loss = (
                        defocus_corr_weight * corr_loss
                        + defocus_strength_weight * strength_loss
                    )
                loss_value += float(aux_loss.detach())
                scaler.scale(aux_loss).backward()
                del corr_loss, strength_loss, aux_loss

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                source=inputs,
                bokeh_strength=bokeh_strength,
                focus_map=focus_map,
                depth_map=depth_map,
                disp_focus=disp_focus,
            )
            primary_outputs, _ = train.split_model_outputs(outputs)
            loss = 0.0
            for crit, weight in zip(criterion, criterion_weight):
                if isinstance(crit, train.LearnedPerceptualImagePatchSimilarity):
                    lpips_loss = crit(
                        primary_outputs.clamp(0.0, 1.0), targets.clamp(0.0, 1.0)
                    )
                    crit.reset()
                    loss = loss + weight * lpips_loss
                else:
                    loss = loss + weight * crit(primary_outputs, targets)
            loss = train.add_prior_sobel_gradient_loss(loss, outputs, targets)
            loss_value += float(loss.detach())
            loss.backward()
            del outputs, loss

            if defocus_corr_weight > 0.0 or defocus_strength_weight > 0.0:
                corr_loss, strength_loss = _defocus_laplacian_consistency_loss_p(
                    model=model,
                    inputs=inputs,
                    bokeh_strength=bokeh_strength,
                    focus_map=focus_map,
                    depth_map=depth_map,
                    disp_focus=disp_focus,
                    focus_shift_max=focus_shift_max,
                    focus_shift_min=focus_shift_min,
                    strength_low_ratio=(
                        strength_low_ratio if defocus_strength_weight > 0.0 else 0.0
                    ),
                    strength_margin=strength_margin,
                )
                aux_loss = (
                    defocus_corr_weight * corr_loss
                    + defocus_strength_weight * strength_loss
                )
                loss_value += float(aux_loss.detach())
                aux_loss.backward()
                del corr_loss, strength_loss, aux_loss

            optimizer.step()

        running_loss += loss_value

        if log_interval > 0 and step % log_interval == 0:
            avg = running_loss / step
            elapsed = time.time() - start
            print(f"train step {step}, loss {avg:.6f}, elapsed {elapsed:.1f}s")

    return running_loss / max(1, step)


@torch.inference_mode()
def _validate_p(model, loader, device, metric_funcs, amp=False):
    model.eval()
    for metric_fn in metric_funcs.values():
        if isinstance(metric_fn, torch.nn.Module):
            metric_fn.to(device)
            metric_fn.eval()
    tracker = train.MetricTracker(*metric_funcs.keys())
    tracker.reset()

    for batch in tqdm(loader, desc="val", leave=False):
        inputs = batch["source"].to(device, non_blocking=True).float()
        targets = batch["target"].to(device, non_blocking=True).float()
        bokeh_strength = batch["bokeh_strength"].to(device, non_blocking=True).float()
        focus_map = batch["focus_map"].to(device, non_blocking=True).float()
        depth_map = batch["depth_map"].to(device, non_blocking=True).float()
        disp_focus = batch["disp_focus"].to(device, non_blocking=True).float()

        with train.autocast(device_type="cuda", enabled=amp, dtype=torch.bfloat16):
            outputs = model(
                source=inputs,
                bokeh_strength=bokeh_strength,
                focus_map=focus_map,
                depth_map=depth_map,
                disp_focus=disp_focus,
            )
            outputs, _ = train.split_model_outputs(outputs)

        for metric_name, metric_fn in metric_funcs.items():
            if isinstance(metric_fn, train.LearnedPerceptualImagePatchSimilarity):
                metric_value = metric_fn(
                    outputs.clamp(0.0, 1.0), targets.clamp(0.0, 1.0)
                ).item()
            else:
                metric_value = metric_fn(outputs, targets).item()
            tracker.update(metric_name, metric_value, n=inputs.size(0))

    return tracker.result()


if __name__ == "__main__":
    train.random_crop_train_tensors = _disable_train_script_crop
    train.train_one_epoch = _train_one_epoch_p
    train.validate = _validate_p
    print(f"Using depth map dropout p={DEPTH_MAP_DROPOUT_P}")
    print(f"Using k in Hypersim defocus map: {USE_K_IN_DEFOCUS}")
    train.main()
