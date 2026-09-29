import argparse
import inspect
import os
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from tqdm import tqdm

from dataset.loader import RealBokeh
from dataset.util import Mode
from method.config import bokehlicious_size_builder
from method.model import Bokehlicious


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_model_outputs(outputs):
    if isinstance(outputs, (tuple, list)):
        primary = outputs[0]
        prior = outputs[1] if len(outputs) > 1 else None
        return primary, prior
    return outputs, None


def sobel_gradients(images):
    kernels = images.new_tensor(
        [
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        ]
    )
    channels = images.size(1)
    kernels = kernels.view(2, 1, 3, 3).repeat(channels, 1, 1, 1)
    padded = F.pad(images.float(), (1, 1, 1, 1), mode="replicate")
    gradients = F.conv2d(padded, kernels, groups=channels)
    return gradients.view(images.size(0), channels, 2, images.size(2), images.size(3))


def add_prior_sobel_gradient_loss(loss, outputs, targets, weight=0.1):
    _, prior = split_model_outputs(outputs)
    if prior is None:
        return loss
    prior_grad = sobel_gradients(prior)
    target_grad = sobel_gradients(targets)
    return loss + weight * F.l1_loss(prior_grad, target_grad)


class MetricTracker:
    def __init__(self, *keys):
        self._keys = keys
        self.reset()

    def reset(self):
        self._total = {k: 0.0 for k in self._keys}
        self._counts = {k: 0 for k in self._keys}

    def update(self, key, value, n=1):
        self._total[key] += float(value) * n
        self._counts[key] += n

    def result(self):
        return {
            k: (self._total[k] / self._counts[k] if self._counts[k] > 0 else 0.0)
            for k in self._keys
        }


def append_metrics_txt(path, epoch, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if is_new:
            f.write("epoch," + ",".join(metrics.keys()) + "\n")
        values = ",".join(f"{metrics[k]:.6f}" for k in metrics.keys())
        f.write(f"{epoch},{values}\n")


def save_checkpoint(path, model, optimizer, scaler, epoch, best_val):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_val": best_val,
    }
    torch.save(state, path)


def extract_model_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must be a dict or a model state_dict.")

    if "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]

    return state


def load_pretrained_weights(model, pretrained_path, device, strict=False):
    checkpoint = torch.load(pretrained_path, map_location=device)
    state_dict = extract_model_state_dict(checkpoint)

    # Strip common DistributedDataParallel prefixes if present.
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }

    if not strict:
        model_state = model.state_dict()
        mismatched = []
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in model_state and value.shape != model_state[key].shape:
                mismatched.append(
                    (key, tuple(value.shape), tuple(model_state[key].shape))
                )
                continue
            filtered_state_dict[key] = value
        state_dict = filtered_state_dict
        if mismatched:
            print("Skipped shape-mismatched pretrained keys:")
            for key, loaded_shape, model_shape in mismatched:
                print(f"  {key}: checkpoint{loaded_shape} -> model{model_shape}")

    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])

    print(
        f"Loaded pretrained weights from {pretrained_path} "
        f"(strict={strict}, missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        print("Missing keys:", ", ".join(missing))
    if unexpected:
        print("Unexpected keys:", ", ".join(unexpected))


def freeze_except_prefixes(model, trainable_prefixes):
    prefixes = tuple(prefix for prefix in trainable_prefixes if prefix)
    if not prefixes:
        raise ValueError("At least one trainable parameter prefix is required.")

    def matches_prefix_or_substring(name):
        return any(name.startswith(prefix) or prefix in name for prefix in prefixes)

    trainable = 0
    frozen = 0
    trainable_names = []
    for name, param in model.named_parameters():
        is_trainable = matches_prefix_or_substring(name)
        param.requires_grad = is_trainable
        if is_trainable:
            trainable += param.numel()
            trainable_names.append(name)
        else:
            frozen += param.numel()

    if trainable == 0:
        raise ValueError(
            "No trainable parameters matched prefixes: " + ", ".join(trainable_prefixes)
        )

    print(
        "Frozen model parameters except prefixes "
        f"{', '.join(prefixes)}: trainable={trainable:,}, frozen={frozen:,}"
    )
    print("Trainable parameter tensors:", ", ".join(trainable_names))


def freeze_rel_pos_attention_blocks(model):
    frozen = 0
    frozen_names = []
    for module_name, module in model.named_modules():
        attention = getattr(module, "attention", None)
        if attention is None:
            continue
        if attention.__class__.__name__ != "DMARelPosAttention":
            continue
        for param in module.parameters():
            if param.requires_grad:
                frozen += param.numel()
            param.requires_grad = False
        frozen_names.append(module_name)

    if not frozen_names:
        raise ValueError("No DMARelPosAttention blocks found to freeze.")

    print(
        "Frozen rel-pos attention blocks: "
        f"{len(frozen_names)} blocks, {frozen:,} parameters"
    )
    print("Frozen rel-pos block modules:", ", ".join(frozen_names))


def build_optimizer(model, args, depth_defocus_prefixes):
    depth_defocus_lr = args.depth_defocus_lr
    backbone_lr = args.backbone_lr
    if depth_defocus_lr is None and backbone_lr is None:
        params = [param for param in model.parameters() if param.requires_grad]
        return torch.optim.Adam(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    depth_defocus_lr = args.lr if depth_defocus_lr is None else depth_defocus_lr
    backbone_lr = args.lr if backbone_lr is None else backbone_lr
    prefixes = tuple(depth_defocus_prefixes)

    def matches_prefix_or_substring(name):
        return any(name.startswith(prefix) or prefix in name for prefix in prefixes)

    depth_defocus_params = []
    backbone_params = []
    depth_defocus_count = 0
    backbone_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if matches_prefix_or_substring(name):
            depth_defocus_params.append(param)
            depth_defocus_count += param.numel()
        else:
            backbone_params.append(param)
            backbone_count += param.numel()

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "name": "backbone",
            }
        )
    if depth_defocus_params:
        param_groups.append(
            {
                "params": depth_defocus_params,
                "lr": depth_defocus_lr,
                "name": "depth_defocus",
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer.")

    print(
        "Using differential learning rates: "
        f"backbone_lr={backbone_lr:g} ({backbone_count:,} params), "
        f"depth_defocus_lr={depth_defocus_lr:g} ({depth_defocus_count:,} params)"
    )
    return torch.optim.Adam(
        param_groups,
        weight_decay=args.weight_decay,
    )


def build_model(model_cls, config, args):
    model_kwargs = dict(config)
    model_params = inspect.signature(model_cls).parameters
    if "use_mres_after_in_stage2" in model_params:
        model_kwargs["use_mres_after_in_stage2"] = args.use_mres_after_in_stage2
    if "use_mres_before_out_stage" in model_params:
        model_kwargs["use_mres_before_out_stage"] = args.use_mres_before_out_stage
    if "mres_num_blocks" in model_params:
        model_kwargs["mres_num_blocks"] = args.mres_num_blocks
    if "attn_version" in model_params:
        model_kwargs["attn_version"] = args.attn_version
    if "use_occ_refine" in model_params:
        model_kwargs["use_occ_refine"] = getattr(args, "use_occ_refine", False)
    if "use_prior" in model_params:
        model_kwargs["use_prior"] = getattr(args, "use_prior", False)
    if "use_radiance_visualization" in model_params:
        model_kwargs["use_radiance_visualization"] = not getattr(
            args, "disable_radiance_visualization", False
        )
    if "out_stage_use_pos_map" in model_params:
        model_kwargs["out_stage_use_pos_map"] = not getattr(
            args, "not_out_stage_pos_map", False
        )
    if "bokeh_embedding_type" in model_params:
        model_kwargs["bokeh_embedding_type"] = getattr(
            args,
            "bokeh_embedding_type",
            "k",
        )
    return model_cls(**model_kwargs)


def build_metric_funcs(metric_names, lpips_net):
    funcs = {}
    for name in metric_names:
        if name == "psnr":
            funcs[name] = lambda out, tgt: psnr(out, tgt, data_range=1.0)
        elif name == "ssim":
            funcs[name] = lambda out, tgt: ssim(out, tgt, data_range=1.0)
        elif name == "lpips":
            lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type=lpips_net, normalize=True
            )
            funcs[name] = lpips_metric
        else:
            raise ValueError(f'Unsupported metric "{name}". Use psnr, ssim, lpips.')
    return funcs


def random_crop_train_tensors(
    inputs,
    targets,
    pos_map,
    bokeh_strength_map,
    crop_size,
    depth_map=None,
):
    if crop_size is None or crop_size <= 0:
        return inputs, targets, pos_map, bokeh_strength_map, depth_map

    _, _, h, w = inputs.shape
    size = min(crop_size, h, w)
    size = (size // 4) * 4
    if size < 4:
        return inputs, targets, pos_map, bokeh_strength_map, depth_map
    if size == h and size == w:
        return inputs, targets, pos_map, bokeh_strength_map, depth_map

    cropped_inputs = []
    cropped_targets = []
    cropped_pos_maps = []
    cropped_strength_maps = []
    cropped_depth_maps = [] if depth_map is not None else None

    for i in range(inputs.shape[0]):
        top = torch.randint(0, h - size + 1, (1,), device=inputs.device).item()
        left = torch.randint(0, w - size + 1, (1,), device=inputs.device).item()
        cropped_inputs.append(inputs[i, :, top : top + size, left : left + size])
        cropped_targets.append(targets[i, :, top : top + size, left : left + size])
        cropped_pos_maps.append(pos_map[i, :, top : top + size, left : left + size])
        cropped_strength_maps.append(
            bokeh_strength_map[i, :, top : top + size, left : left + size]
        )
        if cropped_depth_maps is not None:
            cropped_depth_maps.append(
                depth_map[i, :, top : top + size, left : left + size]
            )

    return (
        torch.stack(cropped_inputs, dim=0),
        torch.stack(cropped_targets, dim=0),
        torch.stack(cropped_pos_maps, dim=0),
        torch.stack(cropped_strength_maps, dim=0),
        (
            torch.stack(cropped_depth_maps, dim=0)
            if cropped_depth_maps is not None
            else None
        ),
    )


def maybe_make_zero_aperture_batch(
    inputs,
    targets,
    bokeh_strength,
    probability,
):
    if probability <= 0:
        return targets, bokeh_strength

    batch_size = inputs.size(0)
    if probability >= 1.0:
        zero_mask = torch.ones(batch_size, device=inputs.device, dtype=torch.bool)
    else:
        zero_mask = torch.rand(batch_size, device=inputs.device) < probability
    if not bool(zero_mask.any()):
        return targets, bokeh_strength

    def sample_mask_for(tensor):
        return zero_mask.view(batch_size, *([1] * (tensor.dim() - 1)))

    return (
        torch.where(sample_mask_for(targets), inputs.detach(), targets),
        torch.where(
            sample_mask_for(bokeh_strength),
            torch.zeros_like(bokeh_strength),
            bokeh_strength,
        )
    )


def laplacian_magnitude(images):
    kernel = images.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    )
    kernel = kernel.view(1, 1, 3, 3).repeat(images.size(1), 1, 1, 1)
    lap = F.conv2d(images.float(), kernel, padding=1, groups=images.size(1))
    return lap.abs().mean(dim=1, keepdim=True)


def pearson_corr_loss(values, targets, eps=1e-6):
    values = values.flatten(1)
    targets = targets.flatten(1)
    values = values - values.mean(dim=1, keepdim=True)
    targets = targets - targets.mean(dim=1, keepdim=True)
    numerator = (values * targets).mean(dim=1)
    value_std = values.square().mean(dim=1).sqrt()
    target_std = targets.square().mean(dim=1).sqrt()
    valid = (value_std > eps) & (target_std > eps)
    if not bool(valid.any()):
        return values.sum() * 0.0
    corr = numerator[valid] / (value_std[valid] * target_std[valid]).clamp_min(eps)
    return 1.0 - corr.mean()


def sample_shifted_focus(depth_map, disp_focus, max_shift, min_shift):
    if depth_map is None or disp_focus is None or max_shift <= 0:
        return None, None

    batch_size = depth_map.size(0)
    focus = disp_focus.view(batch_size, -1).mean(dim=1).view(batch_size, 1, 1, 1)
    sign = torch.where(
        torch.rand_like(focus) < 0.5,
        torch.full_like(focus, -1.0),
        torch.ones_like(focus),
    )
    shift = torch.empty_like(focus).uniform_(min_shift, max_shift)
    alt_focus = (focus + sign * shift).clamp(0.0, 1.0)
    alt_defocus_map = (depth_map - alt_focus).abs().clamp(0.0, 1.0)
    return alt_focus.view(batch_size), alt_defocus_map


def scale_bokeh_strength(bokeh_strength, scale):
    view_shape = [bokeh_strength.size(0)] + [1] * (bokeh_strength.dim() - 1)
    return bokeh_strength * scale.view(*view_shape)


def bokeh_strength_to_spatial_scale(bokeh_strength, target):
    if bokeh_strength.dim() == 0:
        bokeh_strength = bokeh_strength.view(1, 1)
    elif bokeh_strength.dim() > 2:
        bokeh_strength = bokeh_strength.mean(dim=tuple(range(2, bokeh_strength.dim())))
    bokeh_strength = bokeh_strength.view(bokeh_strength.size(0), -1).mean(dim=1)
    return bokeh_strength.view(
        bokeh_strength.size(0),
        *([1] * (target.dim() - 1)),
    )


def defocus_laplacian_consistency_loss(
    model,
    inputs,
    bokeh_strength,
    pos_map,
    depth_map,
    disp_focus,
    focus_shift_max,
    focus_shift_min,
    strength_low_ratio,
    strength_margin,
):
    alt_focus, alt_defocus_map = sample_shifted_focus(
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
        pos_map=pos_map,
        bokeh_strength_map=alt_defocus_map,
        depth_map=depth_map,
        disp_focus=alt_focus,
    )
    alt_outputs, _ = split_model_outputs(alt_outputs)

    source_lap = laplacian_magnitude(inputs.detach())
    alt_lap = laplacian_magnitude(alt_outputs.clamp(0.0, 1.0))
    blur_residual = (source_lap - alt_lap).relu()
    corr_target = alt_defocus_map * bokeh_strength_to_spatial_scale(
        bokeh_strength, alt_defocus_map
    )
    corr_loss = pearson_corr_loss(blur_residual, corr_target)

    strength_loss = inputs.sum() * 0.0
    if strength_low_ratio > 0.0:
        scale = torch.empty(
            inputs.size(0),
            device=inputs.device,
            dtype=inputs.dtype,
        ).uniform_(0.0, strength_low_ratio)
        low_strength = scale_bokeh_strength(bokeh_strength, scale)
        low_outputs = model(
            source=inputs,
            bokeh_strength=low_strength,
            pos_map=pos_map,
            bokeh_strength_map=alt_defocus_map,
            depth_map=depth_map,
            disp_focus=alt_focus,
        )
        low_outputs, _ = split_model_outputs(low_outputs)
        low_lap = laplacian_magnitude(low_outputs.clamp(0.0, 1.0))
        low_residual = (source_lap - low_lap).relu()
        strength_delta = bokeh_strength_to_spatial_scale(
            bokeh_strength - low_strength, alt_defocus_map
        )
        margin = strength_margin * strength_delta.clamp_min(0.0) * alt_defocus_map
        strength_loss = (low_residual - blur_residual + margin).relu()
        strength_loss = (strength_loss * alt_defocus_map).mean()

    return corr_loss, strength_loss


def train_one_epoch(
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
        pos_map = batch["pos_map"].to(device, non_blocking=True).float()
        bokeh_strength_map = (
            batch["bokeh_strength_map"].to(device, non_blocking=True).float()
        )
        depth_map = (
            batch["depth_map"].to(device, non_blocking=True).float()
            if "depth_map" in batch
            else None
        )
        disp_focus = (
            batch["disp_focus"].to(device, non_blocking=True).float()
            if "disp_focus" in batch
            else None
        )
        inputs, targets, pos_map, bokeh_strength_map, depth_map = (
            random_crop_train_tensors(
                inputs=inputs,
                targets=targets,
                pos_map=pos_map,
                bokeh_strength_map=bokeh_strength_map,
                crop_size=crop_size,
                depth_map=depth_map,
            )
        )
        targets, bokeh_strength = maybe_make_zero_aperture_batch(
            inputs=inputs,
            targets=targets,
            bokeh_strength=bokeh_strength,
            probability=zero_aperture_batch_p,
        )

        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0

        if amp:
            with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                outputs = model(
                    source=inputs,
                    bokeh_strength=bokeh_strength,
                    pos_map=pos_map,
                    bokeh_strength_map=bokeh_strength_map,
                    depth_map=depth_map,
                    disp_focus=disp_focus,
                )
                primary_outputs, _ = split_model_outputs(outputs)
                loss = 0.0
                for crit, weight in zip(criterion, criterion_weight):
                    # print(
                    #     f"loss {crit.__class__.__name__}: {(weight * crit(outputs, targets)).item():.6f}"
                    # )
                    if isinstance(crit, LearnedPerceptualImagePatchSimilarity):
                        lpips_loss = crit(
                            primary_outputs.clamp(0.0, 1.0), targets.clamp(0.0, 1.0)
                        )
                        crit.reset()
                        loss = loss + weight * lpips_loss
                    else:
                        loss = loss + weight * crit(primary_outputs, targets)
                loss = add_prior_sobel_gradient_loss(loss, outputs, targets)
            loss_value += float(loss.detach())
            scaler.scale(loss).backward()
            del outputs, loss

            if defocus_corr_weight > 0.0 or defocus_strength_weight > 0.0:
                with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    corr_loss, strength_loss = defocus_laplacian_consistency_loss(
                        model=model,
                        inputs=inputs,
                        bokeh_strength=bokeh_strength,
                        pos_map=pos_map,
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
                        + defocus_corr_weight * corr_loss
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
                pos_map=pos_map,
                bokeh_strength_map=bokeh_strength_map,
                depth_map=depth_map,
                disp_focus=disp_focus,
            )
            primary_outputs, _ = split_model_outputs(outputs)
            loss = 0.0
            for crit, weight in zip(criterion, criterion_weight):
                if isinstance(crit, LearnedPerceptualImagePatchSimilarity):
                    lpips_loss = crit(
                        primary_outputs.clamp(0.0, 1.0), targets.clamp(0.0, 1.0)
                    )
                    crit.reset()
                    loss = loss + weight * lpips_loss
                else:
                    loss = loss + weight * crit(primary_outputs, targets)
            loss = add_prior_sobel_gradient_loss(loss, outputs, targets)
            loss_value += float(loss.detach())
            loss.backward()
            del outputs, loss

            if defocus_corr_weight > 0.0 or defocus_strength_weight > 0.0:
                corr_loss, strength_loss = defocus_laplacian_consistency_loss(
                    model=model,
                    inputs=inputs,
                    bokeh_strength=bokeh_strength,
                    pos_map=pos_map,
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
                    + defocus_corr_weight * corr_loss
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
def validate(model, loader, device, metric_funcs, amp=False):
    model.eval()
    for metric_fn in metric_funcs.values():
        if isinstance(metric_fn, nn.Module):
            metric_fn.to(device)
            metric_fn.eval()
    tracker = MetricTracker(*metric_funcs.keys())
    tracker.reset()

    for batch in tqdm(loader, desc="val", leave=False):
        inputs = batch["source"].to(device, non_blocking=True).float()
        targets = batch["target"].to(device, non_blocking=True).float()
        bokeh_strength = batch["bokeh_strength"].to(device, non_blocking=True).float()
        pos_map = batch["pos_map"].to(device, non_blocking=True).float()
        bokeh_strength_map = batch["bokeh_strength_map"].to(device).float()
        depth_map = (
            batch["depth_map"].to(device, non_blocking=True).float()
            if "depth_map" in batch
            else None
        )
        disp_focus = (
            batch["disp_focus"].to(device, non_blocking=True).float()
            if "disp_focus" in batch
            else None
        )
        with autocast(device_type="cuda", enabled=amp, dtype=torch.bfloat16):
            outputs = model(
                source=inputs,
                bokeh_strength=bokeh_strength,
                pos_map=pos_map,
                bokeh_strength_map=bokeh_strength_map,
                depth_map=depth_map,
                disp_focus=disp_focus,
            )
            outputs, _ = split_model_outputs(outputs)

        for metric_name, metric_fn in metric_funcs.items():
            if isinstance(metric_fn, LearnedPerceptualImagePatchSimilarity):
                metric_value = metric_fn(
                    outputs.clamp(0.0, 1.0), targets.clamp(0.0, 1.0)
                ).item()
            else:
                metric_value = metric_fn(outputs, targets).item()
            tracker.update(metric_name, metric_value, n=inputs.size(0))

    return tracker.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/ssddd/chingheng/Deblurring-Dataset/RealBokeh",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "large", "defocus_deblur"],
    )
    parser.add_argument("--binary-bokeh", action="store_true")
    parser.add_argument("--include-source", action="store_true")
    parser.add_argument("--defocus-deblur", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-dir", type=str, default="checkpoints/train")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--pretrained-strict", action="store_true")
    parser.add_argument("--val-metrics", type=str, default="psnr,ssim,lpips")
    parser.add_argument("--metrics-file", type=str, default="val_metrics.txt")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument(
        "--depth-map-dir",
        type=str,
        default="in_depth",
        help=(
            "Depth map directory inside each split for loaders that support it. "
            "Use da2_depth with --depth-mode disparity for Depth Anything V2 maps."
        ),
    )
    parser.add_argument(
        "--depth-mode",
        type=str,
        default="auto",
        choices=[
            "auto",
            "metric_scaled",
            "metric_minmax",
            "metric_inverse",
            "disparity",
            "raw",
        ],
        help=(
            "Depth preprocessing mode for loaders that support depth maps. "
            "auto maps da2_depth/midas_disparity to disparity and metric "
            "depth dirs such as in_depth/da3_depth to metric_scaled."
        ),
    )
    parser.add_argument(
        "--inverse-depth",
        action="store_true",
        help=(
            "Use inverse-depth conditioning in loaders that support it. The "
            "loaded metric depth map is inverted before building "
            "defocus/focal-depth maps."
        ),
    )
    parser.add_argument(
        "--focus-map-type",
        type=str,
        default="cosine",
        choices=["cosine", "gaussian", "gaussain"],
        help=(
            "Point focus-map encoding used by loaders that support focus_map: "
            "cosine keeps the original radial cosine map; gaussian uses a "
            "Gaussian radial map. The misspelling gaussain is accepted as an alias."
        ),
    )
    parser.add_argument(
        "--focus-map-gaussian-sigma",
        type=float,
        default=None,
        help=(
            "Absolute Gaussian sigma in pixels for --focus-map-type gaussian. "
            "If omitted, sigma is radius / --focus-map-gaussian-sigma-scale."
        ),
    )
    parser.add_argument(
        "--focus-map-gaussian-sigma-scale",
        type=float,
        default=6.0,
        help=(
            "Gaussian sigma divisor for focus maps when absolute sigma is not "
            "set. Larger values make the Gaussian focus map sharper."
        ),
    )
    parser.add_argument("--zero-aperture-batch-p", type=float, default=0.0)
    parser.add_argument("--use-mres-after-in-stage2", action="store_true")
    parser.add_argument("--use-mres-before-out-stage", action="store_true")
    parser.add_argument("--mres-num-blocks", type=int, default=2)
    parser.add_argument("--use-occ-refine", action="store_true")
    parser.add_argument("--use-prior", action="store_true")
    parser.add_argument(
        "--disable-radiance-visualization",
        action="store_true",
        help=(
            "Disable the radiance visualization branch by feeding a zero radiance "
            "map into defocus_fuse. This keeps channel sizes unchanged for ablation."
        ),
    )
    parser.add_argument("--not_out_stage_pos_map", action="store_true")
    parser.add_argument("--attn-version", type=str, default="v1")
    parser.add_argument("--freeze-except-depth-defocus", action="store_true")
    parser.add_argument(
        "--freeze-rel-pos-attention-blocks",
        action="store_true",
        help=(
            "Freeze DMA rel-pos attention blocks after loading pretrained weights. "
            "Useful when fine-tuning rel_then_defocus without destroying focal/depth control."
        ),
    )
    parser.add_argument(
        "--trainable-param-prefixes",
        type=str,
        default=(
            "gen_defocus_scale,defocus_fuse,conv_last,rad_viz,"
            "defocus_head_proj,aperture_scale_proj,rel_gate_scale,rel_gate_bias,d2f"
        ),
        help=(
            "Comma-separated parameter name prefixes kept trainable when "
            "--freeze-except-depth-defocus is enabled."
        ),
    )
    parser.add_argument(
        "--depth-defocus-lr",
        type=float,
        default=None,
        help=(
            "Learning rate for depth/focal defocus parameters. If set together "
            "with --backbone-lr, enables differential learning rates."
        ),
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help=(
            "Learning rate for all other trainable parameters. If omitted while "
            "--depth-defocus-lr is set, defaults to --lr."
        ),
    )
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=0.2)
    parser.add_argument(
        "--defocus-corr-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for Laplacian blur-residual correlation with shifted-focus "
            "defocus maps. Disabled by default."
        ),
    )
    parser.add_argument(
        "--defocus-strength-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for monotonic Laplacian blur-residual consistency between "
            "low and high bokeh strengths. Disabled by default."
        ),
    )
    parser.add_argument(
        "--focus-shift-min",
        type=float,
        default=0.08,
        help="Minimum random normalized focal-depth shift for defocus consistency.",
    )
    parser.add_argument(
        "--focus-shift-max",
        type=float,
        default=0.35,
        help="Maximum random normalized focal-depth shift for defocus consistency.",
    )
    parser.add_argument(
        "--strength-low-ratio",
        type=float,
        default=0.5,
        help=(
            "Upper bound for the random low-strength scale used by "
            "--defocus-strength-weight."
        ),
    )
    parser.add_argument(
        "--defocus-strength-margin",
        type=float,
        default=0.02,
        help="Margin scale for strength monotonicity on Laplacian blur residuals.",
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="vgg",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone used by torchmetrics LPIPS loss/metric.",
    )
    args = parser.parse_args()
    if args.focus_map_type == "gaussain":
        args.focus_map_type = "gaussian"

    set_seed(args.seed)

    if not 0.0 <= args.zero_aperture_batch_p <= 1.0:
        raise ValueError("--zero-aperture-batch-p must be in [0, 1].")
    if args.defocus_corr_weight < 0.0 or args.defocus_strength_weight < 0.0:
        raise ValueError("Defocus consistency loss weights must be >= 0.")
    if args.focus_shift_min < 0.0 or args.focus_shift_max < args.focus_shift_min:
        raise ValueError("--focus-shift-max must be >= --focus-shift-min >= 0.")
    if not 0.0 <= args.strength_low_ratio <= 1.0:
        raise ValueError("--strength-low-ratio must be in [0, 1].")
    if args.defocus_strength_margin < 0.0:
        raise ValueError("--defocus-strength-margin must be >= 0.")
    if args.focus_map_gaussian_sigma is not None and args.focus_map_gaussian_sigma <= 0:
        raise ValueError("--focus-map-gaussian-sigma must be > 0.")
    if args.focus_map_gaussian_sigma_scale <= 0:
        raise ValueError("--focus-map-gaussian-sigma-scale must be > 0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and not use_amp:
        print("AMP requested but CUDA is unavailable, running without AMP.")

    dataset_params = inspect.signature(RealBokeh).parameters
    dataset_accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in dataset_params.values()
    )
    dataset_kwargs = {}
    if "depth_map_dir" in dataset_params or dataset_accepts_kwargs:
        dataset_kwargs["depth_map_dir"] = args.depth_map_dir
    if args.depth_mode != "auto" and (
        "depth_mode" in dataset_params or dataset_accepts_kwargs
    ):
        dataset_kwargs["depth_mode"] = args.depth_mode
    if "inverse_depth" in dataset_params or dataset_accepts_kwargs:
        dataset_kwargs["inverse_depth"] = args.inverse_depth
    if "focus_map_type" in dataset_params or dataset_accepts_kwargs:
        dataset_kwargs["focus_map_type"] = args.focus_map_type
    if "focus_map_gaussian_sigma" in dataset_params or dataset_accepts_kwargs:
        dataset_kwargs["focus_map_gaussian_sigma"] = args.focus_map_gaussian_sigma
    if "focus_map_gaussian_sigma_scale" in dataset_params or dataset_accepts_kwargs:
        dataset_kwargs["focus_map_gaussian_sigma_scale"] = (
            args.focus_map_gaussian_sigma_scale
        )

    train_dataset = RealBokeh(
        data_path=args.data_dir,
        mode=Mode.TRAIN,
        binary_bokeh=args.binary_bokeh,
        defocus_deblur_mode=args.defocus_deblur,
        device="cpu",
        include_source=args.include_source,
        **dataset_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    val_dataset = RealBokeh(
        data_path=args.data_dir,
        mode=Mode.VAL,
        binary_bokeh=args.binary_bokeh,
        defocus_deblur_mode=args.defocus_deblur,
        device="cpu",
        **dataset_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    run_size = args.size
    if args.binary_bokeh and args.size in ("small", "large"):
        run_size = f"{args.size}_bin"
    if args.defocus_deblur:
        run_size = "defocus_deblur"
    config = bokehlicious_size_builder(run_size)

    model = build_model(Bokehlicious, config, args).to(device)

    criterion = [
        nn.L1Loss(),
        LearnedPerceptualImagePatchSimilarity(
            net_type=args.lpips_net, normalize=True
        ).to(device),
    ]
    criterion_weight = [args.l1_weight, args.lpips_weight]

    metric_names = [m.strip().lower() for m in args.val_metrics.split(",") if m.strip()]
    metric_funcs = build_metric_funcs(metric_names, "alex")

    start_epoch = 1
    best_val = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from {args.resume}, start epoch {start_epoch}")
    elif args.pretrained:
        load_pretrained_weights(
            model=model,
            pretrained_path=args.pretrained,
            device=device,
            strict=args.pretrained_strict,
        )

    depth_defocus_prefixes = [
        prefix.strip()
        for prefix in args.trainable_param_prefixes.split(",")
        if prefix.strip()
    ]
    if args.freeze_except_depth_defocus:
        freeze_except_prefixes(model, depth_defocus_prefixes)
    if args.freeze_rel_pos_attention_blocks:
        freeze_rel_pos_attention_blocks(model)

    optimizer = build_optimizer(model, args, depth_defocus_prefixes)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-8
    )
    scaler = GradScaler("cuda", enabled=use_amp)

    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None and use_amp:
            scaler.load_state_dict(ckpt["scaler"])

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            criterion_weight=criterion_weight,
            device=device,
            amp=use_amp,
            log_interval=args.log_interval,
            crop_size=args.crop_size,
            zero_aperture_batch_p=args.zero_aperture_batch_p,
            defocus_corr_weight=args.defocus_corr_weight,
            defocus_strength_weight=args.defocus_strength_weight,
            focus_shift_max=args.focus_shift_max,
            focus_shift_min=args.focus_shift_min,
            strength_low_ratio=args.strength_low_ratio,
            strength_margin=args.defocus_strength_margin,
        )
        scheduler.step()
        print(f"train loss: {train_loss:.6f}")

        save_checkpoint(
            os.path.join(args.save_dir, "latest.pt"),
            model,
            optimizer,
            scaler if use_amp else None,
            epoch,
            best_val,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

        val_results = None
        if args.val_every > 0 and epoch % args.val_every == 0:
            val_results = validate(
                model=model,
                loader=val_loader,
                device=device,
                metric_funcs=metric_funcs,
                amp=use_amp,
            )
            print(
                "val metrics:",
                ", ".join(f"{k}={v:.6f}" for k, v in val_results.items()),
            )
            metrics_path = os.path.join(args.save_dir, args.metrics_file)
            append_metrics_txt(metrics_path, epoch, val_results)

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scaler if use_amp else None,
                epoch,
                best_val,
            )

        if (
            val_results is not None
            and "psnr" in val_results
            and val_results["psnr"] > best_val
        ):
            best_val = val_results["psnr"]
            save_checkpoint(
                os.path.join(args.save_dir, "best.pt"),
                model,
                optimizer,
                scaler if use_amp else None,
                epoch,
                best_val,
            )


if __name__ == "__main__":
    main()
