#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path


def _expand_locations(text):
    """Expand ${FLASHREFOCUS_REPO}, ${FLASHREFOCUS_WEIGHTS} and ${FLASHREFOCUS_DATA} (defaults as in paths.py)."""
    repo = Path(__file__).resolve().parents[2]
    env = {"FLASHREFOCUS_REPO": str(repo),
           "FLASHREFOCUS_WEIGHTS": os.environ.get("FLASHREFOCUS_WEIGHTS", str(repo / "weights")),
           "FLASHREFOCUS_DATA": os.environ.get("FLASHREFOCUS_DATA", str(repo / "data"))}
    for k, v in env.items():
        text = text.replace("${" + k + "}", v)
    return text


def load_options_config(config_path):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            return json.load(f) or {}
        text = _expand_locations(f.read())
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError:
        return parse_simple_yaml(text)


def parse_scalar(value):
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        return [] if not body else [parse_scalar(part) for part in body.split(",")]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("\"'")


def parse_simple_yaml(text):
    data = {}
    current_key = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            data.setdefault(current_key, []).append(parse_scalar(stripped[2:]))
            continue
        key, _, value = stripped.partition(":")
        current_key = key
        data[key] = [] if value.strip() == "" else parse_scalar(value)
    return data


def add_arg(parser, name, default=None, **kwargs):
    parser.add_argument(f"--{name}", default=default, **kwargs)


def add_bool(parser, name, default=False):
    parser.add_argument(
        f"--{name}",
        action=argparse.BooleanOptionalAction,
        default=default,
    )


def normalize_legacy_config_keys(config):
    normalized = dict(config)
    for removed_key in (
        "sft_dino_num_scales",
        "sft_dino_layer_indices",
        "enable_sft_conditioning",
        "sft_cond_channels",
        "sft_target_blocks",
        "enable_semantic_fusion",
        "semantic_fusion_dim",
        "semantic_fusion_target_blocks",
    ):
        normalized.pop(removed_key, None)
    return normalized


def parse_args(input_args=None):
    raw_args = list(sys.argv[1:] if input_args is None else input_args)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(raw_args)

    parser = argparse.ArgumentParser("Stage I training data")
    add_arg(parser, "config", type=str)
    add_arg(parser, "pretrained_model_name_or_path", type=str)
    add_arg(parser, "controlnet_model_name_or_path", type=str)
    add_arg(parser, "init_checkpoint", type=str)
    add_arg(parser, "resume_from_checkpoint", type=str)
    add_arg(parser, "revision", type=str)
    add_arg(parser, "tokenizer_name", type=str)
    add_arg(parser, "output_dir", "runs/deblur")
    add_arg(parser, "seed", type=int)

    add_arg(parser, "dataset_name", "dpdd")
    add_arg(parser, "task_mode", "defocus_deblurring")
    add_arg(parser, "train_data_dir", type=str)
    add_arg(parser, "dpdd_split", "train")
    add_arg(parser, "validation_split", "val")
    add_arg(parser, "hypersim_metadata_name", "bokeh_random_metadata.json")
    add_arg(parser, "hypersim_defocus_scale", 10.0, type=float)
    add_arg(parser, "hypersim_depth_mode", "metric_depth")
    add_arg(parser, "realbokeh_defocus_scale", 50.0, type=float)
    add_arg(parser, "clear_data_dir", type=str)
    add_arg(parser, "mixed_dpdd_data_dir", type=str)
    add_arg(parser, "mixed_deblurring_data_dir", type=str)
    add_arg(parser, "lfdof_data_dir", type=str)
    add_arg(parser, "mixed_dpdd_prob", 0.34, type=float)
    add_arg(parser, "mixed_deblurring_prob", 0.33, type=float)
    add_arg(parser, "lfdof_prob", 0.0, type=float)
    add_arg(parser, "synthetic_prob", 0.33, type=float)
    add_arg(parser, "synthetic_blur_radius_min", 1.0, type=float)
    add_arg(parser, "synthetic_blur_radius_max", 8.0, type=float)
    add_bool(parser, "synthetic_spatially_varying", True)
    add_arg(
        parser,
        "synthetic_blur_type",
        "mixed",
        choices=["disk", "gaussian", "motion", "mixed", "both"],
    )
    add_arg(
        parser,
        "synthetic_blur_map_mode",
        "gradient",
        choices=[
            "random",
            "gradient",
            "linear",
            "directional",
            "constant",
            "uniform",
            "patch",
        ],
    )
    add_arg(parser, "synthetic_gradient_direction", "random")

    add_arg(parser, "resolution", 512, type=int)
    add_arg(parser, "train_resolutions")
    add_arg(parser, "train_batch_size", 1, type=int)
    add_arg(parser, "gradient_accumulation_steps", 1, type=int)
    add_arg(parser, "num_train_epochs", 1, type=int)
    add_arg(parser, "max_train_steps", type=int)
    add_arg(parser, "checkpointing_steps", 500, type=int)
    add_arg(parser, "dataloader_num_workers", 0, type=int)

    add_arg(parser, "learning_rate", 5e-5, type=float)
    add_arg(parser, "controlnet_learning_rate", type=float)
    add_arg(parser, "unet_learning_rate", type=float)
    add_arg(parser, "ram_plus_learning_rate", type=float)
    add_arg(parser, "vae_refiner_learning_rate", type=float)
    add_arg(parser, "text_embedding_adapter_learning_rate", type=float)
    add_arg(parser, "lr_scheduler", "constant")
    add_arg(parser, "lr_warmup_steps", 0, type=int)
    add_arg(parser, "lr_num_cycles", 1, type=int)
    add_arg(parser, "lr_power", 1.0, type=float)
    add_arg(parser, "adam_beta1", 0.9, type=float)
    add_arg(parser, "adam_beta2", 0.999, type=float)
    add_arg(parser, "adam_weight_decay", 0.01, type=float)
    add_arg(parser, "adam_epsilon", 1e-8, type=float)
    add_arg(parser, "max_grad_norm", 1.0, type=float)
    add_bool(parser, "use_8bit_adam", False)
    add_bool(parser, "set_grads_to_none", False)
    add_bool(parser, "train_controlnet", True)

    add_arg(parser, "bokeh_defocus_scale", type=float)

    add_bool(parser, "enable_vae_refiner", False)
    add_bool(parser, "train_vae_refiner_only", False)
    add_bool(parser, "save_frozen_components", False)
    add_bool(parser, "train_vae_decoder", False)
    add_arg(parser, "vae_refiner_checkpoint", type=str)
    add_arg(
        parser,
        "vae_refiner_variant",
        "with_d",
        choices=["with_d", "wo_d"],
    )
    add_arg(parser, "vae_refiner_depth", 2, type=int)
    add_arg(parser, "vae_refiner_map_hidden_channels", 64, type=int)
    add_bool(parser, "vae_refiner_use_pixel_weighting", False)
    add_bool(parser, "vae_refiner_use_high_freq_info", False)
    add_arg(parser, "vae_refiner_l1_weight", 1.0, type=float)
    add_arg(parser, "vae_refiner_mse_weight", 0.0, type=float)
    add_arg(parser, "vae_refiner_pal_weight", 0.0, type=float)
    add_arg(parser, "vae_refiner_pal_eps", 1e-3, type=float)
    add_arg(parser, "vae_refiner_perceptual_weight", 0.0, type=float)
    add_arg(parser, "vae_refiner_lpips_net", "vgg")
    add_arg(parser, "vae_refiner_binary_weight", 0.0, type=float)
    add_arg(parser, "vae_refiner_target_recon_l1_weight", 0.0, type=float)
    add_arg(parser, "vae_refiner_target_recon_mse_weight", 0.0, type=float)
    add_arg(parser, "vae_refiner_target_recon_pal_weight", 0.0, type=float)
    add_arg(parser, "vae_refiner_target_recon_perceptual_weight", 0.0, type=float)

    add_arg(parser, "mixed_precision", None, choices=["no", "fp16", "bf16", None])
    add_bool(parser, "allow_tf32", False)
    add_bool(parser, "gradient_checkpointing", False)
    add_bool(parser, "enable_xformers_memory_efficient_attention", False)

    add_arg(parser, "conditioning_source", "source")
    add_arg(parser, "control_rgb_loss_weight", 0.0, type=float)
    add_bool(parser, "use_rrdb", False)
    add_arg(parser, "caption", "")
    add_arg(parser, "proportion_empty_prompts", 0.0, type=float)
    add_arg(parser, "control_type", "defocus")
    add_bool(parser, "enable_text_embedding_adapter", False)
    add_arg(parser, "text_embedding_adapter_hidden_dim", type=int)
    add_arg(parser, "text_embedding_adapter_dropout", 0.0, type=float)
    add_arg(parser, "text_embedding_adapter_residual_scale", 1.0, type=float)
    add_arg(parser, "text_embedding_adapter_checkpoint", type=str)
    add_arg(parser, "defocus_prior_mode", "raw")
    add_arg(parser, "defocus_band_thresholds", [0.33, 0.66], nargs="*", type=float)

    add_bool(parser, "use_mid_pixelwise_attention", False)
    add_arg(parser, "trainable_modules", ["pixel_attentions"], nargs="*")
    add_arg(parser, "timestep_min", type=int)
    add_arg(parser, "timestep_max", type=int)

    add_bool(parser, "enable_ram_caption", False)
    add_bool(parser, "finetune_ram_plus", False)
    add_arg(parser, "ram_plus_checkpoint_path", type=str)
    add_arg(parser, "ram_plus_image_size", 384, type=int)
    add_arg(parser, "ram_plus_vit", "swin_l")
    add_arg(parser, "ram_plus_fallback_prompt", "a clear source image")
    add_arg(parser, "ram_plus_prompt_threshold", type=float)
    add_arg(parser, "ram_plus_max_tags", 20, type=int)

    add_arg(parser, "num_validation_images", 1, type=int)
    add_arg(parser, "num_validation_samples", 0, type=int)
    add_arg(parser, "validation_steps", 1000, type=int)
    add_arg(parser, "validation_metric", "psnr")
    add_bool(parser, "validation_use_original_resolution", True)
    add_arg(parser, "validation_max_resolution", 1024, type=int)
    add_arg(parser, "validation_size_divisor", 8, type=int)
    add_bool(parser, "validation_seed_per_sample", False)
    add_arg(parser, "validation_scheduler", "unipc")
    add_arg(parser, "validation_prompt", nargs="*")
    add_arg(parser, "validation_image", nargs="*")
    add_bool(parser, "compute_extended_metrics", False)
    add_arg(parser, "num_inference_steps", 20, type=int)
    add_arg(parser, "denoising_start", type=float)
    add_arg(parser, "guidance_scale", 1.0, type=float)
    add_arg(parser, "conditioning_scale", 1.0, type=float)
    add_arg(parser, "negative_prompt", type=str)
    add_bool(parser, "init_latent_with_noise", False)
    add_arg(parser, "added_noise_level", 0, type=int)
    add_arg(parser, "offset_noise_scale", 0.0, type=float)
    add_bool(parser, "use_noise_adjustment", False)
    add_arg(parser, "noise_adjustment_alpha", 0.0, type=float)
    add_arg(parser, "latent_tiled_size", 256, type=int)
    add_arg(parser, "latent_tiled_overlap", 8, type=int)
    add_arg(parser, "iterative_defocus_noise_alpha", 0.0, type=float)
    add_arg(parser, "ecc_xinit_dir", type=str)
    add_bool(parser, "ecc_xinit_required", False)
    add_arg(parser, "ecc_start_timestep", type=int)
    add_arg(parser, "ecc_correction_scale", 0.0, type=float)
    add_arg(parser, "ecc_correction_until_step_ratio", 0.5, type=float)

    add_bool(parser, "push_to_hub", False)
    add_arg(parser, "hub_token", type=str)
    add_arg(parser, "hub_model_id", type=str)
    add_arg(parser, "report_to", "none")

    if config_args.config:
        config = normalize_legacy_config_keys(load_options_config(config_args.config))
        valid = {action.dest for action in parser._actions}
        unknown = sorted(set(config) - valid)
        if unknown:
            raise ValueError(f"Unknown config option(s): {', '.join(unknown)}")
        parser.set_defaults(**config)

    args = parser.parse_args(raw_args)
    if args.mixed_precision == "no":
        args.mixed_precision = None
    return args


if __name__ == "__main__":
    parse_args()
