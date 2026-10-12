"""Compatibility with newer libraries than the ones the code was written for."""
import huggingface_hub


def patch_huggingface_hub():
    """diffusers 0.25 (required by PiSA-SR's model code) imports huggingface_hub.cached_download, removed in hub 0.26.
    Call before importing diffusers."""
    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
