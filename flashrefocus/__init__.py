"""FlashReFocus: interactive image refocusing. Stage I deblurs once with one-step diffusion (flashrefocus.deblurring),
Stage II renders every focal edit with a lightweight defocus-aware renderer (flashrefocus.rendering);
flashrefocus.pipeline combines them."""
from .pipeline import Refocuser, Scene

__all__ = ["Refocuser", "Scene"]
