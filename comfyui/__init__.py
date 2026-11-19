"""ComfyUI custom nodes for FlashReFocus. Link or copy this folder into ComfyUI/custom_nodes/ (see README.md); the
repository root is put on the import path, so the flashrefocus package does not need to be installed separately."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
