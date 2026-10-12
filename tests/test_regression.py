"""End-to-end regression against tests/reference/ (GPU and the released weights; skipped otherwise):
    pytest tests/test_regression.py
On the machine that made the references the outputs are bit-identical (cuDNN autotuning off); other GPUs or library
versions may differ by floating-point rounding, so the test requires PSNR >= 45 dB and reports exact matches."""
import numpy as np
import pytest
import torch
from PIL import Image

from flashrefocus import paths

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not paths.STAGE1.exists(),
                                reason="needs a GPU and the released weights")


@pytest.fixture(scope="module")
def results():
    from flashrefocus import Refocuser
    from regression_cases import IMAGES, outputs
    torch.backends.cudnn.benchmark = False
    refocuser = Refocuser.from_pretrained()
    return {name: outputs(refocuser, name) for name in IMAGES}


def psnr(a, b, peak):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


@pytest.mark.parametrize("name", ["flower", "dock"])
@pytest.mark.parametrize("kind", ["aif", "depth", "bokeh", "two_planes", "tilt_shift"])
def test_matches_reference(results, name, kind):
    from regression_cases import reference_path
    reference = np.asarray(Image.open(reference_path(name, kind)))
    output = results[name][kind]
    assert output.shape == reference.shape
    value = psnr(output, reference, 65535 if kind == "depth" else 255)
    print(f"{name} {kind}: {'exact' if value == float('inf') else f'{value:.1f} dB'}")
    assert value >= 45


def test_separate_unets_are_identical():
    from flashrefocus.deblurring import Deblurrer
    from flashrefocus.image_io import read_image, to_signed
    from regression_cases import HERE
    torch.backends.cudnn.benchmark = False
    x = to_signed(read_image(HERE / "data" / "flower.png", "cuda"))
    swapped = Deblurrer.from_pretrained()(x)
    torch.cuda.empty_cache()
    separate = Deblurrer.from_pretrained(separate_unets=True)(x)
    assert torch.equal(swapped, separate)
