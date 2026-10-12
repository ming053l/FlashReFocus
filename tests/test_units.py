"""Unit tests that need neither weights nor a GPU:  pytest tests/test_units.py"""
import numpy as np
import pytest
import torch
from PIL import Image

from flashrefocus.deblurring.deblurrer import refine_in_tiles
from flashrefocus.image_io import image_to_tensor, tensor_to_uint8, to_signed, to_unit_range
from flashrefocus.layers import LayerNormFunction, NAFBlock
from flashrefocus.rendering import depth_edits
from flashrefocus.rendering.depth import disparity_to_depth
from flashrefocus.rendering.renderer import Renderer, condition_maps


def test_image_round_trip_is_exact():
    arr = np.random.RandomState(0).randint(0, 256, (37, 53, 3), dtype=np.uint8)
    x = image_to_tensor(Image.fromarray(arr))
    assert x.shape == (1, 3, 37, 53) and x.dtype == torch.float32
    assert np.array_equal(tensor_to_uint8(x), arr)
    # through [-1, 1] and back, truncation can lose one level where the arithmetic rounds below an integer
    assert np.abs(tensor_to_uint8(to_unit_range(to_signed(x))).astype(int) - arr).max() <= 1


def test_image_to_tensor_crops_to_multiple():
    assert image_to_tensor(np.zeros((37, 53, 3), np.uint8), multiple=8).shape == (1, 3, 32, 48)


def test_depth_at_rounds_and_clamps():
    depth = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    assert depth_edits.depth_at(depth, 1.6, 0.4) == depth[0, 0, 2]
    assert depth_edits.depth_at(depth, -5, 10) == depth[0, 2, 0]


@pytest.mark.parametrize("da, db", [(0.2, 0.7), (0.9, 0.3), (1.0, 0.95)])
def test_two_focal_planes(da, db):
    depth = torch.linspace(0, 1, 101).view(1, 1, 101)
    edited = depth_edits.two_focal_planes(depth, da, db)
    assert float(edited.min()) >= 0 and float(edited.max()) <= 1
    for plane in (da, db):          # both planes end up at the focal depth D_A: both sharp
        i = int(round(plane * 100))
        assert abs(float(edited[0, 0, i]) - da) < 1e-6
    assert float((edited - da).abs().max()) > 0.05    # elsewhere the defocus survives (no collapse when D_A is far)


def test_tilt_shift_band():
    d = depth_edits.tilt_shift(100, 8, row=0.4, band=0.1)
    assert torch.allclose(d[0, 32:48], torch.full_like(d[0, 32:48], 0.5))   # inside the band: the focal depth
    column = d[0, :, 0]
    assert torch.all(column[1:] >= column[:-1])                             # monotonic from top to bottom


def test_disparity_to_depth():
    disparity = torch.tensor([[[0.0, 1.0], [2.0, float("nan")]]])
    depth = disparity_to_depth(disparity)
    assert float(depth.max()) == 1.0 and float(depth.min()) == 0.0
    assert float(depth[0, 1, 0]) == 0.0            # largest disparity: nearest
    assert torch.equal(disparity_to_depth(torch.ones(1, 4, 4)), torch.ones(1, 4, 4))


def test_condition_maps():
    depth = torch.linspace(0, 1, 16).view(1, 1, 1, 16).expand(1, 1, 8, 16)
    signed, cond = condition_maps(depth, torch.tensor([[0.5]]), torch.tensor([[0.3]]), (8, 16))
    assert cond.shape == (1, 3, 8, 16)
    assert torch.allclose(signed, depth - 0.5)
    assert torch.all(cond[0, 1, :, :8] == 1)        # M^fg: the focal plane and what lies before it
    assert torch.allclose(cond[0, 2], torch.full((8, 16), 0.3))


def test_renderer_pads_and_crops():
    torch.manual_seed(0)
    model = Renderer().eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 30, 45), torch.tensor([[0.2], [0.8]]), torch.rand(2, 1, 30, 45),
                    torch.tensor([[0.3], [0.6]]))
    assert out.shape == (2, 3, 30, 45) and torch.isfinite(out).all()


def test_naf_block_shapes():
    x = torch.rand(2, 8, 9, 10)
    assert NAFBlock(8)(x).shape == x.shape
    assert NAFBlock(8, cond_channels=3)(x, torch.rand(2, 3, 9, 10)).shape == x.shape


def test_layer_norm_backward():
    x = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, w, b: LayerNormFunction.apply(x, w, b, 1e-6), (x, w, b))


def test_refine_in_tiles_matches_whole_image():
    """For a pointwise refiner the blended tiles must reproduce the whole-image result."""
    pointwise = lambda x, latent: x * 2 + latent.repeat_interleave(8, -1).repeat_interleave(8, -2)[:, :3]
    x, latent = torch.rand(1, 3, 72, 104), torch.rand(1, 4, 9, 13)
    assert torch.allclose(refine_in_tiles(pointwise, x, latent, tile=48, overlap=16), pointwise(x, latent), atol=1e-6)


def test_decoder_refiner_starts_as_the_vae_decoder():
    """The fusion residuals are zero-initialised, so an untrained refiner decodes like the plain VAE decoder."""
    from diffusers import AutoencoderKL
    from flashrefocus.deblurring.decoder_refiner import DecoderRefiner
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=(32, 64, 64, 64), layers_per_block=1, norm_num_groups=16,
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4).eval()
    refiner = DecoderRefiner(vae, guidance_channels=16).eval()
    x, latent = torch.rand(1, 3, 64, 64) * 2 - 1, torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        assert torch.allclose(refiner(x, latent), vae.decode(latent).sample, atol=1e-5)
