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


def test_more_depth_edits():
    depth = torch.rand(1, 32, 48)
    mask = torch.zeros(32, 48, dtype=torch.bool)
    mask[8:16, 8:20] = True
    assert torch.allclose(depth_edits.tilted_band(32, 48, 0.5, 0.4, 0.0), depth_edits.tilt_shift(32, 48, 0.4), atol=1e-5)
    assert torch.equal(depth_edits.focal_planes(depth, [0.3, 0.7]), depth_edits.two_focal_planes(depth, 0.3, 0.7))
    r = depth_edits.focus_range(depth, 0.5, 0.1)
    assert torch.all(r[(depth - 0.5).abs() <= 0.1] == 0.5)
    assert torch.all(depth_edits.one_sided(depth, 0.5, "foreground") >= 0.5)
    assert torch.all(depth_edits.one_sided(depth, 0.5, "background") <= 0.5)
    assert depth_edits.radial_focus(32, 48, 0.5, 0.5)[0, 16, 24] == 0.5
    assert torch.all(depth_edits.keep_sharp(depth, mask, 0.4)[0][mask] == 0.4)
    assert torch.all(depth_edits.set_mask_depth(depth, mask, 1.0)[0][mask] == 1.0)
    moved = depth_edits.move_away(depth, mask, 0.5, 0.3)
    assert torch.all((moved[0][mask] - 0.5).abs() >= 0.3 - 1e-6) and torch.equal(moved[0][~mask], depth[0][~mask])
    iso = depth_edits.isolate(depth, mask, 0.5, 0.2)
    assert torch.all(iso[0][mask] == 0.5) and torch.all((iso[0][~mask] - 0.5).abs() >= 0.2 - 1e-6)


def test_spotlight_grows_into_hands_but_not_neighbours():
    depth = torch.full((1, 32, 32), 0.8)
    depth[:, 8:24, 8:16] = 0.4                                   # the subject ...
    depth[:, 8:24, 16:19] = 0.41                                 # ... and a hand the mask missed
    depth[:, 8:24, 20:23] = 0.41                                 # a neighbour at the same depth
    mask = torch.zeros(1, 32, 32, dtype=torch.bool)
    mask[:, 8:24, 8:16] = True
    others = torch.zeros_like(mask)
    others[:, 8:24, 20:23] = True
    edited, grown = depth_edits.spotlight(depth, mask, 0.4, gap=0.5, others=others, grow=3)
    assert grown[:, 8:24, 16:19].all() and not grown[:, 8:24, 20:23].any()
    assert torch.allclose(edited[grown], torch.tensor(0.4))
    assert ((edited[~grown] - 0.4).abs() >= 0.5 - 1e-6).all()


def test_overlay_fill_and_paste_back():
    from flashrefocus.overlays import fill_region, paste_back
    img = torch.full((1, 3, 32, 32), 0.2)
    mask = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
    mask[..., 12:20, 12:20] = True
    titled = torch.where(mask, torch.ones_like(img), img)        # a white "title"
    filled = fill_region(titled, mask)
    assert (filled - 0.2).abs().max() < 1e-4                     # the title is gone
    back = paste_back(filled, titled, mask, feather=0)
    assert torch.allclose(back, titled, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel runs on CUDA")
def test_fused_attention_matches_reference():
    pytest.importorskip("triton")
    from flashrefocus.rendering.defocus_bias_attention import DefocusBiasAttention
    from flashrefocus.rendering.fused import fused_attention_forward
    torch.manual_seed(0)
    att = DefocusBiasAttention(96, 3).cuda().eval()
    att.defocus_alpha.data.uniform_(0.5, 2.0)
    for h, w in [(13, 70), (40, 33)]:                           # lengths that are not multiples of the block size
        x = torch.randn(2, h, w, 96, device="cuda")
        defocus = torch.rand(2, 1, h, w, device="cuda")
        strength = torch.tensor([[0.3], [1.0]], device="cuda")
        with torch.no_grad():
            ref = DefocusBiasAttention.forward(att, x, defocus, strength)
            out = fused_attention_forward(att, x, defocus, strength)
        assert (out - ref).abs().max() < 2e-3                    # TF32 matmuls in the kernel
