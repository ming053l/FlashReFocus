from method.model_dma_v3 import Bokehlicious as _BokehliciousDMAV3


class Bokehlicious(_BokehliciousDMAV3):
    """
    Bokehlicious DMA-v3 variant with split attention blocks.

    In each DMA basic layer with k attention blocks, the first floor(k / 2)
    blocks use pure relative-position bias and the remaining blocks use pure
    defocus-map bias.
    """

    def __init__(self, *args, attn_version: str = "rel_then_defocus", **kwargs):
        super().__init__(*args, attn_version=attn_version, **kwargs)


BokehliciousRelDMAV3 = Bokehlicious
BokehliciousDMA = Bokehlicious
