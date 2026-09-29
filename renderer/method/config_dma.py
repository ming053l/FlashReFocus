from method.config import bokehlicious_config_builder, bokehlicious_size_builder
from method.model_dma import Bokehlicious, BokehliciousDMA


def build_bokehlicious_dma(size: str):
    return BokehliciousDMA(**bokehlicious_size_builder(size))
