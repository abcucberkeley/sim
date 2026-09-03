import os
from pathlib import Path

if os.name == "nt":
    os.add_dll_directory(str(Path(__file__).parent))

from sirius._sirius_ext import (
    Buffer,
    Device,
    DeviceProperties,
    DeviceType,
    FFT,
    PixelType,
    PlanRigor,
    Stream,
    TiffCompression,
    TiffFile,
    TiffImageInfo,
    TiffInfo,
    TiffLayout,
    TiffLevel,
    built_with_cuda,
    built_with_nvtiff,
    cuda_available,
    cuda_device_count,
    device_properties,
    inspect_tiff,
    read_tiff,
    synchronize_device,
    to_device,
    write_tiff,
)

__all__ = [
    "Buffer",
    "Device",
    "DeviceProperties",
    "DeviceType",
    "FFT",
    "PixelType",
    "PlanRigor",
    "Stream",
    "TiffCompression",
    "TiffFile",
    "TiffImageInfo",
    "TiffInfo",
    "TiffLayout",
    "TiffLevel",
    "built_with_cuda",
    "built_with_nvtiff",
    "cuda_available",
    "cuda_device_count",
    "device_properties",
    "inspect_tiff",
    "read_tiff",
    "synchronize_device",
    "to_device",
    "write_tiff",
]
