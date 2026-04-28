"""Tests for the TIFF I/O bindings (sirius.read_tiff, write_tiff, TiffCompression).

The C++ binding glues both `writeTiff` (2-D) and `writeTiffStack` (3-D) under the
single overloaded Python name `write_tiff`. `read_tiff` always goes through
`readTiffStackAny`, so it always returns a 3-D ndarray (depth=1 for files that
were written from a 2-D image).
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import sirius
from _helpers import silenced_stderr


# Every scalar dtype with a registered write_tiff overload.
SUPPORTED_DTYPES = [
    np.int8, np.uint8,
    np.int16, np.uint16,
    np.int32, np.uint32,
    np.float32, np.float64,
]


def _sample_2d(dtype, rows: int = 4, cols: int = 5, seed: int = 0) -> np.ndarray:
    """Build a small representative 2-D image of the given dtype."""
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal((rows, cols)).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=(rows, cols), dtype=dtype)


def _sample_3d(dtype, depth: int = 3, rows: int = 4, cols: int = 5, seed: int = 0) -> np.ndarray:
    """Build a small representative 3-D stack of the given dtype."""
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal((depth, rows, cols)).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=(depth, rows, cols), dtype=dtype)


class TestTiffCompressionEnum(unittest.TestCase):
    def test_all_members_exposed(self):
        for name in ("NoCompression", "Lzw", "Deflate"):
            self.assertTrue(hasattr(sirius.TiffCompression, name), name)

    def test_distinct_values(self):
        values = {
            sirius.TiffCompression.NoCompression,
            sirius.TiffCompression.Lzw,
            sirius.TiffCompression.Deflate,
        }
        self.assertEqual(len(values), 3)


class TestSingleImageRoundTrip(unittest.TestCase):
    """Write a 2-D image and read it back; payload and dtype must match.

    `read_tiff` always returns 3-D, so the original 2-D image comes back
    as a depth-1 stack -- compare against `original[None, ...]`.
    """

    def test_each_dtype(self):
        for dtype in SUPPORTED_DTYPES:
            with self.subTest(dtype=dtype.__name__):
                original = _sample_2d(dtype)
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, "single.tif")
                    sirius.write_tiff(path, original)
                    loaded = sirius.read_tiff(path)

                self.assertEqual(loaded.dtype, original.dtype)
                self.assertEqual(loaded.shape, (1,) + original.shape)
                np.testing.assert_array_equal(loaded[0], original)


class TestStackRoundTrip(unittest.TestCase):
    """Write a 3-D stack and read it back; payload, shape, and dtype must match."""

    def test_each_dtype(self):
        for dtype in SUPPORTED_DTYPES:
            with self.subTest(dtype=dtype.__name__):
                original = _sample_3d(dtype)
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, "stack.tif")
                    sirius.write_tiff(path, original)
                    loaded = sirius.read_tiff(path)

                self.assertEqual(loaded.dtype, original.dtype)
                self.assertEqual(loaded.shape, original.shape)
                np.testing.assert_array_equal(loaded, original)


class TestCompressionModes(unittest.TestCase):
    """All three compression modes must round-trip losslessly for both ranks."""

    COMPRESSIONS = (
        sirius.TiffCompression.NoCompression,
        sirius.TiffCompression.Lzw,
        sirius.TiffCompression.Deflate,
    )

    def test_2d_round_trip_each_mode(self):
        original = _sample_2d(np.uint16, rows=16, cols=16)
        for comp in self.COMPRESSIONS:
            with self.subTest(comp=comp):
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, "img.tif")
                    sirius.write_tiff(path, original, comp=comp)
                    loaded = sirius.read_tiff(path)
                np.testing.assert_array_equal(loaded[0], original)

    def test_3d_round_trip_each_mode(self):
        original = _sample_3d(np.float32, depth=3, rows=16, cols=16)
        for comp in self.COMPRESSIONS:
            with self.subTest(comp=comp):
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, "stack.tif")
                    sirius.write_tiff(path, original, comp=comp)
                    loaded = sirius.read_tiff(path)
                np.testing.assert_array_equal(loaded, original)

    def test_compression_keyword_is_optional(self):
        original = _sample_2d(np.uint8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "img.tif")
            sirius.write_tiff(path, original)  # no comp= given
            loaded = sirius.read_tiff(path)
        np.testing.assert_array_equal(loaded[0], original)


class TestDimensionsPreserved(unittest.TestCase):
    """Non-square, non-power-of-two shapes must come back exactly."""

    SHAPES_2D = [(1, 1), (1, 100), (100, 1), (17, 31), (256, 256)]
    SHAPES_3D = [(1, 1, 1), (5, 16, 16), (3, 17, 31), (2, 64, 32)]

    def test_2d_shapes(self):
        for shape in self.SHAPES_2D:
            with self.subTest(shape=shape):
                original = _sample_2d(np.uint16, rows=shape[0], cols=shape[1])
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, "img.tif")
                    sirius.write_tiff(path, original)
                    loaded = sirius.read_tiff(path)
                self.assertEqual(loaded.shape, (1,) + shape)
                np.testing.assert_array_equal(loaded[0], original)

    def test_3d_shapes(self):
        for shape in self.SHAPES_3D:
            with self.subTest(shape=shape):
                original = _sample_3d(
                    np.uint16, depth=shape[0], rows=shape[1], cols=shape[2]
                )
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, "stack.tif")
                    sirius.write_tiff(path, original)
                    loaded = sirius.read_tiff(path)
                self.assertEqual(loaded.shape, shape)
                np.testing.assert_array_equal(loaded, original)


class TestReadTiffShapeContract(unittest.TestCase):
    """`read_tiff` always returns 3-D, regardless of how the file was written."""

    def test_single_page_is_depth_one(self):
        original = _sample_2d(np.float64)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "img.tif")
            sirius.write_tiff(path, original)
            loaded = sirius.read_tiff(path)
        self.assertEqual(loaded.ndim, 3)
        self.assertEqual(loaded.shape[0], 1)

    def test_returned_array_is_writable(self):
        # The numpy array is backed by an Eigen::Tensor capsule -- writes
        # should not segfault and should not propagate back to disk.
        original = _sample_2d(np.uint8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "img.tif")
            sirius.write_tiff(path, original)
            loaded = sirius.read_tiff(path)
            loaded[0, 0, 0] = 0
            reloaded = sirius.read_tiff(path)
        # Disk copy is unaffected by mutating the in-memory array.
        np.testing.assert_array_equal(reloaded[0], original)


class TestErrors(unittest.TestCase):
    def test_read_nonexistent_file_raises(self):
        with silenced_stderr():
            with self.assertRaises(RuntimeError):
                sirius.read_tiff("/no/such/sirius_test_file.tif")

    def test_write_unsupported_rank_raises(self):
        # 1-D and 4-D arrays have no registered overload.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.tif")
            with self.assertRaises(TypeError):
                sirius.write_tiff(path, np.zeros(8, dtype=np.uint8))
            with self.assertRaises(TypeError):
                sirius.write_tiff(path, np.zeros((2, 2, 2, 2), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
