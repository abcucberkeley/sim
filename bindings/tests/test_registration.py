"""Masked FFT registration and mosaic stitching."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import sirius


def scene(shape, seed=0):
    """Textured field every tile is cut from."""
    rng = np.random.default_rng(seed)
    z, y, x = shape
    gz, gy, gx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    field = (
        np.sin(0.31 * gx + 0.7) * np.cos(0.23 * gy + 1.1)
        + 0.6 * np.sin(0.11 * (gx + gy))
        + 0.4 * np.cos(0.47 * gz + 0.19 * gx)
    )
    return (field + rng.uniform(-0.05, 0.05, shape) + 2.0).astype(np.float32)


class TestNextFastFFTSize(unittest.TestCase):
    def test_smooth_sizes(self):
        self.assertEqual(sirius.next_fast_fft_size(16), 16)
        self.assertEqual(sirius.next_fast_fft_size(17), 18)
        self.assertEqual(sirius.next_fast_fft_size(1021), 1024)
        for n in range(1, 200):
            m = sirius.next_fast_fft_size(n)
            self.assertGreaterEqual(m, n)
            for f in (2, 3, 5, 7):
                while m % f == 0:
                    m //= f
            self.assertEqual(m, 1)


class TestMaskedNcc(unittest.TestCase):
    def test_matches_numpy_normalized_cross_correlation(self):
        rng = np.random.default_rng(3)
        fixed = rng.normal(size=(9, 11))
        moving = rng.normal(size=(5, 4))
        corr, overlap = sirius.masked_ncc(fixed, moving)
        self.assertEqual(corr.shape, (1, 13, 14))
        self.assertEqual(overlap.shape, corr.shape)

        # Spot-check a fully overlapping displacement against numpy.
        shift = (0, 2, 3)
        index = tuple(s + m - 1 for s, m in zip(shift, (1,) + moving.shape))
        window = fixed[shift[1]:shift[1] + moving.shape[0], shift[2]:shift[2] + moving.shape[1]]
        expected = np.corrcoef(window.ravel(), moving.ravel())[0, 1]
        self.assertEqual(overlap[index], moving.size)
        self.assertAlmostEqual(corr[index], expected, places=9)

    def test_index_to_shift_convention(self):
        field = scene((1, 24, 30), seed=1).astype(np.float64)
        block = field[:, 5:17, 9:21]
        result = sirius.register_translation_masked(field, block)
        self.assertTrue(result.valid)
        self.assertEqual(tuple(result.integer_shift), (0, 5, 9))
        self.assertGreater(result.correlation, 0.999)

    def test_mask_excludes_corrupted_voxels(self):
        field = scene((1, 64, 64), seed=2).astype(np.float64)
        tile = field[:, 20:52, 24:56].copy()
        mask = np.ones(tile.shape, dtype=np.uint8)
        tile[:, :14, :] = 60.0          # a saturated block
        mask[:, :14, :] = 0

        options = sirius.MaskedNccOptions()
        options.required_overlap_fraction = 0.25
        masked = sirius.register_translation_masked(field, tile, moving_mask=mask, options=options)
        unmasked = sirius.register_translation_masked(field, tile, options=options)
        self.assertEqual(tuple(masked.integer_shift), (0, 20, 24))
        self.assertNotEqual(tuple(unmasked.integer_shift), (0, 20, 24))

    def test_search_range_is_respected(self):
        field = scene((1, 32, 32), seed=4).astype(np.float64)
        block = field[:, 11:21, 7:17]
        options = sirius.MaskedNccOptions()
        options.max_shift = [0, 4, 4]
        limited = sirius.register_translation_masked(field, block, options=options)
        self.assertTrue(limited.valid)
        self.assertLessEqual(abs(limited.integer_shift[1]), 4)
        self.assertLessEqual(abs(limited.integer_shift[2]), 4)

    def test_dtype_mismatch_raises(self):
        a = np.zeros((8, 8), dtype=np.float32)
        b = np.zeros((4, 4), dtype=np.float64)
        with self.assertRaises(ValueError):
            sirius.register_translation_masked(a, b)


class TestStitching(unittest.TestCase):
    def setUp(self):
        self.field = scene((2, 60, 110), seed=5)
        self.tiles = [self.field[:, :, 0:60].copy(), self.field[:, :, 50:110].copy()]
        self.truth = [(0.0, 0.0, 0.0), (0.0, 0.0, 50.0)]
        self.nominal = [(0.0, 0.0, 0.0), (0.0, 3.0, 56.0)]   # stage error
        self.options = sirius.StitchOptions()
        self.options.search_radius = [1, 10, 12]

    def test_pair_registration_corrects_the_stage_position(self):
        match = sirius.register_tile_pair(
            self.tiles[0], self.nominal[0], self.tiles[1], self.nominal[1], self.options
        )
        self.assertTrue(match.accepted)
        self.assertGreater(match.correlation, 0.99)
        # The field is smooth, so the correlation peak is broad and the
        # sub-voxel refinement lands a fraction of a voxel off the truth.
        np.testing.assert_allclose(match.displacement, [0, 0, 50], atol=0.3)
        np.testing.assert_allclose(match.nominal_displacement, [0, 3, 56])

    def test_plan_and_fuse_reproduce_the_field(self):
        layout = sirius.plan_stitch(self.tiles, self.nominal, self.options)
        self.assertEqual(len(layout.positions), 2)
        self.assertEqual(len(layout.matches), 1)
        np.testing.assert_allclose(layout.positions[0], self.truth[0], atol=1e-9)
        np.testing.assert_allclose(layout.positions[1], self.truth[1], atol=0.4)
        self.assertEqual(tuple(layout.canvas_extent), (2, 60, 110))

        fused = sirius.fuse_tiles(
            self.tiles, layout.positions, layout.canvas_origin, layout.canvas_extent, self.options
        )
        self.assertEqual(fused.shape, (2, 60, 110))
        np.testing.assert_allclose(fused, self.field, atol=1e-5)

    def test_optimize_positions_uses_the_measured_displacements(self):
        match = sirius.TileMatch()
        match.fixed, match.moving = 0, 1
        match.displacement = [0.0, 0.0, 50.0]
        match.correlation = 0.99
        match.accepted = True
        fitted = sirius.optimize_tile_positions(self.nominal, [match])
        np.testing.assert_allclose(fitted[0], [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(fitted[1], [0, 0, 50], atol=0.1)

    def test_tiff_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [os.path.join(d, f"tile{i}.tif") for i in range(2)]
            for path, tile in zip(paths, self.tiles):
                sirius.write_tiff(path, tile)
            out = os.path.join(d, "mosaic.tif")
            fused, layout = sirius.stitch_tiff_tiles(
                paths, self.nominal, self.options, output_path=out
            )
            self.assertEqual(fused.shape, (2, 60, 110))
            np.testing.assert_allclose(layout.positions[1], self.truth[1], atol=0.4)
            np.testing.assert_allclose(fused, self.field, atol=1e-5)
            np.testing.assert_array_equal(sirius.read_tiff(out), fused)


if __name__ == "__main__":
    unittest.main()
