"""Tests of sirius.workbench: loading a synthetic TIFF and running a pipeline
of numpy steps (the code path of "Export pipeline as Python script")."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest

try:
    import scipy  # noqa: F401
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
from pathlib import Path

import numpy as np

try:
    import tifffile  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    tifffile = None


def _load_workbench():
    """sirius.workbench of this tree. The installed package may be another
    checkout (an editable install elsewhere), so load the file beside this test
    when the import does not resolve to it."""
    here = Path(__file__).resolve().parents[1] / "python" / "sirius" / "workbench.py"
    try:
        import sirius.workbench as wb  # type: ignore

        if Path(wb.__file__).resolve() == here:
            return wb
    except Exception:  # noqa: BLE001
        pass
    spec = importlib.util.spec_from_file_location("sirius_workbench_under_test", here)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sirius_workbench_under_test"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


wb = _load_workbench()


@unittest.skipIf(tifffile is None, "tifffile not installed")
class TestRunPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(3)
        # (t, z, c, y, x) ImageJ hyperstack: 2 channels, 3 time points, 4 planes
        cls.data = (rng.random((3, 4, 2, 16, 24)) * 1000).astype(np.uint16)
        cls.path = os.path.join(cls.tmp.name, "stack.tif")
        tifffile.imwrite(cls.path, cls.data, imagej=True, resolution=(1 / 0.1, 1 / 0.1),
                         metadata={"axes": "TZCYX", "spacing": 0.3, "unit": "um"})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_load_dataset_reads_hyperstack_dims_and_voxel(self):
        a, meta = wb.load_dataset(self.path)
        self.assertEqual(a.shape, (2, 3, 4, 16, 24))
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(meta["dims"], {"c": 2, "t": 3, "z": 4, "y": 16, "x": 24})
        self.assertAlmostEqual(meta["voxel_um"][2], 0.3, places=6)
        self.assertAlmostEqual(meta["voxel_um"][0], 0.1, places=6)
        np.testing.assert_array_equal(a[1, 2, 3], self.data[2, 3, 1].astype(np.float32))

    def test_pipeline_einsum_contrast_merge(self):
        pipeline = {
            "version": 1,
            "steps": [
                {"kind": "load", "name": "Load", "enabled": True, "params": {}},
                {"kind": "einsum", "name": "Einsum reduce", "enabled": True,
                 "params": {"axes": "czyx", "reduction": "mean"}},
                {"kind": "contrast", "name": "Contrast", "enabled": True,
                 "params": {"low": 1.0, "high": 99.0, "gamma": 1.0}},
                {"kind": "deskew", "name": "Deskew + rotate", "enabled": False, "params": {}},
                {"kind": "merge", "name": "Merge channels", "enabled": True,
                 "params": {"blend": "Additive", "colors": ["#ff0000", "#00ff00"]}},
            ],
        }
        messages = []
        out, meta = wb.run_pipeline(self.path, pipeline, progress=lambda f, m: messages.append((f, m)))
        self.assertEqual(out.shape, (3, 1, 4, 16, 24))
        self.assertTrue(meta["rgb"])
        self.assertEqual(meta["dims"]["c"], 3)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        # red = channel 0 after contrast, green = channel 1, blue = nothing
        expected = self.data.astype(np.float32).mean(axis=0)  # (z, c, y, x)
        ch0 = np.transpose(expected, (1, 0, 2, 3))[0]
        lo, hi = np.percentile(ch0, [1.0, 99.0])
        red = np.clip((ch0 - lo) / (hi - lo), 0, 1)
        np.testing.assert_allclose(out[0, 0], red, atol=1e-4)
        self.assertEqual(float(out[2].max()), 0.0)
        self.assertEqual(messages[-1][0], 1.0)

    def test_unsupported_step_raises_or_is_skipped(self):
        pipeline = [{"kind": "load", "params": {}}, {"kind": "decon", "enabled": True, "params": {}}]
        with self.assertRaises(NotImplementedError) as cm:
            wb.run_pipeline(self.path, pipeline)
        self.assertIn("decon", str(cm.exception))
        out, meta = wb.run_pipeline(self.path, pipeline, strict=False)
        self.assertEqual(meta["skipped"], ["decon"])
        self.assertEqual(out.shape, (2, 3, 4, 16, 24))

    def test_json_string_pipeline_and_crop(self):
        import json

        pipeline = json.dumps({"steps": [{"kind": "croppad", "params": {"origin": [1, 2, -2], "size": [2, 8, 10], "fill": -1}}]})
        out, meta = wb.run_pipeline(self.path, pipeline)
        self.assertEqual(out.shape, (2, 3, 2, 8, 10))
        self.assertEqual(float(out[0, 0, 0, 0, 0]), -1.0)  # padded column
        np.testing.assert_array_equal(out[0, 0, 0, 0, 2:], self.data[0, 1, 0, 2, 0:8].astype(np.float32))


def _sirius_extension():
    try:
        import sirius  # type: ignore

        return sirius if hasattr(sirius, "SimReconstructor") else None
    except Exception:  # noqa: BLE001
        return None


@unittest.skipIf(_sirius_extension() is None, "sirius extension not importable")
class TestSimStep(unittest.TestCase):
    DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

    def test_sim_step_reproduces_the_reference_reconstruction(self):
        sirius = _sirius_extension()
        raw = sirius.read_tiff(str(self.DATA / "raw.tif"), dtype=np.float32)
        expected = sirius.read_tiff(str(self.DATA / "raw_proc.tif"), dtype=np.float32)
        params = {"params_file": str(self.DATA / "config.txt"), "otf": str(self.DATA / "otf.tif")}
        r = wb.run_step("sim", params, raw, {"voxel_um": [0.08, 0.08, 0.125]}, device="cpu")
        self.assertEqual(r.array.shape, (1, 1) + expected.shape)
        rel = np.max(np.abs(r.array[0, 0] - expected)) / np.max(np.abs(expected))
        self.assertLess(rel, 1e-5)
        self.assertEqual(len(r.info["fits"]), 1)
        self.assertEqual(len(r.info["fits"][0]["k0"]), 3)
        self.assertAlmostEqual(r.meta["voxel_um"][0], 0.04, places=6)
        self.assertEqual(r.meta["dims"]["z"], 9)
        # the same parameters again reuse the reconstructor
        r2 = wb.run_step("sim", params, raw, {"voxel_um": [0.08, 0.08, 0.125]}, device="cpu")
        np.testing.assert_array_equal(r2.array, r.array)

    def test_sim_step_needs_an_otf_file(self):
        raw = np.zeros((15, 8, 8), np.float32)
        with self.assertRaises(ValueError):
            wb.run_step("sim", {"angles": 3, "phases": 5}, raw)


class TestSteps(unittest.TestCase):
    def test_reductions_keep_axes_with_length_one(self):
        a = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.float32).reshape(2, 3, 4, 5, 6)
        r = wb.run_step("einsum", {"axes": "ctyx", "reduction": "max"}, a)
        self.assertEqual(r.array.shape, (2, 3, 1, 5, 6))
        np.testing.assert_array_equal(r.array[:, :, 0], a.max(axis=2))
        r = wb.run_step("maxproj", {"axis": "z"}, a)
        self.assertEqual(r.array.shape, (2, 3, 1, 5, 6))
        r = wb.run_step("meant", {}, a)
        self.assertEqual(r.array.shape, (2, 1, 4, 5, 6))
        np.testing.assert_allclose(r.array[:, 0], a.mean(axis=1), rtol=1e-6)

    @unittest.skipUnless(_HAVE_SCIPY, "connected components need scipy")
    def test_threshold_labels_components(self):
        a = np.zeros((1, 1, 4, 10, 10), np.float32)
        a[0, 0, :, 1:3, 1:3] = 5.0
        a[0, 0, :, 6:9, 6:9] = 7.0
        r = wb.run_step("threshold", {"channel": 0, "threshold": 1.0}, a)
        self.assertIsNotNone(r.labels)
        self.assertEqual(r.labels.shape, (1, 4, 10, 10))
        self.assertEqual(int(r.labels.max()), 2)
        r2 = wb.run_step("threshold", {"channel": 0, "threshold": 1.0, "min_voxels": 20}, a)
        self.assertEqual(int(r2.labels.max()), 1)

    def test_resample_and_bleach(self):
        a = np.ones((1, 2, 4, 8, 8), np.float32)
        a[0, 1] *= 0.5
        r = wb.run_step("bleach", {}, a)
        np.testing.assert_allclose(r.array[0, 1], 1.0)
        meta = {"voxel_um": [0.1, 0.1, 0.4]}
        r = wb.run_step("resample", {"voxel": [0.2, 0.2, 0.2]}, a, meta)
        self.assertEqual(r.array.shape, (1, 2, 8, 4, 4))
        self.assertAlmostEqual(r.meta["voxel_um"][2], 0.2, places=6)

    @unittest.skipUnless(_HAVE_SCIPY, "connected components need scipy")
    def test_channel_by_name(self):
        a = np.zeros((2, 1, 2, 4, 4), np.float32)
        a[1] = 3.0
        meta = {"channels": [{"label": "DAPI", "wavelength_nm": 405}, {"label": "GFP", "wavelength_nm": 488}]}
        r = wb.run_step("threshold", {"channel": "488", "threshold": 1.0}, a, meta)
        self.assertEqual(r.info["channel"], 1)
        self.assertEqual(int(r.labels.max()), 1)


if __name__ == "__main__":
    unittest.main()
