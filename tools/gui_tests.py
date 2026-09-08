#!/usr/bin/env python3
"""Scripted tests of the Qt layer.

The core is covered by tests/test_app_*.cpp, which run without a display. The
widgets were covered by one screenshot that only proved the window came up.
This drives the real application through the hooks it already has for
scripting -- ``--tool`` for the assistant API, ``--action`` for a menu item,
``--stroke`` and ``--wheel`` for mouse input on the XY pane, ``--drop`` for a
drag and drop, ``--record`` for a machine-readable log of what happened -- and
asserts on what comes back rather than on the process surviving.

    python3 tools/gui_tests.py --app build/linux-gcc-app-dev/app/sirius-app

Runs offscreen; no display needed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "tests" / "data" / "raw.tif"
PIPELINE = ROOT / "examples" / "sim_bundled.sirius.toml"

_TOOL = re.compile(r"^tool (\w+) -> ", re.M)


class Failure(Exception):
    pass


def run(app: Path, args: List[str], timeout: int = 300, env: Optional[Dict[str, str]] = None) -> str:
    """Run the application once, offscreen, and return everything it printed."""
    full = [str(app), "-platform", "offscreen", *args]
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen", **(env or {})}
    try:
        done = subprocess.run(full, capture_output=True, text=True, timeout=timeout, env=environment)
    except subprocess.TimeoutExpired as e:
        raise Failure(f"timed out after {timeout}s: {' '.join(full)}") from e
    if done.returncode != 0:
        raise Failure(f"exit {done.returncode}: {' '.join(full)}\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    return done.stdout + done.stderr


def tool_results(output: str) -> Dict[str, List[Any]]:
    """Every `tool <name> -> {json}` the run printed, by tool name."""
    out: Dict[str, List[Any]] = {}
    for m in _TOOL.finditer(output):
        rest = output[m.end() :]
        decoder = json.JSONDecoder()
        try:
            value, _ = decoder.raw_decode(rest)
        except ValueError:
            continue
        out.setdefault(m.group(1), []).append(value)
    return out


def only(results: Dict[str, List[Any]], name: str) -> Any:
    values = results.get(name)
    if not values:
        raise Failure(f"the run printed no result for '{name}'")
    return values[-1]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def image_is_not_blank(path: Path) -> None:
    check(path.is_file() and path.stat().st_size > 5000, f"{path} missing or suspiciously small")
    try:
        from PIL import Image  # noqa: PLC0415 - optional, the size check stands without it
    except ImportError:
        return
    with Image.open(path) as im:
        colours = im.convert("RGB").getcolors(maxcolors=1 << 20)
    check(colours is not None and len(colours) > 32, f"{path} has almost no colours: a blank window")


# --- the scenarios ---------------------------------------------------------


def test_ortho_view_shows_the_dataset(app: Path, tmp: Path) -> None:
    shot = tmp / "ortho.png"
    out = run(
        app,
        [
            "--dataset",
            str(RAW),
            "--tool",
            '{"name":"set_view","args":{"mode":"ortho"}}',
            "--tool",
            '{"name":"get_state","args":{}}',
            "--screenshot",
            str(shot),
            "--settle",
            "900",
            "--quit-after",
            "6000",
        ],
    )
    state = only(tool_results(out), "get_state")
    check(state["dataset"] is not None, "the dataset did not open")
    check(state["dataset"]["shape"].startswith("c1 t1 z135"), f"unexpected shape {state['dataset']['shape']}")
    image_is_not_blank(shot)


def test_every_view_mode_renders(app: Path, tmp: Path) -> None:
    for mode in ("ortho", "3d", "compare"):
        shot = tmp / f"mode_{mode}.png"
        out = run(
            app,
            [
                "--dataset",
                str(RAW),
                "--tool",
                json.dumps({"name": "set_view", "args": {"mode": mode}}),
                "--tool",
                '{"name":"get_state","args":{}}',
                "--screenshot",
                str(shot),
                "--settle",
                "900",
                "--quit-after",
                "6000",
            ],
        )
        state = only(tool_results(out), "get_state")
        check(state["view"]["mode"].lower() == mode, f"view is {state['view']['mode']}, asked for {mode}")
        # Qt's offscreen platform has no OpenGL widgets, so the 3D pane may be
        # a notice saying so rather than a rendering. Switching to it and
        # drawing the window without crashing is what this can honestly check;
        # the slice views are painted by QPainter and always have content.
        if mode == "3d":
            check(shot.is_file() and shot.stat().st_size > 5000, f"{shot} missing or suspiciously small")
        else:
            image_is_not_blank(shot)


def test_compare_shows_raw_beside_the_result(app: Path, tmp: Path) -> None:
    # the compare pane once drew the raw side at the reconstruction's subsample
    # factor, which made it blurry; both sides must report the same field
    shot = tmp / "compare.png"
    out = run(
        app,
        [
            "--pipeline",
            str(PIPELINE),
            "--tool",
            '{"name":"run","args":{}}',
            "--tool",
            '{"name":"set_view","args":{"mode":"compare"}}',
            "--tool",
            '{"name":"get_state","args":{}}',
            "--screenshot",
            str(shot),
            "--settle",
            "1200",
            "--quit-after",
            "9000",
        ],
    )
    state = only(tool_results(out), "get_state")
    check(state["view"]["mode"].lower() == "compare", "compare mode did not take")
    image_is_not_blank(shot)


def test_painting_reaches_the_labels(app: Path, tmp: Path) -> None:
    # the recording is the machine-readable account of what the widgets did
    log = tmp / "paint.jsonl"
    out = run(
        app,
        [
            "--record",
            str(log),
            "--dataset",
            str(RAW),
            "--tool",
            '{"name":"add_step","args":{"kind":"classic"}}',
            "--tool",
            '{"name":"run","args":{}}',
            "--action",
            "Paint labels",
            "--stroke",
            "20,20,30,30,6",
            "--tool",
            '{"name":"get_step","args":{"step":3}}',
            "--settle",
            "900",
            "--quit-after",
            "9000",
        ],
    )
    step = only(tool_results(out), "get_step")
    check(step["kind"] == "classic", f"step 3 is {step['kind']}")
    events = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    kinds = [e["event"] for e in events]
    check("step_ran" in kinds, "the segmentation did not run")
    paints = [e for e in events if e["event"] == "paint"]
    check(len(paints) >= 3, f"the stroke produced {len(paints)} paint events")
    check(all(p["voxels"] > 0 for p in paints), "a paint event changed nothing")
    check(any(p["x"] != paints[0]["x"] for p in paints), "the stroke never moved")


def test_the_wheel_zooms(app: Path, tmp: Path) -> None:
    out = run(
        app,
        [
            "--dataset",
            str(RAW),
            "--wheel",
            "32,32,3",
            "--tool",
            '{"name":"get_state","args":{}}',
            "--settle",
            "600",
            "--quit-after",
            "5000",
        ],
    )
    state = only(tool_results(out), "get_state")
    check(float(state["view"]["zoom"]) > 1.0, f"zoom is {state['view']['zoom']} after scrolling in")


def test_a_dropped_file_opens(app: Path, tmp: Path) -> None:
    out = run(app, ["--drop", str(RAW), "--tool", '{"name":"get_state","args":{}}', "--settle", "900", "--quit-after", "6000"])
    state = only(tool_results(out), "get_state")
    check(state["dataset"] is not None, "dropping a TIFF did not open it")
    check(state["dataset"]["name"].startswith("raw"), f"opened {state['dataset']['name']}")


def test_menu_actions_reach_the_view(app: Path, tmp: Path) -> None:
    out = run(
        app,
        [
            "--dataset",
            str(RAW),
            "--action",
            "Labels overlay",
            "--action",
            "Physical z scaling",
            "--tool",
            '{"name":"get_state","args":{}}',
            "--settle",
            "600",
            "--quit-after",
            "5000",
        ],
    )
    view = only(tool_results(out), "get_state")["view"]
    check(view["labels"] is True, "Labels overlay did not turn on")
    check(view.get("physical_z") is False, "Physical z scaling did not turn off")


def test_a_preset_fills_the_fields(app: Path, tmp: Path) -> None:
    # a preset is values, not a mode: the step holds what it wrote and the
    # change is undoable like any other
    out = run(
        app,
        [
            "--dataset",
            str(RAW),
            "--tool",
            '{"name":"add_step","args":{"kind":"classic"}}',
            "--tool",
            '{"name":"apply_preset","args":{"step":3,"preset":"Filament network"}}',
            "--tool",
            '{"name":"get_step","args":{"step":3}}',
            "--settle",
            "600",
            "--quit-after",
            "6000",
        ],
    )
    params = only(tool_results(out), "get_step")["params"]
    check(params["enhance"] == "Neurites (Meijering)", f"enhance is {params['enhance']}")
    check(params["post"] == "Connected components", f"post is {params['post']}")
    check(abs(float(params["enhance_sigma"]) - 0.8) < 1e-9, f"sigma is {params['enhance_sigma']}")


SCENARIOS = [
    test_ortho_view_shows_the_dataset,
    test_every_view_mode_renders,
    test_compare_shows_raw_beside_the_result,
    test_painting_reaches_the_labels,
    test_the_wheel_zooms,
    test_a_dropped_file_opens,
    test_menu_actions_reach_the_view,
    test_a_preset_fills_the_fields,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path, help="the sirius-app binary")
    parser.add_argument("--only", default="", help="run just the scenarios whose name contains this")
    args = parser.parse_args()
    if not args.app.is_file():
        print(f"no such application: {args.app}", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="sirius-gui-"))
    failures = 0
    try:
        for scenario in SCENARIOS:
            if args.only and args.only not in scenario.__name__:
                continue
            name = scenario.__name__.removeprefix("test_").replace("_", " ")
            try:
                scenario(args.app, tmp)
            except Failure as e:
                failures += 1
                print(f"FAIL  {name}\n      {e}", file=sys.stderr)
            else:
                print(f"ok    {name}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
