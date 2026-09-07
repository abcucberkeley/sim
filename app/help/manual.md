---
title: SIRIUS manual
figure: One window: operations, viewer, parameters, diagnostics
---

SIRIUS processes multi-dimensional microscopy data (channels × time × z × y × x) with an ordered, freely reorderable stack of optional operations. The viewer shows the output of any step; the parameters dock edits the selected step; the diagnostics area explains what a step did. Steps run top to bottom, a skipped step passes its data through unchanged and every edit is undoable.

$$
\text{Load} \rightarrow \text{step}_1 \rightarrow \cdots \rightarrow \text{step}_N
$$

## The window

- **Operations** (left): the pipeline. The checkbox enables or skips a step; ▲▼ reorder; ◉ shows a step's output in the viewer. *Load* is pinned first and cannot be disabled. *Add a processing step* opens the grouped library.
- **Viewer** (centre): *Ortho* shows XY with YZ, XZ and a z projection; *3D* ray-casts the volume; *Compare* puts the raw data next to the viewed step. The tool strip selects Navigate, Probe, Measure, ROI or Paint; the crosshair only moves in Probe mode.
- **Parameters** (right): the selected step's parameters, the backend (CUDA, CPU, HPC) and the cache policy (memory, disk, recompute), with *Run step*, *View* and *Remove*.
- **Diagnostics** (bottom): per-kind panels — spectra and fitted pattern vectors for SIM, convergence for deconvolution, histograms for contrast, the label review table for segmentation, alignment statistics for stitching and registration.
- **Assistant** (✦): drives the same operations through a typed tool API; every action lands in the undo stack and is shown as a card.

## Running

*Run all enabled* (⌘R) runs every enabled step; *Run step* runs the selected one and whatever it depends on. Outputs are cached per step according to the cache policy; changing a parameter invalidates exactly the steps downstream of it. The status bar shows the progress and the memory the caches hold.

## Backends

- **CUDA** runs steps with a GPU path (SIM reconstruction, FFTs, TIFF decoding) on the selected device; the others run on the CPU.
- **CPU** runs everything on the host with OpenMP.
- **HPC** sends steps the Python worker implements to a worker started under Slurm (see *python/slurm*); the connection is configured in Preferences.

## Parameters

| Parameter | Explanation |
|---|---|
| **Pipeline files** <br> .sirius.toml | *File ▸ Save pipeline* writes every step with its parameters; *Load pipeline preset* restores one onto the current dataset. |
| **Export** <br> TIFF · zarr | *Export result…* writes any step's output with full control over the container: strips or tiles, compression, pyramid levels, chunk shape, pixel type and scaling. |
| **Folder datasets** <br> sirius-dataset.toml | *File ▸ Open folder as dataset…* opens one file per channel, tile or time point as a single dataset: a regular expression parses the names once, the result is saved beside the files and reused. The viewer's tile chooser and the Load step's *Tile* pick the tile; *Stitch* fuses all of them. |
| **Models** <br> Segment menu | *Segment ▸ Download model…* fetches segmentation models from Hugging Face into the local model store and points a Torch segmentation step at them. |
| **User operations** <br> Window menu | *Window ▸ User operations…* (also the link at the foot of the add menu) lists the Python files that define user steps, shows load errors, and edits or creates them in place; saving reloads the step. |
| **Layout** <br> Window menu | Docks can be moved, floated (also to another monitor) and reset; the layout is saved between sessions. |

## Note

Press F1 on any step for its help page, or ⌘/ for the keyboard shortcuts.
