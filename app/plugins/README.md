# User operations (plugins)

A plugin is one Python file that adds a processing step to the workbench.
The application discovers plugins through its Python worker at start-up and
on **Process ▸ Reload plugins**, and gives each one a parameter form, an
entry in the "Add a processing step" menu, undo, caching, assistant tools and
a help page, all generated from the file. The same file runs unchanged on
the HPC backend, because the worker on the cluster loads it too.

Directories searched, in order:

1. `$SIRIUS_PLUGIN_DIRS` (several directories separated by `:`),
2. `~/.sirius/plugins`,
3. `plugins/` next to the application (this directory in a checkout).

## File format

```python
STEP = {
    "kind": "dog_filter",                 # identifier, unique, saved in pipeline files
    "name": "Difference of Gaussians",    # shown in the menu and the step list
    "group": "Intensity",                 # menu group (Reconstruct, Reduce, Intensity, Geometry, Combine, Segment, or your own)
    "params": [
        {"key": "sigma_lo", "label": "σ low", "type": "double", "default": 1.0, "min": 0.1, "max": 50,
         "unit": "px", "help": "…"},
    ],
    "separable_over_t": True,   # optional: the worker receives one time point at a time
    "produces_labels": False,   # optional: run returns instance labels too
    "needs_labels": False,      # optional: ctx.labels carries the input's labels
    "help": "…",                # optional Markdown + LaTeX; run.__doc__ otherwise
}

def run(data, params, meta, ctx):
    """# Title      <- this docstring is the help page (Markdown, $LaTeX$)"""
    ...
    return output
```

- `data` is a float32 array `(c, t, z, y, x)`; `meta` is a dict with `dims`,
  `voxel_um` (x, y, z), `channels` (label, wavelength_nm, color), `rgb`.
- `params` holds every parameter (defaults filled in).
- `ctx.progress(fraction, message="")` drives the progress bar,
  `ctx.cancelled()` should be polled in long loops, `ctx.log(...)` writes to
  the worker log, `ctx.labels` is the input's `uint32 (t, z, y, x)` label
  volume when `needs_labels` is set.
- Return the output array (lower ranks are expanded: `(z, y, x)` becomes
  `(1, 1, z, y, x)`), or a tuple `(output, labels)`, `(output, diagnostics)`,
  `(output, labels, diagnostics)`, or a dict with `output`, `labels`,
  `diagnostics` and `meta` (overrides: `voxel_um`, `channels`, `rgb`).

Parameter types: `double`, `int`, `bool`, `choice` (with `choices`), `path`
(with an optional file `filter`), `string`, `channel` (index into the
input's channels), `axes` (subset of `ctzyx`), `double_list`, `string_list`;
optional `min`, `max`, `step`, `decimals`, `unit`, `help`, `advanced`.

Diagnostics is a dict with any of `summary`, `facts` (`{name: value}`),
`warnings`, `footer`, `table` (`{"caption", "header", "rows"}`) and `images`
(`[{"title", "meta", "data": 2-D array, "log": bool}]`); they appear in the
diagnostics dock and are readable by the assistant.

A file that fails to import is listed with its error in the log; a `kind`
that a built-in operation already uses is refused. Files starting with `_`
are ignored. See `dog_filter.py` for a complete example.
