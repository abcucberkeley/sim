---
title: Operation plugin API and help pages
figure: An operation plugin: code, parameter specs, help page
---

Every processing step in SIRIUS is an *operation*: a C++ class that declares its parameters and turns an input array into an output array. The parameters dock, the pipeline file, the Python export and the assistant's tools are all generated from the declaration, so a new operation is one source file plus one help page.

$$
\text{Operation}: (\text{input},\; \text{params}) \rightarrow (\text{output},\; \text{diagnostics})
$$

## Writing an operation

An operation implements `sirius::app::Operation` (see `app/core/operation.hpp`):

- `info()` returns the `OpInfo`: kind, display name, menu group, the list of `ParamSpec`s, the diagnostics kind and flags such as `separableOverT` or `hasGpuPath`.
- `summary(params, inputMeta)` is the one-line description shown in the ops row.
- `validate(params, inputMeta)` reports errors (the step cannot run) and warnings (it can, but the user should know).
- `outputMeta(params, inputMeta)` predicts the output shape without running.
- `run(input, params, ctx)` does the work on the worker thread, reporting progress through `ctx.report()` and checking `ctx.throwIfCancelled()`.

Register the factory in `app/core/ops/builtin_list.cpp` and the operation appears in the *Add a processing step* menu under its group.

## Help pages

Pages are Markdown files with LaTeX and images stored next to the operation code in `app/help/<kind>.md`, installed beside the executable so that anyone on the team can edit them (`SIRIUS_HELP_DIR` overrides the location). The reader (`parseHelpMarkdown`) understands this layout:

- A front matter block delimited by `---` lines with `title:`, `figure:` (caption of the figure slot) and optional `figure_path:` (image file beside the page).
- The first paragraph after the front matter is the introduction.
- The first `$$ … $$` block is the display formula.
- A `## Parameters` section holding a two-column table. The first cell is `**Name** <br> range`, the second the explanation; an inline `$…$` at the end of the explanation is rendered as the parameter's formula.
- An optional `## Note` section is the footer note.

Any other section is rendered as ordinary Markdown below the parameter table. Inline math uses `$…$`; the supported LaTeX subset covers fractions, sub- and superscripts, Greek letters, `\mathbf`, `\tilde`, `\hat`, `\sum`, `\prod`, `\text`, `\left … \right`, `\cdot`, `\times`, `\in`, `\mid`, `\ast`, `\star`, `\nabla` and `\rightarrow`.

## Parameters

| Parameter | Explanation |
|---|---|
| **kind** <br> identifier | Short lower-case identifier of the operation, also the name of its help page and of its entry in pipeline files. |
| **group** <br> Reconstruct · Reduce · Intensity · Geometry · Combine · Segment | Where the operation appears in the *Add a processing step* menu. |
| **params** <br> ParamSpec list | Typed parameter declarations: key, label, type, default, range, choices, unit and help text. $\text{ParamSet} = \{(\text{key}, \text{value})\}$ |

## Note

Drop a PNG or SVG onto a help window to add a figure: the image is copied next to the page and referenced from it.
