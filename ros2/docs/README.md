# Deliverable chapter: the ROS 2 module

LaTeX documentation of the `physics_amm` ROS 2 bundle, written as a
self-contained chapter for a CoreSense deliverable.

## Layout

- `ros2-chapter/` — **the hand-over unit**: chapter body, additive preamble
  snippet, bibliography entries, TikZ figures, and a snapshot of the
  `LearnModel.action` interface file. Everything a deliverable master needs.
- `standalone.tex` + `Makefile` — a local compile harness that approximates
  the CoreSense deliverable look (sans-serif, teal accents). Not part of the
  hand-over; it exists so the chapter can be built and reviewed on its own.

## Building the standalone preview

Requires TeX Live with `latexmk` (packages: tikz, listings, booktabs,
algorithm2e, hyperref, microtype).

```bash
make          # -> standalone.pdf (chapter + bibliography, ~11 pages)
make clean
```

## Integrating into the deliverable (for the document editor)

1. Copy the `ros2-chapter/` folder next to your master `.tex` file.
2. In the master **preamble**, add:
   ```latex
   \input{ros2-chapter/chapter-preamble}
   ```
   The snippet is strictly additive: it defines `ros2`-prefixed colors,
   two `listings` styles, and TikZ styles — nothing global is redefined.
   It requires `xcolor`, `listings`, `booktabs`, `algorithm2e`, and `tikz`
   with libraries `positioning, arrows.meta, fit, backgrounds, calc`.
3. In the master **body**, where the chapter belongs, add:
   ```latex
   \input{ros2-chapter/chapter}
   ```
4. Merge `ros2-chapter/refs.bib` into your bibliography (add it to the
   `\bibliography{...}` list, or import the entries into your database).
   All keys are `ros2:`-prefixed, so they cannot clash with yours.

The chapter body uses only standard commands (no custom macros), all its
labels are `ros2:`-prefixed, and all internal paths are relative to the
master document — it drops into any `report`/`book`-class master unchanged
and numbers itself from the surrounding document.

## Maintenance notes

- `ros2-chapter/listings/LearnModel.action` is a **snapshot copy** of
  `../physics_amm_msgs/action/LearnModel.action` (with the one non-ASCII
  dash replaced for `listings` compatibility). If the interface changes,
  refresh the copy.
- Tables in the chapter transcribe the interface files and
  `physics_amm/config/params.yaml` — those files are authoritative; update
  the tables when they change.
