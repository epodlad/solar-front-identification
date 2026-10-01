# Version 1.1.0 — 1 October 2026

- Adds four observed-image movies (E11, E05, SUVI, AIA G1), their captions, and retained additional E11/SUVI inputs.
- Adds `python run.py animations`; `python run.py all` now includes the movies and links them in `results/SUMMARY.md`.
- Uses the same fixed display scales, feature tracks, and conditional P100 annotations throughout each sequence. No interpolated observations or new fits.
- Reuses the original E05/AIA arrays and all numerical reference results.
- Preserves the numerical solver, conservation checks, model states, measured input tables, and seven article figures from version 1.0.0.
- Updates the reading bibliography and movie provenance documentation.

# Version 1.0.0

First standalone package for Solar Front Identification.

- Full-precision E11/E05 inputs and explicit algebraic conservation tests.
- Three regular ideal-MHD fan constructions and a separate SI verifier.
- Original Solar Orbiter CDFs, acquisition-aware alignment and interval-sensitive Walén fits.
- Retained SUVI feature measurements, AIA image/ray subsets and geometric checks.
- Seven reproducible Python/Matplotlib figures.
- English setup, methods, data, input and result guides.

The calculations have been separated from historical working directories. The numerical solver snapshot and the prescribed states are preserved. The first characteristic-speed figure now takes values directly from the full-precision checks; its former plotting constants were rounded. The primary Walén annotation is displayed to two decimal places. Figure creator metadata identifies Python/Matplotlib.

This release does not assert an accepted article, assign an article DOI, or change any original event study's priority.
