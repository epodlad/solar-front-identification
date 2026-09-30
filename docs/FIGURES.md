# Figure guide

Run `python run.py all`. Every PDF has a PNG companion. The numerical figure names retain the article's original file names; the displayed article figure number is one greater.

| Article figure | Output file | Script | Inputs |
| --- | --- | --- | --- |
| 1 | `figure00_observational_overview.pdf` | `plot_observations.py` | E11/E05 displays, SUVI FITS, AIA display cache, aligned MAG |
| 2 | `figure01_local_pairs.pdf` | `plot_models.py` | Recalculated full-precision single-front tests |
| 3 | `figure02_solar_orbiter.pdf` | `plot_models.py` | Recalculated CDF alignment and Walén fits |
| 4 | `figure03_three_fans.pdf` | `plot_models.py` | Recalculated wave speeds and types |
| 5 | `figure04_model_states.pdf` | `plot_models.py` | Recalculated states and continuous rarefaction samples |
| 6 | `figure05_suvi_constraint.pdf` | `plot_observations.py` | SUVI FITS and recomputed common-geometry test |
| 7 | `figure06_aia_comparison.pdf` | `plot_observations.py` | AIA display cache, retained feature track and recalculated model positions |

Model labels on solar images mark the association being tested. They are not independent observational classifications. Red/blue differences show increases/decreases in the stated image quantity. An intensity decrease alone does not establish a density decrease or a rarefaction.

PDF creator metadata identifies the actual Python/Matplotlib version. Draft notes, personal paths and creation timestamps are not placed in the figures. Instrument headers and data acquisition metadata remain in the data files because they are needed for reproduction.
