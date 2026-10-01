# Supporting movies: solar-front identification

Four animations accompany the manuscript *Identifying Solar and Heliospheric Fronts With Conservation Laws and the MHD Riemann Problem*, by Olena Podladchikova.

Open the MP4 or GIF files in `movies/` to watch them. Each movie shows observed images at the retained observation times. Playback is accelerated; the timestamps on the panels give the observation times. Repeated video frames hold each exposure on screen. No intermediate solar image is synthesized.

## Contents

| Movie | Observations | Time range (UT) | Selected exposures |
|---|---|---|---|
| S1 | AIA 193 Å, E11, 13 June 2010 | 05:38:08.07–05:41:08.07 | 16 |
| S2 | AIA 193 Å, E05, 16 February 2011 | 14:25:32.84–14:31:08.84 | 29 |
| S3 | SUVI 195 Å, 10 September 2017 | 15:54:24.75–15:58:24.76 | 5 |
| S4 | AIA 193 Å, G1, 10 September 2017 | 16:00:41.84–16:05:29.84 | 5 |

The full captions are in `docs/Supporting_Information.pdf`.

## Regenerate the movies

From the repository root, install `requirements.txt`, then run `python run.py animations`.
`python run.py all` includes the scientific calculations, figures, and movies.
Results are written under `results/movies/`, or under the directory selected with `--output`.
FFmpeg is optional: when installed and on the executable search path it adds MP4 exports;
otherwise GIF, PNG, and PDF outputs are still created.
Movie timestamps show the observation times, not playback time.

## Inputs and interpretation

- **S1:** Retained E11 brightness and base-difference displays, already spatially smoothed with a Gaussian standard deviation of 1.20 arcsec. The reference is 05:35:08.07 UT. The fixed difference scale is ±12 DN/s. Crest curves are drawn only where a retained fit exists: the inner curve in frames 2–15 and the outer curve in frames 4–12 of the original sequence. No curve is extrapolated into other frames. These are brightness crests, not wave classifications. The corresponding article-frame input at 05:39:20.05 UT is unchanged.
- **S2:** Registered, exposure-normalized AIA images. The reference is 14:25:32.84 UT. Yellow boxes preserve sixteen alternative EIS sampling positions. Times are reconstructed from the retained time offsets using the article frame at 14:28:44.84 UT as the anchor; printed times are rounded. These are sampling alternatives, not a fitted front.
- **S3:** The retained SUVI radiance grid and its 15:30:24.69 UT reference. Five actual exposures are shown; their spacings are approximately 70, 50, 50, and 70 s. C and E set the common position and scale of the P100 model at each time. Purple shading gives its linked fast-rarefaction interval, and the dotted yellow outline is aperture B. The input coordinates reproduce the existing geometric test. They do not identify an observed complete wave fan. No extra spatial or temporal smoothing is applied here.
- **S4:** The five selected AIA display grids and their preceding frames. Running differences use displayed brightness, not calibrated radiance. The earliest retained image remains in the input cache but is not a running-difference reference. G1 is an assumed projected great-circle path. Colored marks reproduce the fixed P100 predictions; the circle is the previously selected image maximum. The first fast-shock position alone was aligned in the original comparison. No later position is adjusted here.

Each movie uses fixed color limits throughout its own sequence. These limits differ between instruments and dates and cannot be used for cross-event radiometric comparisons. All subtraction is performed in floating point. Model and measured coordinates are preserved.

`data/animations/SOURCES.json` describes the added inputs. `MANIFEST.json` records checksums of all packaged files. The unchanged numerical baseline is version 1.0.0:

- Code: https://github.com/epodlad/solar-front-identification
- Archive: https://doi.org/10.5281/zenodo.23072065

Version 1.1.0 adds these animations to the reproduction package. The original version 1.0.0 remains separately archived. Existing E05 and AIA image grids and the retained model annotations are reused from their original package paths; only the E11 sequence and SUVI radiance cube are added in `data/animations/`.

AIA data are courtesy of NASA/SDO and the AIA science team. SUVI data are provided by NOAA/NCEI. The observational interpretations are credited in the article and supporting captions. Original observational data-use terms continue to apply. The animation script is provided under the Apache License 2.0 in `LICENSE`.
