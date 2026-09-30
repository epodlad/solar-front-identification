# Data guide

The package contains a compact subset sufficient for its documented commands. It is not a mirror of any mission archive. Observation times, units, versions and acquisition intervals are retained. [SOURCES.json](../data/SOURCES.json) gives the exact retained retrieval records and checksums. A newer archive version need not reproduce an older result byte for byte.

| Included material | Meaning and use |
| --- | --- |
| `data/E11`, `data/E05` | Prescribed complete neighboring plasma states, both normalized and dimensional. These are conditional constructions, not all measured quantities. |
| `data/fan` | Prescribed initial states, numerical starting guess and AIA sampling arrays. |
| `data/solo/raw` | The two original V03 MAG and SWA/PAS CDF files used for 2021-08-30. |
| `data/suvi` | Two original 195 Å FITS exposures for the comparison figure, the retained C/E measurements for twelve processing variants, and the image-coordinate geometry. |
| `data/imaging` | E11 retained original/difference displays and fitted crest coordinates; E05 registered exposure-normalized image grid and candidate aperture positions. |
| `data/geometry` | AIA/EUVI observer headers and digitized published model contours for the approximate ellipsoid recovery. |
| `data/eis_coverage` | Raster extents, slit times, spectral-window information and saved conditional coverage estimates. These support the coverage discussion; the release does not perform a new EIS density inversion. |

## Solar Orbiter

The calculation reads `solo_L2_mag-rtn-normal_20210830_V03.cdf` and `solo_L2_swa-pas-grnd-mom_20210830_V03.cdf`. Only 10:18–10:23 UT enters the analysis. The boundary core is 10:21:24–10:21:28 UT. Exact acquisition endpoints are derived from `Epoch` and `Half_interval` in PAS; the CDF quality fields are exported with the aligned samples.

The source is the [ESA Solar Orbiter Archive](https://soar.esac.esa.int/soar/). Archive requests can return several versions: use the named V03 files and verify the supplied hashes. The reference boundary is Event 2, current sheet CS2, in [Suen et al. (2023)](https://doi.org/10.1051/0004-6361/202345922). This package is an independent diagnostic calculation for that previously studied boundary.

Data citations: [MAG](https://doi.org/10.5270/esa-ux7y320) and [SWA](https://doi.org/10.5270/esa-ahypgn6). Instrument papers: [Horbury et al. (2020)](https://doi.org/10.1051/0004-6361/201937257) and [Owen et al. (2020)](https://doi.org/10.1051/0004-6361/201937259). Solar Orbiter is a mission of international collaboration between ESA and NASA, operated by ESA.

## SUVI

The native GOES-16 SUVI 195 Å FITS exposures are 2017-09-10 15:30:24 and 15:58:24.76 UT, version `v_0_1_0`. Each exposure is sampled using its own WCS onto a common image-coordinate grid. The stored image quantity is radiance in W m⁻² sr⁻¹. These September 2017 observations predate the operational phase; use the archive's calibration notes for this version.

The source is [NOAA/NCEI SUVI](https://www.ncei.noaa.gov/products/goes-r-solar-ultraviolet-imager-suvi). Direct archive URLs are in `SOURCES.json`. Please credit NOAA/NCEI, GOES-16 SUVI, and the instrument team, and cite [Seaton and Darnel (2018)](https://doi.org/10.3847/2041-8213/aaa28e).

The five-epoch geometric test starts from the retained feature measurements. It reproduces their relationship to the calculated fan; it does not repeat every earlier image-detection choice. The later trailing-emission analysis used a different local cut. It is not recomputed by this geometric-test script and must not be interpreted as the same aperture B.

## AIA

The AIA observations are from NASA/SDO and the AIA science team. Instrument reference: [Lemen et al. (2012)](https://doi.org/10.1007/s11207-011-9776-8).

- **E11, 2010-06-13:** the retained AIA 193 Å original and base-difference displays correspond to 05:39:20.05 UT, with reference 05:35:08.07 UT. The existing spatial smoothing and fitted inner/outer crest curves are preserved. The PNGs reproduce the article display; they are not an invertible calibrated image cube. The event was studied by Kozarev et al. (2011) and Ma et al. (2011). The database record is [E11](https://rmo-solar.org/#r123-E11).
- **E05, 2011-02-16:** the stored registered grid contains exposure-normalized AIA 193 Å values in DN s⁻¹. The plot subtracts frame 1 (14:25:32.84 UT) from frame 17 (14:28:44.84 UT). Yellow boxes show candidate placements of the EIS sampling region, not a uniquely resolved wave front. The published diagnostics are from Veronig et al. (2011). The database record is [E05](https://rmo-solar.org/#r123-E05).
- **2017-09-10:** `spherical_cut_cache.npz` contains the seven retained image grids and primary profiles. `ray_samples.npz` contains the exact 17-ray samples used for the feature search. The original images were obtained through Helioviewer in JPEG2000 format. These arrays contain display levels, not calibrated radiance. They support the stated positional comparison; the scripts do not infer density, temperature, emission measure or radiometric uncertainty from them.

The 2017 cache times are 15:51:05.84, 15:59:29.84, 16:00:41.84, 16:01:53.84, 16:03:05.85, 16:04:17.84 and 16:05:29.84 UT. The last five frames are the selected sequence. Requests can return the nearest available frame; use the actual acquisition time, not the requested filename time.

The AIA script reproduces the primary track and nine retained median-based processing choices. It also evaluates the corresponding nine mean-based choices, giving an explicitly documented set of eighteen combinations. Their range is a processing sensitivity, not a statistical confidence interval.

To start a new radiometric study, obtain the calibrated AIA observations through an appropriate science archive and repeat the processing for that purpose. The package intentionally avoids relabeling the saved display arrays as FITS measurements. [Helioviewer documentation](https://api.helioviewer.org/docs/v2/) describes its image access. No new external download is required for this release's commands.

Credit: Courtesy of NASA/SDO and the AIA, EVE, and HMI science teams. The [SDO data-use guidance](https://sdo.gsfc.nasa.gov/data/rules.php) distinguishes scientific data from browse imagery. The attribution and provenance retained here do not constitute an additional permission or endorsement from an instrument team.

## Geometry and published work

The numerical contour coordinates in `data/geometry/digitized_points.npz` were extracted from the published model curves in Figure 2 of [Hu et al. (2019)](https://doi.org/10.3847/1538-4357/ab2055). The fit reproduces an approximate graphical recovery, not the authors' original parameter table. The digitization protocol explains the retained constraints. The publisher's PDF and figure images are not redistributed. Repeating the initial digitization requires the cited figure; the numerical fit and subsequent geometry run directly from the included coordinates.

The local normals describe approximately 16:00 UT under assumed heights. They are not transferred unchanged to 16:05 UT. Observer headers support a separate STEREO-A occultation check; no matching image feature or new triangulated trajectory is supplied.

## What is and is not regenerated

`run.py all` recalculates the three single-front tests, three fans and independent checks, CDF alignment and Walén fits, the SUVI affine test, the AIA profile selection, the approximate ellipsoid fit, local normals, the projection illustration, STEREO visibility and seven figures. Starting products for image tracking are identified above. The full historical image processing, EIS spectral inversion, alternative unfinished models and later SUVI reference-image survey are outside this command. Their absence must not be mistaken for a new verification of those additional analyses.
