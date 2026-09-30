# Solar Front Identification

**From plasma states to a single-front test and a connected MHD wave system.**


This package accompanies *Identifying Solar and Heliospheric Fronts With Conservation Laws and the MHD Riemann Problem*, by Olena Podladchikova. It contains the inputs, Python calculations, numerical checks and plotting scripts used for the article. It can be used independently of the Riemann Map Operator website.

Start with the three complete state pairs. The code checks mass, momentum, magnetic induction and energy, then uses entropy and characteristic speeds to distinguish the retained fast and slow shocks. A second calculation builds three connected Riemann fans and predicts the positions of their waves. Solar Orbiter, SUVI and AIA examples show how measurements can test these connections.

## Quick start

Use Python 3.11 or 3.12 in a new environment. From the extracted package directory:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py verify-files
python run.py all
```

The calculations run locally. No account, API key or observation download is needed for this supplied subset. Installing the Python dependencies requires internet access. Results are written to `results/`; the input data are left unchanged. A repeated run replaces files in the selected results directory. To keep a run, give it a separate directory:

```bash
python run.py all --output my_results
```

**Open `results/SUMMARY.md` first.** It lists the outcomes, main numerical values and links to the figures.

The seven figures are saved as PDF and PNG in `results/figures/`. The command stops if a calculation fails; its log is kept in the same results directory. All scientific figures are produced with Python and Matplotlib. Solar images remain observational raster data within the plots.

## What to read first

| Question | Start here |
| --- | --- |
| What do LEFT and RIGHT mean? | [Method and conventions](docs/METHOD.md) |
| Which data are measured and which are prescribed? | [Data guide](docs/DATA.md) |
| Which result should I obtain? | [Expected results](docs/RESULTS.md) |
| Which script makes a figure? | [Figure guide](docs/FIGURES.md) |
| How can I use my own state pair? | [Single-front input example](docs/INPUTS.md) |

## Run one part

```bash
python run.py single-fronts
python run.py fans
python run.py solar-orbiter
python run.py suvi
python run.py aia
python run.py geometry
python run.py figures
```

`suvi` and `geometry` use the generated P100 fan, so run `fans` first. `figures` uses the generated checks, not copied reference answers; run `checks` or `all` first. `checks` runs every calculation without plotting.

## Package contents

- `data/`: full-precision model inputs and the compact observation subset.
- `scripts/`: calculations, independent checks and figure generation.
- `vendor/rmo/`: the unchanged numerical solver snapshot used for the fans.
- `reference_results/`: retained numerical results for comparison.
- `figures/`: the seven figures regenerated and checked for this release.
- `docs/`: methods, data sources, units and examples.
- `MANIFEST.json`: SHA-256 checksums of the packaged files.
- `CITATION.cff` and `.zenodo.json`: citation and archive metadata.

The solver is included so that the calculation does not depend on a changing website or a separate RMO installation. Its source attribution and license are preserved in `vendor/SOURCE.json` and `NOTICE`.

## Scientific scope

E11 and E05 are conditional complete plasma-state pairs. They are not unique reconstructions of their solar events. P080, P100 and P120 follow one regular solution branch; numerical convergence does not establish uniqueness. The Solar Orbiter test supports an Alfvénic relation and retains the sampling limitation of the short boundary. The SUVI association fails its stated geometric test. The AIA tracks do not identify a contact or rarefaction.

These distinctions make the package useful: a valid model supplies explicit conditions and predicted positions that further observations can test.

## Citation and reuse

Please cite the associated article when available, this software release, and the original studies and instrument teams listed in [DATA.md](docs/DATA.md). No article DOI or acceptance status is assigned here.

The code is distributed under Apache-2.0. Observational data retain their original terms and acknowledgements, including the Solar Orbiter data license. See [NOTICE](NOTICE); the code license does not relicense observations.
