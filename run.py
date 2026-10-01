#!/usr/bin/env python3
"""Run the article calculations and generate the seven figures and four supporting movies."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        choices=[
            "all",
            "checks",
            "figures",
            "animations",
            "single-fronts",
            "fans",
            "solar-orbiter",
            "suvi",
            "aia",
            "geometry",
            "verify-files",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results",
        help="Generated results directory (default: results/)",
    )
    args = parser.parse_args()
    out = args.output.resolve()
    if (
        out == ROOT
        or out.is_relative_to(ROOT / "data")
        or out.is_relative_to(ROOT / "reference_results")
    ):
        parser.error("Use a separate output directory, such as results/.")
    if args.task == "verify-files":
        records = json.loads((ROOT / "MANIFEST.json").read_text())
        bad = [
            path
            for path, sha in records.items()
            if not (ROOT / path).is_file()
            or hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha
        ]
        if bad:
            raise SystemExit("Checksum mismatch: " + ", ".join(bad))
        print(f"All {len(records)} packaged files match their checksums.")
        return
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        FRONT_OUTPUT=str(out),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(out / ".matplotlib"),
    )

    def run(name, *extra):
        print(f"Running {name} ...", flush=True)
        log = out / (Path(name).stem + ".log")
        with log.open("w") as stream:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / name), *map(str, extra)],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            print(log.read_text()[-6000:])
            raise SystemExit(f"{name} failed. Details: {log}")

    tasks = {
        "all": {
            "single-fronts",
            "fans",
            "solar-orbiter",
            "suvi",
            "aia",
            "geometry",
            "figures",
            "animations",
        },
        "checks": {"single-fronts", "fans", "solar-orbiter", "suvi", "aia", "geometry"},
    }.get(args.task, {args.task})
    if "single-fronts" in tasks:
        run("check_single_fronts.py")
    if "fans" in tasks:
        (out / "fan/inputs/reference_model").mkdir(parents=True, exist_ok=True)
        for rel in [
            "MODEL_PROTOCOL.json",
            "inputs/reference_model/left_right.json",
            "inputs/reference_states.csv",
            "inputs/display_feature_primary.csv",
        ]:
            shutil.copy2(ROOT / "data/fan" / rel, out / "fan" / rel)
        run("solve_fans.py")
        run("check_fans.py")
    if "solar-orbiter" in tasks:
        run(
            "check_solar_orbiter.py",
            "--source",
            ROOT / "data/solo/raw",
            "--out",
            out / "solo",
        )
    if "suvi" in tasks:
        run("check_suvi.py")
    if "aia" in tasks:
        run("check_aia.py")
    if "geometry" in tasks:
        run("fit_published_geometry.py")
        run("check_geometry.py")
    if "figures" in tasks:
        required = [
            "single_fronts/checks.json",
            "fan/independent_model_checks.json",
            "solo/checks.json",
            "suvi/primary_positions.csv",
        ]
        if any(not (out / p).exists() for p in required):
            raise SystemExit(
                "Run python run.py checks before generating figures, using the same --output directory."
            )
        run("plot_models.py")
        run("plot_observations.py")
    if "animations" in tasks:
        run("plot_animations.py")
    if args.task in {"all", "checks", "figures"}:
        run("write_summary.py")
    versions = {
        name: importlib.metadata.version(name)
        for name in [
            "numpy",
            "scipy",
            "pandas",
            "matplotlib",
            "astropy",
            "cdflib",
            "Pillow",
        ]
    }
    (out / "environment.json").write_text(
        json.dumps({"python": sys.version.split()[0], **versions}, indent=2) + "\n"
    )
    print(f"Finished. Results: {out}")
    if "animations" in tasks:
        print(f"Movies: {out / 'movies'}")
    if (out / "SUMMARY.md").exists():
        print(f"Open {out / 'SUMMARY.md'} for the results and figure links.")


if __name__ == "__main__":
    main()
