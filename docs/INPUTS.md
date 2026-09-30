# Reading and changing the inputs

## A complete neighboring state pair

Open `data/E11/E11_F1_normalized_input.json`. The essential fields are:

```json
{
  "gamma": 1.6666666666666667,
  "front_speed": 0.0,
  "left": {"rho": 1.0, "p": 0.05969671755027295,
           "u": [1.0, 0.0, 0.0], "B": [0.5, 0.4484735890395955, 0.0]},
  "right": {"rho": 1.5625, "p": 0.14835167793452875,
            "u": [0.64, 0.20698781032596716, 0.0],
            "B": [0.5, 0.8624492096915298, 0.0]}
}
```

The order of each vector is normal, first tangent, second tangent. Density, velocity, pressure and magnetic field are normalized by rho0, U0, rho0 U0² and sqrt(mu0 rho0 U0²). The normalized permeability is one. Companion `*_SI_states.json` files give the dimensional states and frame conventions. Use these complete values rather than copying rounded article tables.

For your own example, copy the file and the `check_single_fronts.py` driver into a separate study. Supply your normal, front speed and common basis, and record which quantities are measured or assumed. Update the case list, dimensional velocity scale and expected family. The retained driver deliberately tests the three documented examples; it does not silently accept an arbitrary state pair as a shock.

## The three fan inputs

`data/fan/inputs/reference_model/left_right.json` supplies the two initial states and dimensional normalization. `reference_states.csv` supplies only the numerical starting guess. `data/fan/MODEL_PROTOCOL.json` describes the pressure variations and acceptance thresholds.

Changing the starting states can leave the retained branch or encounter a degeneracy. A failure to converge is not evidence that no physical solution exists. Any new solution needs the independent checks again.

## Observations

Do not replace the supplied observations with a different calibration or interval while retaining the old scientific conclusions. Treat that change as a new run, preserve the acquisition metadata, and compare the results explicitly. `--output` provides a convenient way to keep separate runs.
