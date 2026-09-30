# Method and conventions

## One front

A complete input state supplies mass density, thermal pressure, three velocity components and three magnetic-field components. Both sides use one orthonormal normal/tangential basis, the same units and the same reference frame. `front_speed` is the front velocity along the chosen normal.

LEFT and RIGHT denote spatial position. The signed mass flux determines which side is upstream. The normal velocity in the front rest frame is `u_n - front_speed`. This distinction matters: E11 has LEFT upstream; both E05 examples have RIGHT upstream.

`check_single_fronts.py` independently evaluates seven conserved fluxes, the continuity of the normal magnetic field, positive density and pressure, compression, entropy increase, and the fast/slow characteristic inequalities. Its equations are explicit and do not call the Riemann solver. The supplied pairs are regular compressive shocks. The script is not a general classifier for intermediate shocks, contacts, rotational discontinuities or degenerate states.

## A connected fan

The initial LEFT and RIGHT states bound the entire proposed wave system. They differ from the states adjacent to one already formed shock. A regular fan is ordered fast−, Alfvén−, slow−, contact, slow+, Alfvén+, fast+. Some waves can have zero strength. The signs refer to the local plasma; a minus-family wave can travel toward positive x in the chosen frame.

`solve_fans.py` adjusts five parameters until the candidate states at the contact have the same velocity and tangential magnetic field. Contact pressure is shared by construction; density may differ. Shocks use algebraic jump conditions. Rarefactions use isentropic wave curves integrated with DOP853. The common normal magnetic field remains fixed.

The starting guess is the retained P100 intermediate-state solution. P080 and P120 change only the initial LEFT thermal pressure by factors 0.8 and 1.2. Temperature changes with pressure; the other independent inputs are fixed. This is a local branch calculation, not a search for every possible MHD solution.

`check_fans.py` evaluates the result independently in SI units. It does not call the wave-construction routines. The checks include all discontinuity fluxes, positivity, shock entropy and characteristic inequalities, contacts, Alfvénic rotations, rarefaction isentropes and wave ordering. The fixed reference scales are written with each flux residual. A small residual measures numerical consistency, not observational accuracy.

## Observational checks

Solar Orbiter: MAG vectors are averaged over each recorded PAS acquisition interval, including its start and excluding its end. No interpolation adds proton measurements. A constant de Hoffmann–Teller velocity minimizes the mean squared inferred convection electric field. A common slope and intercept fit the pooled RTN components. Those components are not independent acquisitions.

SUVI: the script starts from the retained C/E feature measurements. Placing the model fast shock at C and the contact at E fixes every other wave position through a common affine mapping. An independently moved rarefaction would no longer belong to that same mapped fan.

AIA: G1 is an assumed great-circle sampling path, not a measured three-dimensional trajectory. The calculation repeats the primary peak/minimum selection from stored ray samples and reports sensitivity to averaging and smoothing. The model is placed along G1 using one initial alignment, an assumed elapsed model time of 120 s, and unchanged wave speeds. This mapping does not solve spherical MHD.

Geometry: a chosen projection angle aligns two selected features by construction. A separate fit recovers an approximate ellipsoid from digitized model contours published by Hu et al. (2019); its local normals are evaluated at stated assumed heights. STEREO visibility is checked from observer geometry. These calculations do not supply new stereoscopic tie points.
