# Expected results

The displayed numbers below are rounded for reading. JSON/CSV output retains full precision. Numerical residuals can change slightly with dependency versions and floating-point arithmetic while remaining below the stated thresholds.

| Calculation | Expected result |
| --- | --- |
| E11, 2010-06-13 | Conditional fast shock; compression 1.56; entropy increase about 0.17 c_v. |
| E05 fast, 2011-02-16 | A conditional fast-shock pair. |
| E05 slow, 2011-02-16 | A different conditional slow-shock pair. |
| All three single fronts | Seven normalized flux differences below 1e-12; positive shock entropy and correct characteristic ordering. |
| P080 / P100 / P120 | Outer fast-shock speeds 746.59 / 750.00 / 753.39 km s⁻¹. |
| Inner slow+ connection | Rarefaction in P080, shock in P100 and P120. |
| Fan verification | All independent condition flags true; fixed-scale flux residuals below 1e-9. |
| Solar Orbiter, padding 10 / 15 / 20 s | Six / nine / eleven PAS vectors; Walén slopes about 0.78 / 0.92 / 0.97. |
| Solar Orbiter core | One PAS acquisition within the four-second core; eight MAG samples in that acquisition. |
| SUVI association | Primary predicted fast rarefaction outside aperture B after the first epoch; final position outside B for all twelve variants. |
| AIA G1 | Primary maximum 42.20° → 57.95°; conditional surface-feature speed about 664.32 km s⁻¹. |
| AIA sensitivity | All eighteen explicitly listed mean/median, width and smoothing combinations have net outward displacement. |
| AIA trailing minimum | Remains brighter than the earlier reference; no contact or rarefaction identification. |
| Projection illustration | About 36.51° aligns the selected peak/contact separation by construction. |
| Recovered published geometry | Local normal inclination approximately 28° / 35° / 42° at assumed heights 0 / 50 / 100 Mm. |
| STEREO-A visibility | No selected point visible over the tested 0–200 Mm heights. |

E11 was motivated by Kozarev et al. (2011) and Ma et al. (2011); E05 by Veronig et al. (2011). The Solar Orbiter boundary was already analyzed by Suen et al. (2023). The geometric reconstruction starts from model curves published by Hu et al. (2019). References and the exact scope of each reproduction are in DATA.md.

`reference_results/` preserves the earlier full-precision results. The independent checks are recomputed; passing them is not inferred merely from agreement with a saved table. The release validation record reports the comparison with those retained values.
