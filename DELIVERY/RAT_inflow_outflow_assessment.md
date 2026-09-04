# RAT as an inflow/outflow data source for GNN-ReservoirNet - evaluation report

**Date:** 2026-08-29 | **Subject:** [UW-SASWE/RAT](https://github.com/UW-SASWE/RAT) (Reservoir Assessment Tool 3.x) | **Question:** can RAT supply the daily inflow/outflow targets the project's P0-1 data fix needs?

*Method: six research dimensions investigated and cross-verified against primary sources (the repository, official docs, the GMD 2024 and RAT 2.0 papers, live data portals); drafted as independent chapters; adversarially reviewed; nine review fixes applied and verified before delivery.*

# Executive summary & verdict

**Verdict.** RAT (the Reservoir Assessment Tool) is impressive, well-built, and genuinely relevant — but it is **not** a downloadable inflow/outflow dataset for our ten dams. It is a do-it-yourself satellite-plus-hydrological-model pipeline that must be installed and run per basin: Linux, a 129 GB database, three free accounts, and roughly 1–2 hours of compute per reservoir-year [^src-cost]. Its inflow output is uncalibrated natural-flow model output that the core papers never validated against gauges, and its outflow validated poorly (KGE ≤ 0) [^src-gmd]. As "found free data" it disappoints; as an open-source modeling framework layered on top of observed state-portal data, it is a credible — arguably unique — complement.

## The five facts that decide it

- **Code only, no data.** The repository ships software, not outputs: no precomputed inflow/outflow time series exist for anywhere, including our dams, and the 129 GB "global database" is static model *input* [^src-repo]. Nothing RAT-derived is published on Zenodo, HydroShare, or figshare [^src-survey].
- **Zero of our ten dams are configured.** The only India demo is the Hidkal dam tutorial (Krishna basin); its basin files (flow direction, VIC soil/domain, MetSim domain) come pre-made, so five of our dams — Srisailam, Nagarjuna Sagar, Almatti, Tungabhadra, plus Ujjani if the Bhima is included — start near-turnkey, while Cauvery, Godavari, Narmada and Tapi need user-built files [^src-docs]. The tutorial also skips the altimetry step, proving RAT runs fine without it.
- **Accuracy is asymmetric.** Storage change is well validated (r 0.77–0.82; ~7% nRMSE) and daily Penman evaporation is provided — but inflow is uncalibrated VIC natural flow that "requires careful calibration efforts," and outflow, a mass-balance residual, is weak at daily scale (KGE ≤ 0 at all three published test reservoirs) [^src-gmd]. Swapping our current storage-change proxy for raw RAT inflow would trade one proxy for another unless we calibrate per basin.
- **Run cost is bounded but real.** Linux-only (WSL2 likely fine but undocumented), ~129–140 GB free disk, three accounts (AVISO, NASA PPS, Google Earth Engine), sequential multi-basin compute — feasible on one workstation, incompatible with Kaggle/Colab limits [^src-cost].
- **No global dataset covers India.** Observed daily inflow/outflow for these dams exists only through Indian state portals — Karnataka's manual CSV (with Inflow/Outflow/Canal/Evaporation columns) is already in hand, while the global observed-reservoir collections (ResOpsUS, GDROM, ResOpsBR) contain zero of our dams [^src-alternatives]. RAT complements the observed-data hunt; it cannot replace it.

## Recommended path

1. **Keep the observed-data hunt as the backbone** — continue state-portal collection (NWDP) per state; this remains the only true training truth.
2. **Run the 30-minute Jason-3 track check first**: intersect our dam coordinates with RAT's `j3_tracks.geojson`. Jason-3 tracks alone decide whether RAT's best (altimetry-based) storage path exists per dam — Sentinel-3/SARAL literature does not transfer.
3. **Pilot one dam on WSL2** — Srisailam or Nagarjuna Sagar, using the pre-made Hidkal/Krishna package — and use the UW-hosted Kerala RAT CSVs (2015–2025, exact RAT output schema, Indian dams) as an instant offline test corpus requiring zero setup [^src-survey].
4. **Validate before trusting inflow**: calibrate VIC per basin and check calibrated RAT inflow against observed-portal overlaps per dam. Meanwhile, use RAT storage-change and evaporation immediately in the Stage-2 mass-balance stage, and treat RAT outflow as a weak-signal feature, not ground truth.
5. **Use GloFAS v5.0 (1980–2025) and GEOGloWS only as pre-2015 context features** — RAT's best products start ~2015 anyway, and neither alternative is validated at our dams [^src-alternatives].

## Confidence and what would change the verdict

Confidence is high on the structural facts (code-only repo, zero preconfigured dams, validation asymmetry, no India coverage in global datasets) and medium-high on cost extrapolations from a single official benchmark. The verdict downgrades if the Jason-3 checks fail for all Krishna dams *and* per-basin calibration proves prohibitive — RAT then shrinks to a storage-change/evaporation feature source. It flips to trivial adoption if UW publishes precomputed peninsular-India RAT outputs (none exists today) or a validated modeled-inflow archive covering India emerges.

---

[^src-repo]: RAT repository and distribution — https://github.com/UW-SASWE/RAT (code-only; global database = static inputs).
[^src-docs]: RAT official documentation and Hidkal (Krishna) tutorial — https://rat-satellitedams.readthedocs.io/.
[^src-gmd]: RAT 3.0 validation paper, Geoscientific Model Development (2024) — https://gmd.copernicus.org/articles/17/3137/2024/ — plus RAT 2.0 paper full text (validation statistics, inflow-calibration caveat).
[^src-cost]: RAT installation and compute-cost documentation — https://rat-satellitedams.readthedocs.io/ (129 GB database, account requirements, official Hidkal benchmark of 1–2 h per reservoir-year).
[^src-survey]: Survey of Zenodo/HydroShare/figshare for RAT-derived datasets (https://zenodo.org ; https://www.hydroshare.org ; https://figshare.com) and verification of UW-hosted Kerala CSV downloads — no published dataset found.
[^src-alternatives]: Cross-dataset coverage check (ResOpsUS, GDROM v2, ResOpsBR: 0/10 dams) and alternatives review (GloFAS v5.0, GEOGloWS, CWC products).

---

# What RAT actually provides — and what it doesn't

RAT 3.0 is a **code pipeline, not a data product**. The GitHub repository ships Python source, documentation, and tests; it contains no precomputed inflow or outflow time series. Time series exist only after a user runs the 14-step pipeline on their own machine, under their own credentials [^readme][^tree]. Adopting RAT therefore means adopting an operational workflow — compute, accounts, satellite downloads — not subscribing to a dataset.

## The seven outputs and where each comes from

A run produces seven per-reservoir variables, each written as a **per-dam CSV** under `final_outputs/` (subfolders `inflow`, `dels`, `evaporation`, `outflow`, `aec`, `sarea_tmsos`) [^computational][^tut]. The README states the derivation logic plainly: "RAT models the **Inflow (I)** and the **Evaporation (E)** of each reservoir. Finally, RAT uses the modeled I, and E, and estimated ΔS, to estimate the **Outflow (O)** from reservoirs." [^readme] The variables fall into three classes:

- **Modeled.** Inflow comes from VIC 5 + MetSim (daily meteorology disaggregated to 6-hourly) plus a routing model that converts "Routing streamflow values to inflow in m³/s" [^computational][^config]. Evaporation is "calculated using Penman Equation ( in mm)," driven by VIC meteorology [^computational].
- **Observed (satellite).** Surface area: "Calculate and extract surface area time series for each reservoir using Google Earth Engine (GEE)," fusing Landsat 5/7/8/9, Sentinel-2, and SAR into one series via TMS-OS, with an optional SAR-based BOT filter correcting optical bias [^computational][^config]. Elevation: "Extracts elevation for a reservoir using Jason-3 altimeter for those reservoirs lying on the path of Jason-3," via AVISO with the EGM2008 geoid applied [^computational][^globaldb].
- **Inferred / residual.** Storage change "using trapezoidal rule (in m³)" from area integrated over the AEC; outflow is a mass-balance residual: "finally outflow is estimated using mass-balance approach in m³/s" [^computational].

| Variable | Class | Derivation | Cadence |
|---|---|---|---|
| Inflow | Modeled | VIC 5 + MetSim + routing | Daily |
| Evaporation | Modeled | Penman equation (VIC met) | Daily |
| Surface area | Observed | GEE optical+SAR, TMS-OS | 2–5 days |
| Elevation | Observed | Jason-3 altimetry (AVISO) | Jason-3 cycle dates only |
| Storage change (ΔS) | Inferred | Area integrated over AEC | 2–5 days (inherits area) |
| Outflow | Residual | Mass balance: I − ΔS − E | 2–5 days |
| AEC | Static | SRTM-30 in GEE, or user-supplied | Static |

The docs are candid about outflow: "RAT provides outflow estimations at a frequency of 2-5 days limited by the aggregate sampling frequency of satellites it uses to compute reservoir storage change. This can produce uncertainties about the occurrences in between satellite overpasses." [^tut] The tutorial confirms area is estimated "at a frequency of 2-5 days" [^tut]. Elevation exists only on Jason-3 cycle dates, and only for reservoirs on a ground track; the commonly quoted ~10-day repeat period is **UNVERIFIED** in RAT's own docs.

## Operating requirements and the "global database"

Running the pipeline requires external services and credentials: a Google Earth Engine service account, an AVISO login ("for reservoir height data"), and NASA IMERG login, plus NOAA GFS / NCEP GEFS meteorology — the latter enabling near-real-time runs, since "the end date can now be set to a date on or before the current day" with recent met data "sourced as nowcasts" [^secrets][^config]. The `rat init -g` "global database" downloads as a zip from **Dropbox or Google Drive**, but it is **static inputs, not outputs**: GRDC basin shapefiles, GRanD v1.3 dams and reservoirs, SRTM30-plus elevation, NTSG 1/16° flow directions, per-continent VIC parameters, Jason-3 tracks, the EGM2008 geoid, and the compiled routing model [^globaldb][^initconfig][^config]. Its zip size and dam counts are not stated in the documentation — treat any specific figure as **UNVERIFIED** until the archive itself is inspected.

The flagship tutorial is the **2019 Karnataka floods / Hidkal dam** case (Krishna basin, basin_id 2312, GRanD id 4773), with expected `final_outputs` downloadable from Dropbox — a tutorial artifact, not a general database [^tut]. Notably, it skips altimetry entirely ("steps = [1,2,3,4,5,6,7,8,9,10,12,13,14] #Not running altimeter"), and the docs warn "users should apply their own judgement about the extrapolated region of the AEC" where SRTM is extrapolated below the Feb-2000 water level [^tut][^computational].

## What this means for the team

- **Daily inflow targets are genuinely available** — but they are fully modeled (VIC+MetSim+routing), so they are useful as *independent* cross-checks against gate/CWC records over the team's 2005–present window, not as ground truth; their quality is bounded by VIC calibration.
- **Outflow cannot serve as a daily target.** It is a mass-balance residual sampled at 2–5 days, with the docs themselves flagging "uncertainties about the occurrences in between satellite overpasses" [^tut] — a structural mismatch with a homogeneous daily training set.
- **Elevation coverage is a per-dam risk.** Only Jason-3-track reservoirs get level series; which of the team's 10 peninsular dams lie on tracks is undetermined here and must be checked.
- **Adoption is an infrastructure decision.** Expect GEE/AVISO/IMERG accounts, Linux compute, and pipeline operations to generate all series for the 10 dams — and parse the CSV headers carefully, since docs state m³/s while the tutorial's inflow plot is labeled m³/day.

---

[^readme]: Reservoir Assessment Tool (RAT) 3.0 — README / UW-SASWE, University of Washington. Accessed 2026-08-29. https://github.com/UW-SASWE/RAT
[^tree]: UW-SASWE/RAT full repo file tree via GitHub API. Fetched 2026-08-29. https://github.com/UW-SASWE/RAT
[^computational]: Computational Model — RAT 3.0 documentation. Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/en/latest/Model/ComputationalModel/
[^globaldb]: RAT Global Database — RAT 3.0 documentation. Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/en/latest/RAT_Data/GlobalDatabase/
[^config]: RAT Configuration File — RAT 3.0 documentation. Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/en/latest/Configuration/rat_config/
[^secrets]: RAT Secrets File — RAT 3.0 documentation. Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/en/latest/Configuration/secrets/
[^tut]: 2019 Karnataka Floods tutorial — RAT 3.0 documentation. Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/en/latest/Tutorials/KarnatakaFloods/
[^initconfig]: `src/rat/cli/rat_init_config.py` — UW-SASWE/RAT source. Accessed 2026-08-29. https://github.com/UW-SASWE/RAT/blob/main/src/rat/cli/rat_init_config.py

---

# Coverage: RAT and our 10 reservoirs

**Bottom line:** RAT ships with **0/10 of our dams pre-configured**; its only Indian example (Hidkal Dam, Krishna basin) **skipped the altimetry step**; five dams can reuse Krishna's pre-made files while the other five need team-built basin files; and satellite water-level coverage is a checkable hypothesis, not a given.

## Nothing ships pre-configured

RAT contains no reservoir list or catalogue at all — reservoirs enter only through user-supplied vector files named in the YAML config, and the bundled test basins are Nueces (Texas) and Gunnison (Colorado) [^rat-repo][^rat-tutorial]. The RAT v3 paper names none of our ten either; its India content is a Kerala-2018 flood study plus operational Mekong deployments [^rat-gmd]. "Global" here means global *inputs*, not ready-to-run Indian reservoirs.

## The Hidkal precedent — with a catch

The one official India case is the 2019 Karnataka Floods tutorial for **Hidkal Dam** (Ghataprabha river, Belagavi district — a Krishna-basin tributary), complete with a downloadable `custom_data` package: dam point file, reservoir polygon, Krishna basin shapefile (`basin_id = 2312`), Krishna flow-direction grid, Metsim domain and VIC soil/domain files [^rat-tutorial]. The catch sits in the run config: `steps = [1…10, 12, 13, 14]`, annotated *"#Not running altimeter"* — **step 11, altimetry, was skipped in RAT's own India example** [^rat-tutorial]. RAT's India-validated outputs are therefore inflow, surface area and storage change — not satellite water level.

## What configuring a new dam requires

Arbitrary new reservoirs are explicitly supported: the docs' illustration is that running "Hoover Dam" means swapping in new station, reservoir, basin, flow-direction, Metsim-domain and VIC files [^rat-tutorial]. For each of our dams that means a station file (dam point with id/name/lon/lat columns), a reservoir polygon (id, name, area in km²), a basin shapefile with a unique id column, a flow-direction grid, Metsim domain and VIC soil/domain files, plus registered credentials for AVISO, IMERG and Google Earth Engine [^rat-config-docs]. The underlying global data — GRDC basins, GRanD dams/reservoirs v1.3, GRWL, DRT flow directions, SRTM30+, EGM2008 — covers India by construction [^rat-init]. Per-dam GRanD IDs are UNVERIFIED; each dam needs its own entry.

## Per-dam coverage at a glance

| Dam (basin) | Published altimetry evidence | RAT setup basis | Status |
|---|---|---|---|
| Srisailam (Krishna) | Sentinel-3, 1.4 m drop (blog-grade) [^srisailam-linkedin] | Krishna files exist (Hidkal) | PARTIAL — needs track check |
| Nagarjuna Sagar (Krishna) | Sentinel-3 SRAL (peer-reviewed) [^ns-s3] | Krishna files exist | BEST of 10 — needs track check |
| Almatti (Krishna) | — none found | Krishna files exist | UNVERIFIED |
| Tungabhadra (Krishna) | — none found | Krishna files exist | UNVERIFIED |
| Ujjani (Krishna) | — none found | Krishna files exist (Ujjani only if the Bhima sub-basin is included) | UNVERIFIED |
| Mettur (Cauvery) | — none found | Custom basin files needed | UNVERIFIED |
| Krishnaraja Sagara (Cauvery) | — none found; small (~100 km² class), weakest case | Custom basin files needed | UNVERIFIED — riskiest |
| Jayakwadi (Godavari) | — none for level; Sentinel/GEE *extent* study [^jayakwadi-gee] | Custom basin files needed | UNVERIFIED |
| Sardar Sarovar (Narmada) | — none found (large reservoir, plausible) | Custom basin files needed | UNVERIFIED |
| Ukai (Tapi) | SARAL/AltiKa pass 825 (study figure) [^ukai-saral] | Custom basin files needed | PARTIAL |

*"—" = no evidence found in this pass. Absence of literature is not absence of ground tracks: altimetry satellites overfly all of peninsular India. Per-dam DAHITI/G-REALM coverage also UNVERIFIED.* [^dahiti]

## The Jason-3-only nuance (critical)

RAT's altimetry step reads exactly one track geometry: its global config maps the altimeter track file to `global_altimetry/j3_tracks.geojson` — Jason-3 ground tracks [^rat-init]. The Indian altimetry literature we found uses *other* missions: peer-reviewed Sentinel-3 SRAL at Nagarjuna Sagar [^ns-s3], blog-grade Sentinel-3 at Srisailam [^srisailam-linkedin], SARAL/AltiKa over Ukai [^ukai-saral]. Those studies prove the reservoirs are observable — they do **not** put a Jason-3 track over them, so they do not by themselves enable a RAT water-level series. (Jason-class validation on Indian reservoirs exists, R² up to 0.99, but which dams UNVERIFIED [^verma2021].) **The cheap, decisive check: intersect the 10 dam coordinates — better, the reservoir polygons — with RAT's shipped `j3_tracks.geojson`.** One short GIS session settles level-data feasibility for all ten dams.

## Uneven prep burden: five dams vs five

Krishna is essentially done: Hidkal's pre-made basin shapefile, flow-direction grid, Metsim domain and VIC files directly serve **five of our dams** — Srisailam, Nagarjuna Sagar, Almatti, Tungabhadra, and Ujjani if the pre-made Krishna files include the Bhima sub-basin — though each still needs its own dam point and reservoir polygon. The other five — Mettur and Krishnaraja Sagara (Cauvery), Jayakwadi (Godavari), Sardar Sarovar (Narmada), Ukai (Tapi) — span **four basins with no published RAT setup**, where flow-direction grids, Metsim domains and VIC files must be built by the team. That is a moderate GIS effort, not a blocker. Two risk flags: Krishnaraja Sagara is small (~100 km² class; G-REALM includes only lakes ≥ 100 km² [^grealm-ca]), and Jayakwadi is elongated — the classic worst case for satellite altimetry.

## What this means for the team

- **Plan for configuration, not just installation.** 0/10 dams pre-configured; Krishna's files are pre-made, the other four basins need team-built inputs before any forecast runs.
- **Sequence Krishna first.** Five dams ride Hidkal's package — use it to de-risk the pipeline while Cauvery/Godavari/Narmada/Tapi files are built.
- **Don't commit to RAT water-level series yet.** The only India run skipped altimetry and RAT reads Jason-3 tracks only; run the `j3_tracks.geojson` intersection for all 10 dams before promising level inputs.
- **Keep the fallback in scope.** RAT's inflow/surface-area/storage-change pipelines run without the altimeter step, so even where the Jason-3 check fails (7/10 dams remain UNVERIFIED), core forecasting inputs are unaffected.

## Citations

[^rat-repo]: UW-SASWE/RAT GitHub repository (main branch; Git Trees API listing, 202 files; incl. `docs/Plugins/Swot.md`, `docs/Tutorials/KarnatakaFloods.md`, `src/rat/cli/rat_init_config.py`). Accessed 2026-08-29. https://github.com/UW-SASWE/RAT
[^rat-tutorial]: RAT Documentation — "2019 Karnataka Floods" tutorial (Hidkal Dam, Karnataka, India; full config snippet with custom_data paths; steps list excluding altimeter). Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/ (source: docs/Tutorials/KarnatakaFloods.md)
[^rat-config-docs]: RAT Documentation — "RAT Configuration File" (12 sections; basin_shpfile/GRDC, SRTM30+, GRanD dams & reservoirs v1.3, GRWL, DRT flow directions, station & reservoir column dictionaries). Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/ (source: docs/Configuration/rat_config.md)
[^rat-init]: RAT source — `src/rat/cli/rat_init_config.py` (SUFFIXES_GLOBAL: `global_altimetry/j3_tracks.geojson`, `geoidegm2008grid.mat`, `GRanD_dams_v1_3_filtered.shp`, `GRanD_reservoirs_v1_3.shp`, `mrb_basins.json`, `World_e-Atlas-UCSD_SRTM30-plus_v8.tif`, `global_drt_flow_16th.tif`). Accessed 2026-08-29. https://raw.githubusercontent.com/UW-SASWE/RAT/main/src/rat/cli/rat_init_config.py
[^rat-gmd]: Minocha, S. et al. (2024). "Reservoir Assessment Tool version 3.0: a scalable and user-friendly software platform to mobilize the global water management community." Geoscientific Model Development 17, 3137–. Full text searched 2026-08-29. https://gmd.copernicus.org/articles/17/3137/2024/
[^ns-s3]: "Water Level Retrieval and Water Body Mapping: A Case Study of Nagarjuna Sagar Reservoir" (2020). ResearchGate publication 345869506. https://www.researchgate.net/publication/345869506
[^srisailam-linkedin]: Bhoda, S. K. — "Altimetry for Reservoir Ops: Sentinel-3/SAR for Water Level" (LinkedIn Pulse; secondary source). https://www.linkedin.com/pulse/altimetry-reservoir-ops-sentinel-3sar-water-level-santosh-kumar-bhoda-zuhwc
[^ukai-saral]: ResearchGate figure 284452676 — "Landsat-8 image of Ukai reservoir shown along with SARAL pass number 825" (from a Ukai reservoir altimetry study). https://www.researchgate.net/figure/Landsat-8-image-of-Ukai-reservoir-is-shown-along-with-SARAL-pass-number-825-Red-dot_fig2_284452676
[^verma2021]: Verma, K. et al. (2021). "Satellite altimetry for Indian reservoirs." Water Science and Engineering (Jason-2/Jason-3 evaluation; R²=0.99 best, RMSE from 0.11 m). https://wse.hhu.edu.cn/en/article/doi/10.1016/j.wse.2021.09.001 (reservoir list not extracted this pass; UNVERIFIED which dams)
[^grealm-ca]: Central Asia Climate Portal — G-REALM description ("Currently, lakes ≥ 100km2 are included"). https://centralasiaclimateportal.org/cacip_tools/global-reservoirs-and-lakes-monitor-g-realm/
[^dahiti]: DAHITI — Database for Hydrological Time Series of Inland Waters, DGFI-TUM. https://dahiti.dgfi.tum.de/en/products/water-level-altimetry/ (target-dam presence UNVERIFIED; portal requires interactive search)
[^jayakwadi-gee]: "Surface water dynamics analysis based on Sentinel imagery and Google Earth Engine Platform: a case study of Jayakwadi dam" (2021). ResearchGate publication 351614432. https://www.researchgate.net/publication/351614432

---

# Chapter 4 — How accurate are RAT's estimates?

RAT's accuracy story is uneven, and the unevenness matters more for this team than any single headline number: the products the project most wants — inflow and outflow — are precisely the ones validated least.

What the current version does and does not prove: RAT 3.0 is a software and scalability paper ("This paper will hereafter focus mostly on the software model development aspect of RAT 3.0") with no new quantitative validation tables [^gmd2024]. Every accuracy number below therefore comes from RAT 2.0 and companion application studies [^gmd2024].

**Storage change is the well-validated product.** RAT 2.0 was checked against in-situ records from Thailand's Electricity Generating Authority and Royal Irrigation Department at three Mekong reservoirs — Sirindhorn, Ubol Ratana, Lam Pao — over 2019–2021 [^das2022]. From RAT 1.0 to 2.0, "the average correlation increased from 0.59 to 0.77, the average normalized RMSE decreased from 15.3 % to 7.1 %"; altimetry-based estimates do best, at "an average correlation of 0.82, normalized RMSE of 7.3 %", but only for reservoirs on Jason-3 ground tracks [^das2022].

| Product | Evidence base | Headline numbers |
|---|---|---|
| Storage change (ΔS) | RAT 2.0 vs in-situ, 3 Thai Mekong reservoirs, 2019–2021 [^das2022] | avg r 0.59 → 0.77 (v1 → v2); altimetry r 0.82; nRMSE ~7 % |
| Outflow | RAT 2.0 daily estimates vs observed, same sites [^das2022] | daily r 0.13–0.55; KGE ≤ 0 at all three |
| Inflow | No quantitative gauge validation found in the core papers [^das2022] [^gmd2024] | Qualitative only: Kerala 2018 flood hydrograph timing [^gmd2024] [^suresh2024] |
| Evaporation | No accuracy test found in the core papers [^das2022] [^gmd2024] | Penman-based since RAT 2.0; skill unreported |

**Outflow is weak, and predictably so.** In RAT's mass balance, outflow is the residual — inflow minus storage change minus evaporation — so it inherits every error in the other terms. Inflow comes from an uncalibrated VIC model ("the VIC global parameters are uncalibrated" [^gmd2024]) and evaporation accuracy is unreported. RAT 2.0's daily outflow validation shows per-reservoir correlations of 0.55, 0.38 and 0.13 with KGE at or below zero (−0.08, −0.3, −0.3); RAT 1.0's monthly outflow was no better [^das2022]. The authors themselves note that "the calibration and choice of the hydrologic model play a crucial role in RAT's performance" [^gmd2024] — the KGE pattern is exactly what compounding residuals produce.

**Inflow is never quantitatively gauge-validated in the core papers.** RAT 2.0 obtained in-situ inflow data yet publishes no inflow metrics — its tables cover storage change and outflow only, and no inflow accuracy statement was found in the full text [^das2022]. The only India comparison is qualitative: during the 2018 Kerala floods, RAT 3.0 hydrographs at Banasurasagar and Idamalayaar dams captured "the timing of the peak flooding, the onset of sudden flooding, and the time to recede", with no numeric metrics in the v3 paper [^gmd2024] [^suresh2024]. Skill is user-dependent: "Achieving an accurate match between simulated and observed inflow requires careful calibration efforts" [^gmd2024]. RAT inflow is also *natural* (unregulated) flow; the ResORR add-on models upstream regulation and "clearly improves simulations when compared to the baseline (unregulated inflow) for most dams" in the US Cumberland/Tennessee test — again without numbers [^gmd2024].

**Structural limits to plan around.** Optical scenes with >90 % cloud cover are discarded — a real constraint in monsoon climates — while SAR penetrates cloud but threshold-based extent methods "have a tendency to underestimate the surface area" [^das2022]. The multi-sensor optical+SAR product (TMS-OS, 2–-day cadence) effectively starts around 2015, with Sentinel-2 and Landsat-8; pre-2015 hindcasts need a single-sensor fallback of unspecified accuracy [^gmd2024]. Area–elevation curves fall back to SRTM if not supplied, and sedimentation remains globally unresolved for lack of time-varying DEMs [^gmd2024].

**The counter-evidence is real.** RAT 2.0+3.0 runs operationally for the Mekong River Commission (via ADPC) and has been independently installed and tested in the Texas Gulf, Mesopotamia, Kerala, Columbia, Nile, Indus and Amazon basins [^gmd2024]. Sustained intergovernmental adoption, plus the Kerala flood-timing result, suggests the platform is decision-useful even where formal metrics are missing.

**What this means for the team**

- **ΔS and evaporation are the trustworthy products.** Storage change is validated to r ≈ 0.77–0.82 with ~7 % nRMSE — usable now as features or weak supervision.
- **Inflow needs per-basin calibration plus validation.** Treat RAT inflow as a starting point: calibrate the hydrologic model per basin and check against observed inflow from dam-portal records before trusting it as a target.
- **Outflow is a weak-signal feature only.** With daily r 0.13–0.55 and KGE ≤ 0, it should never serve as ground truth.
- **No peninsular-India validation exists yet.** None of the reviewed validations covers large South Indian reservoirs — the team would be among the first to quantify RAT accuracy there, which is both a risk and a publication opportunity.

[^gmd2024]: Minocha, S., Hossain, F., Das, P., Suresh, S., Khan, S., Darkwah, G., Lee, H., Galelli, S., Andreadis, K., and Oddo, P.: Reservoir Assessment Tool version 3.0: a scalable and user-friendly software platform to mobilize the global water management community. Geosci. Model Dev., 17, 3137–3156. University of Washington / UW-SASWE. 2024. https://gmd.copernicus.org/articles/17/3137/2024/

[^das2022]: Das, P., Hossain, F., et al.: Reservoir Assessment Tool 2.0 (RAT 2.0). Environmental Modelling & Software, 2022 (exact article title/volume not captured by PDF text extraction — UNVERIFIED). https://www.sciencedirect.com/science/article/abs/pii/S136481522200233X

[^suresh2024]: Suresh, S., Hossain, F., Minocha, S., Das, P., Khan, S., Lee, H., Andreadis, K., and Oddo, P.: Satellite-based tracking of reservoir operations for flood management during the 2018 extreme weather event in Kerala, India. Remote Sens. Environ., 307, 114149. 2024. (cited within [^gmd2024]; numeric metrics from this paper UNVERIFIED). https://doi.org/10.1016/j.rse.2024.114149 (verify on first click)

---

# What it takes to run RAT

**License — GPL-3.0, with a metadata wrinkle.** The LICENSE file, setup.py, and README all agree: RAT is GNU GPL v3 [^gh-license] [^gh-setup] [^gh-readme]; a stray PyPI classifier in pyproject.toml claims MIT — a metadata bug, not a license change [^gh-pyproject]. Copyleft binds only on distribution: modified versions must ship under GPL, with build and install instructions [^gh-readme]. An internal research project that clones RAT, runs it, and keeps code and results in-house triggers none of this; only redistributing a modified version would.

**Platform and install.** RAT is Linux-only: the docs' first requirement is a Linux operating system [^docs-userguide], because the VIC hydrological model runs exclusively on Unix [^gmd2024]. On the team's Windows laptops, WSL2 is the obvious route — plausible, but undocumented and untested by the maintainers (UNVERIFIED). Installation is conda-first — `mamba install rat -c conda-forge` [^gh-readme] [^docs-userguide] — with no pip requirements file; the ~26-entry conda environment carries a heavy geospatial stack (GDAL, rasterio, geopandas, rioxarray, netCDF4, dask, cfgrib, xarray) plus Fortran build tooling [^gh-env]. Heavy, but routine to install.

**Accounts: exactly three, nothing more.** All credentials go into one secrets file [^docs-userguide] [^docs-secrets]: AVISO (Jason-3 altimetry for reservoir height), NASA PPS IMERG (precipitation — the "Near-Realtime Products" checkbox at registration is mandatory, or login fails [^docs-gettingready] [^gpm-nrt]), and a Google Earth Engine service account for storage change, with Cloud projects required to register for Earth Engine since June 2024 [^docs-gettingready] [^gee-notice]. Just as notable is the absence: no Copernicus CDS/ERA5 account and no AWS S3 anywhere in the docs or dependencies [^gh-env] [^docs-cli]. The 129 GB global database (≥140 GB advised) comes via Google Drive or Dropbox through `rat init` [^docs-userguide] [^docs-cli].

**Compute: disk and time, not GPU.** The GMD paper says a standard laptop with 8 GB RAM, four cores, and a 512 GB hard disk suffices [^gmd2024]; the binding constraints are disk space and sequential CPU time. This is where the team's Kaggle/Colab plan breaks: RAT is CPU hydrology (VIC, MetSim, routing) plus cloud-side GEE queries, so its GPUs are irrelevant, while Kaggle and Colab ephemeral disks (~50–100 GB) cannot hold the 129 GB database plus outputs, and ~12-hour session caps collide with day-scale runs (feasibility assessment). The official numbers:

| Practical number | Value |
|---|---|
| Global database | 129 GB; ≥140 GB advised in the project directory [^docs-cli] |
| Hardware floor | 8 GB RAM / 4 cores / 512 GB disk [^gmd2024] |
| Official benchmark | ~1–2 h per reservoir-year (Hidkal dam, Krishna basin, 2019; altimetry step skipped) [^docs-karnataka] |
| Cold-start penalty | +800 simulated days of spin-up without initial-state files [^docs-karnataka] |
| Multi-basin runs | Strictly sequential — no parallel speedup [^docs-basins] |
| 10 dams × 20 years | Weeks of sequential compute — UNVERIFIED extrapolation |

That last row deserves honesty: the only published timing is one reservoir-year, and no per-decade benchmark exists anywhere in the docs or the paper, so the weeks-of-compute figure is the team's extrapolation, not a documented claim [^docs-karnataka].

**Repo health: alive, but one pair of hands.** Release v3.0.19 (November 2025) shipped with a SWOT plugin, and roughly ten tagged releases since early 2024 imply a quarterly cadence [^gh-releases]; the last main-branch commit was 2025-11-24, with activity continuing into January 2026 [^gh-commits] [^gh-repo]. But nearly all recent commits are Sanchit Minocha's — a bus factor of one to two [^gh-pyproject] — and the four genuinely open issues have sat stale since February 2023 [^gh-issues-open], even as August 2025 bug reports were closed within days [^gh-issues-closed]. Maintenance arrives in bursts tied to feature work, not steady triage.

**New reservoirs: documented, but scattered — and this is the hidden cost.** There is no single "adding a reservoir" page; the "2019 Karnataka Floods" tutorial plays that role — an end-to-end India example (Hidkal dam, Krishna basin) with Krishna's flow-direction, MetSim, and VIC files pre-made via Dropbox [^docs-karnataka]. Minimal GIS inputs are modest — a reservoir boundary polygon, dam coordinates, and a basin polygon — and RAT auto-derives area–elevation curves from SRTM in GEE when none is supplied [^docs-userguide] [^docs-globaldb]. Better yet, if the team's ten dams are GRanD dams inside GRDC major-basin polygons (as these major-basin reservoirs are expected to be — per-dam GRanD coverage still to verify), the default route runs nearly out-of-the-box [^docs-userguide] [^docs-globaldb]. The catch: any dam outside GRanD v1.3 needs user-built hydro-model files with no official tooling shown, and the pre-made Krishna files do not generalize. For a ten-dam plan, this is the single largest hidden cost.

**What this means for the team**

- **Pilot on WSL2 first** — the plausible but undocumented route for their Windows laptops (UNVERIFIED); verify it works before planning around it.
- **Register the three accounts early** (AVISO; NASA PPS IMERG with the NRT checkbox; GEE service account with the 2024 project registration) — friction, not blockers; all into one secrets file [^docs-gettingready] [^docs-secrets].
- **Run on a 512 GB-class machine or a VM, not Kaggle/Colab**; treat 10 dams × 20 years as weeks of sequential compute (UNVERIFIED) and validate with a one-reservoir-year run before committing [^docs-karnataka].
- **Check GRanD/GRDC coverage of all ten dams now**; budget real GIS and hydro-informatics effort for any dam that falls outside [^docs-globaldb] [^docs-karnataka].

---

[^gh-repo]: UW-SASWE/RAT — GitHub repository metadata via REST API (license, stars, forks, open_issues_count, pushed_at). Accessed 2026-08-29. https://api.github.com/repos/UW-SASWE/RAT
[^gh-commits]: UW-SASWE/RAT — commits API, latest 8 commits on default branch. Accessed 2026-08-29. https://api.github.com/repos/UW-SASWE/RAT/commits?per_page=8
[^gh-issues-open]: UW-SASWE/RAT — open issues/PRs via API. Accessed 2026-08-29. https://api.github.com/repos/UW-SASWE/RAT/issues?state=open&per_page=25
[^gh-issues-closed]: UW-SASWE/RAT — recently closed issues/PRs via API. Accessed 2026-08-29. https://api.github.com/repos/UW-SASWE/RAT/issues?state=closed&per_page=8
[^gh-releases]: UW-SASWE/RAT — releases API (tags v3.0.10–v3.0.19). Accessed 2026-08-29. https://api.github.com/repos/UW-SASWE/RAT/releases?per_page=10
[^gh-license]: RAT LICENSE file (GNU General Public License v3, full text 35,148 chars). Accessed 2026-08-29. https://raw.githubusercontent.com/UW-SASWE/RAT/main/LICENSE
[^gh-setup]: RAT setup.py (license = "GPL-3.0"). Accessed 2026-08-29. https://raw.githubusercontent.com/UW-SASWE/RAT/main/setup.py
[^gh-readme]: RAT README.md (conda-forge install, citation list, license statement). Accessed 2026-08-29. https://raw.githubusercontent.com/UW-SASWE/RAT/main/README.md
[^gh-pyproject]: RAT pyproject.toml (authors, requires-python, dependencies, classifiers). Accessed 2026-08-29. https://raw.githubusercontent.com/UW-SASWE/RAT/main/pyproject.toml
[^gh-env]: RAT environment.yml — conda dependency list. Accessed 2026-08-29. https://raw.githubusercontent.com/UW-SASWE/RAT/main/environment.yml
[^docs-userguide]: "User-Guide", RAT 3.x documentation (requirements, installation, initialization, testing). Docs undated; accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/en/latest/QuickStart/UserGuide/
[^docs-gettingready]: "Getting Ready", RAT 3.x documentation (AVISO, IMERG, GEE credential walkthroughs with screenshots). Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/en/latest/QuickStart/GettingReady/
[^docs-secrets]: "Secrets File", RAT 3.x documentation. Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/en/latest/Configuration/secrets/
[^docs-cli]: "Command Line Interface Functionality" (`rat init/test/run/configure`; 129 GB warning), RAT 3.x documentation. Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/en/latest/Commands/cli_commands/
[^docs-basins]: "RAT execution for Multiple Basins", RAT 3.x documentation. Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/en/latest/Configuration/basins_metadata/
[^docs-karnataka]: "2019 Karnataka Floods" — official step-by-step tutorial for a custom Indian reservoir (Hidkal dam, Krishna basin), incl. hardware requirements, timing, and config code. Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/en/latest/Tutorials/KarnatakaFloods/
[^docs-globaldb]: "Global Database" (GRDC MRB, GRanD v1.3, NTSG flow direction, SRTM30-plus, EGM2008 geoid; GEE SRTM AEC), RAT 3.x documentation. Accessed 2026-08-29. https://rat-satellitedams.readthedocs.io/en/latest/RAT_Data/GlobalDatabase/
[^gmd2024]: Minocha, S., Hossain, F., Das, P., Suresh, S., Khan, S., Darkwah, G., Lee, H., Galelli, S., Andreadis, K., Oddo, P. (2024). "Reservoir Assessment Tool version 3.0: a scalable and user-friendly software platform to mobilize the global water management community." Geoscientific Model Development, 17, 3137–3156. https://gmd.copernicus.org/articles/17/3137/2024/
[^gee-notice]: Google Earth Engine announcements group — "[Notice] Removing access for unregistered Cloud projects" (2024-06-17 registration deadline). https://groups.google.com/g/google-earthengine-announce/c/fPq4zEKdTSo
[^gpm-nrt]: NASA Global Precipitation Measurement — "PPS Near Real-time" data access page (registration checkbox requirement). https://gpm.nasa.gov/data/sources/pps-nrt

---

# Chapter 6 — Existing RAT data you can download today

RAT is not only code: the University of Washington's SASWE group runs a live RAT 3.0 web application [^rat-app], and its bundled configuration reveals exactly seven precomputed regions — Texas ×3 (Sabine, Trinity, Colorado), Mesopotamia (Tigris–Euphrates), the Mekong, and two South-Asia instances covering Kerala and the Indus [^rat-js]. That is the entire public footprint; no Krishna, Godavari, Cauvery, Bhima, Tapi, or Narmada region exists anywhere in the app.

| Instance | Coverage | CSV download status |
|---|---|---|
| Kerala (UW-hosted) | ~19–44 Kerala reservoirs: Idukki, Kakki, Mullaperiyar, Malampuzha, Banasurasagar, … [^kerala-geojson] | **Verified** for Idukki (inflow, outflow, evaporation, dS) [^idukki-inflow] [^idukki-outflow] |
| Indus | 27 dams — Pakistan (Tarbela, Mangla, Warsak, Chashma) plus Indian Himalayan (Bhakra, Pong, Salal, Baglihar, Thein, Nangal) [^indus-geojson] | Not verified |
| Mekong | 25+ stations across Vietnam/Laos/Cambodia/Thailand/China (Lower Sesan II, Yali, Nam Ngum, Nam Theun 2, …) [^mekong-geojson] [^mekong-site] | Presumed same pattern; unverified |
| Texas ×3, Mesopotamia | Listed in the app's region map [^rat-js] | Not examined |
| CWRDM Kerala (partner-operated) | Kerala reservoirs, operational adoption [^cwrdm] | **UNVERIFIED** (server timed out) [^cwrdm-doc] |

**Downloads are plain CSVs with no authentication.** The web app fetches its data client-side from predictable URLs of the form `https://depts.washington.edu/saswe/{regionFolder}/data/{Basin}/{variableFolder}/{dam}.csv` [^rat-js]. We verified this end-to-end on Kerala's Idukki reservoir: <https://depts.washington.edu/saswe/kerala/data/Kerala/inflow/idukki.csv> downloads directly and holds 3,565 daily rows from 2015-10-24 to 2025-07-26 under the header `date,inflow (m3/d)` [^idukki-inflow]. The companion series — outflow, evaporation (mm), and dS (storage change, m³) — downloaded from the same folder pattern under `outflow/`, `evaporation/`, and `dels/` [^idukki-outflow]. No login, no API key.

**What the Kerala instance is good for.** Its dams are not ours, but its format is: the Kerala CSVs are a ready-made Indian test corpus in RAT's exact output schema — daily `date,inflow (m3/d)`, `date,outflow (m3/d)`, `date,evaporation (mm)`, `date,dS (m3)` — a schema scientifically exercised on 19 Kerala reservoirs during the 2018 floods [^suresh-preprint]. The team can build, parse, and unit-test the data-ingestion half of its pipeline today with zero setup, before ever running RAT itself.

**Two caveats.** First, UW-hosted series are static snapshots, not live feeds: the verified Kerala data ends 2025-07-26 — roughly 13 months stale as of this writing (2026-08-29) — and no update schedule is published (update frequency: UNVERIFIED) [^idukki-inflow]. Second, Kerala's government water-research centre CWRDM operates its own RAT 3.0 Kerala instance, co-developed with UW SASWE under NASA Applied Sciences support [^cwrdm]; its pages surfaced via search but timed out on direct fetch, so its download capability remains UNVERIFIED [^cwrdm-doc].

**Bottom line: no shortcut to the target dams.** None of the ten target dams — Srisailam, Nagarjuna Sagar, Almatti, Tungabhadra, Ujjani, Mettur, Krishnaraja Sagara, Jayakwadi, Sardar Sarovar, Ukai — appears in any RAT station file or region list; their basins simply are not among the seven live regions [^kerala-geojson] [^indus-geojson] [^rat-js]. Nor is RAT output archived elsewhere: no RAT datasets were found on Zenodo, HydroShare, or figshare, for India or globally — RAT is distributed as code plus a small test dataset via GitHub/pip [^rat-github] [^gmd] [^ems-paper].

## What this means for the team

- Download the verified Kerala CSVs now and use them as an Indian test corpus for pipeline development — the schema (`date,inflow (m3/d)` … `date,dS (m3)`) is exactly what a future self-run RAT deployment will produce.
- Do not expect precomputed coverage of the 10 target dams: no instance or public archive covers them, so running RAT 3.0 over the Krishna/Godavari/Cauvery/Bhima/Tapi/Narmada basins is the only data path.
- Treat UW-hosted series as ~13-months-stale historical snapshots, and validate RAT's model-derived fluxes against CWC/India-WRIS gauge records before training on them.
- Re-probe the CWRDM operational instance (downloads UNVERIFIED) later — its in-country deployment is the closest analogue to a maintained, current RAT installation.

## Sources

[^rat-app]: Reservoir Assessment Tool (RAT) 3.0 — live web app, SASWE Research Group, University of Washington. Fetched 2026-08-29. http://depts.washington.edu/saswe/rat/
[^rat-js]: RAT 3.0 web app JavaScript bundle (`index-994fbf0f.js`) — source of region map, region+folder mapping, and CSV URL template. Fetched 2026-08-29. https://depts.washington.edu/saswe/rat/assets/index-994fbf0f.js
[^kerala-geojson]: `Kerala_station.geojson` — station file listing Kerala reservoirs in live RAT app. Fetched 2026-08-29. https://depts.washington.edu/saswe/rat/station_geojsons/Kerala_station.geojson
[^indus-geojson]: `Indus_station.geojson` — station file listing 27 Indus-basin dams (Pakistan + Indian Himalayan). Fetched 2026-08-29. https://depts.washington.edu/saswe/rat/station_geojsons/Indus_station.geojson
[^mekong-geojson]: `mekong_station.geojson` — station file listing 25+ Mekong-basin dams/stations. Fetched 2026-08-29. https://depts.washington.edu/saswe/rat/station_geojsons/mekong_station.geojson
[^mekong-site]: "Reservoir Assessment Tool (RAT) 3.0 — Mekong" site (UW-SASWE). Accessed via search 2026-08-29. https://depts.washington.edu/saswe/mekong/howtocite.html
[^idukki-inflow]: Idukki (Kerala) daily inflow CSV — verified direct download, 3,565 rows, 2015-10-24→2025-07-26. Fetched 2026-08-29. https://depts.washington.edu/saswe/kerala/data/Kerala/inflow/idukki.csv
[^idukki-outflow]: Idukki (Kerala) daily outflow / evaporation / dS CSVs — verified direct downloads. Fetched 2026-08-29. https://depts.washington.edu/saswe/kerala/data/Kerala/outflow/idukki.csv ; https://depts.washington.edu/saswe/kerala/data/Kerala/evaporation/idukki.csv ; https://depts.washington.edu/saswe/kerala/data/Kerala/dels/idukki.csv
[^suresh-preprint]: Suresh, S. et al. (2023). "Satellite-based Tracking of Reservoir Operations for Flood Management during the 2018 Kerala Floods" — HESS preprint. https://hess.copernicus.org/preprints/hess-2023-193/
[^cwrdm]: "Reservoir Assessment Tool Kerala (RAT 3.0)" — Centre for Water Resources Development and Management (CWRDM), Kerala. Accessed via search 2026-08-29; direct fetch timed out. https://cwrdm.kerala.gov.in/reservoir-assessment-tool-kerala-rat-30
[^cwrdm-doc]: "RAT — Documentation" — CWRDM-hosted RAT Kerala instance docs (IP-hosted server). Accessed via search 2026-08-29; direct fetch timed out on re-check. http://210.212.226.241:8080/RAT_Kerala_CWRDM/documentation.html
[^rat-github]: UW-SASWE/RAT — official GitHub repository (code distribution; test dataset via CLI). https://github.com/UW-SASWE/RAT ; docs: https://rat-satellitedams.readthedocs.io/
[^gmd]: Minocha, S. et al. (2024). "Reservoir Assessment Tool version 3.0: a scalable and user-friendly software platform to mobilize the global water management community." Geoscientific Model Development, 17, 3137–. https://gmd.copernicus.org/articles/17/3137/2024/ (page fetch failed twice on 2026-08-29 from research environment)
[^ems-paper]: Minocha, S. et al. (2025). "Reservoir assessment tool (RAT): a Python package for monitoring the dynamic state of reservoirs and analyzing dam operations." (journal attribution unverified — DOI resolves at tandfonline.com while Environmental Modelling & Software is an Elsevier title; PDF mirror: https://depts.washington.edu/saswe/rat/user_manual/Rat_pythonl_paper_EMS_v1.pdf) — verify the journal name before ship.

---

# 7. Alternatives and the recommended data strategy

## 7.1 What exists besides RAT — and what doesn't

The alternatives search returned its most important result first: **no global dataset of observed reservoir inflow or outflow covers India.** The three flagship observed-operations datasets of 2021–2026 — ResOpsUS (679 US reservoirs), GDROM v2 (2,017 CONUS reservoirs) and ResOpsBR+CARS (142 Brazilian reservoirs) — cover **0 of our 10 dams** [^ressopsus][^gdrom2][^resopsbr]. Such datasets get built where operator records are public; India is missing. That is exactly the gap RAT-type estimation targets — and exactly why consolidating state-portal data matters.

What the search did surface falls into four usable groups.

**GloFAS v5.0 reanalysis (1980–2025) — best long-history modeled inflow.** Copernicus' new historical reanalysis tracks daily global river flows from 1980 through 2025, superseding the GloFAS-ERA5 v2.1 product the team already knows, under a free and open license [^glofas5][^glofas-ewds][^glofas-era5]. A first nationwide GloFAS evaluation for India now exists (drought-monitoring framing) [^glofas-india]. Caveat: it routes near-natural flow and does not know gate operations, so upstream dam cascades can distort "inflow" at regulated sites.

**GEOGloWS ECMWF Streamflow Service — easiest modeled-inflow pull.** A REST API with no authentication serves a daily retrospective plus ensemble forecasts [^geoglows-web][^geoglows2]; the service has been formally introduced to India via the World Bank National Hydrology Program [^geoglows-india]. But no published validation of its retrospective in peninsular India was found (UNVERIFIED). Spot-check it against the KRS 2011–2020 observed truth before investing any extraction effort.

**CWC products — constraints, not inflow archives.** The weekly bulletin reports live storage only (no inflow/outflow columns) for 166 reservoirs every Thursday; all 10 dams are likely among the monitored set (per-dam verify) [^cwc-bulletin][^cwc-rsb]. CWC's inflow forecasts at 128 reservoirs are a real-time monsoon product used for gate operations — not a curated historical archive [^cwc-ff].

**Satellite altimetry (G-REALM, DAHITI, Hydroweb, GROWL) — level only.** These provide water-level series, not discharge [^grealm][^dahiti][^hydroweb][^growl]. They cannot produce inflow, but they independently validate storage reconstructions. Per-dam coverage for our 10 dams is UNVERIFIED, and narrow-gorge reservoirs such as Srisailam and Nagarjuna Sagar are known-hard altimetry targets.

| Source | Inflow | Outflow | Type | Dams of our 10 | Period | Access |
|---|---|---|---|---|---|---|
| ResOpsUS / GDROM v2 / ResOpsBR+CARS | Yes | Yes | Observed | **0** | varies, ends ~2020 | Low (open archives) [^ressopsus][^gdrom2][^resopsbr] |
| NWDP state portals (baseline) | Yes | Yes (+canal/evap in KA) | Observed | Potentially all 10; KA CSVs already in hand | Recent years; depth varies by state | Medium (portal scraping) |
| GloFAS v5.0 reanalysis | Modeled proxy | No | Modeled | All 10 (any grid point) | 1980–2025 | Medium (EWDS account + API) [^glofas5] |
| GEOGloWS (GESS) | Modeled proxy | No | Modeled | All 10 (nearest reach) | Start year UNVERIFIED | **Low** (REST, no auth) [^geoglows-web] |
| CWC weekly bulletin | No | No (storage only) | Observed | Likely all 10 among 166 (verify) | Weekly (Thu) | Medium (weekly scrape) [^cwc-bulletin] |
| CWC inflow forecasts | Forecast | Partly | Modeled/hybrid | 128 stations (per-dam UNVERIFIED) | Real-time, no archive | Medium [^cwc-ff] |
| G-REALM / DAHITI / Hydroweb / GROWL | No (level only) | No | Observed (altimetry) | Per-dam UNVERIFIED | 1992–present, varies | Low (registration) [^growl] |

## 7.2 The recommended strategy: observed where it exists, RAT where it doesn't

Layer the pipeline by trust:

1. **Observed truth.** Daily inflow/outflow from state portals (NWDP) remain the only observed source for these dams; Karnataka's manual CSVs — Inflow, Outflow, Canal, Evaporation — are already downloaded. Wherever this layer exists, it is the training truth; everything else is a feature or a check.
2. **RAT as gap-filler and feature engine.** RAT's trustworthy, validated products are storage-change ΔS (r ≈ 0.77–0.82) and daily Penman evaporation — exactly the mass-balance terms the project's Stage 2 (`S(t+1) = S(t) + I − E − R`) lacks. Its inflow becomes a target only after per-basin VIC calibration and validation against the observed subsets; its outflow (KGE ≤ 0 at daily scale in the only published validation) is a weak-signal feature, never ground truth.
3. **Pre-2015 context.** RAT's best products start ~2015 (the multi-sensor era); its single-sensor Landsat mode reaches the early 1980s — usable for climatology features, not targets. GloFAS v5.0 and GEOGloWS span the full training window but enter only as context features until validated at these dams.
4. **Independent validation.** Altimetry levels and CWC weekly storage stay outside the training loop, as independent observations of reservoir state for checking RAT's ΔS reconstruction and the overall water balance.

The result is a deliberately **two-era dataset**: 2015–present is the core (observed targets plus RAT ΔS/evaporation features); 2005–2015 is the context era (observed where available, GloFAS/GEOGloWS features elsewhere). A model built this way never mistakes a modeled proxy for a target.

Three first actions, in order:

1. **Jason-3 track check (30 minutes).** RAT's level step consumes Jason-3 tracks exclusively (a SWOT plugin exists but is unproven for India). Intersect the 10 dam coordinates with the `j3_tracks.geojson` shipped with RAT to see which dams can feed its level path; RAT runs fully without that step anyway (the official India tutorial did).
2. **One-dam pilot on WSL2.** Run Srisailam or Nagarjuna Sagar first — Krishna-basin files are pre-made via the official Hidkal tutorial, and one Krishna run covers 5 of the 10 dams (Ujjani only if the Bhima is included). Budget roughly 1–2 hours of compute per reservoir-year.
3. **Validation gate.** Wherever RAT inflow overlaps observed state-portal data, validate before any wider rollout. If calibrated RAT inflow clears the bar, extend it; if not, RAT still pays for itself as ΔS plus evaporation.

[^ressopsus]: ResOpsUS, a dataset of historical reservoir operations in the contiguous United States (Steyaert et al., Scientific Data). 2022. https://www.nature.com/articles/s41597-022-01134-7
[^gdrom2]: GDROM v2: An Inventory of Operation Variables Time Series and Rules for 2,017 Large Reservoirs across the CONUS (Zheng et al., Scientific Data). 2025. https://www.nature.com/articles/s41597-025-06162-7
[^resopsbr]: ResOpsBR+CARS: Reservoir Operations Brazil (Zenodo). 2025. https://zenodo.org/records/16096623
[^glofas5]: The new Copernicus GloFAS v5.0 hydrological reanalysis has been released (Copernicus EMS news). 2025/2026. https://global-flood.emergency.copernicus.eu/news/252-the-new-copernicus-glofas-v50-hydrological-reanalysis-has-been-released/
[^glofas-ewds]: River discharge and related historical data from the Global Flood Awareness System (CEMS-GloFAS historical, EWDS/Climate CDS). 2026. https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical
[^glofas-era5]: GloFAS-ERA5 operational global river discharge reanalysis 1980–2018 (Harrigan et al., ESSD). 2020. https://essd.copernicus.org/articles/12/2043/2020/
[^glofas-india]: Evaluating the GloFAS discharge reanalysis product for hydrological drought monitoring in India: performance across temporal scales and flow regimes (Taylor & Francis / figshare). ~2025. https://tandf.figshare.com/articles/journal_contribution/Evaluating_the_GloFAS_discharge_reanalysis_product_for_hydrological_drought_monitoring_in_India_performance_across_temporal_scales_and_flow_regimes/32296719
[^geoglows-web]: GEOGloWS ECMWF Streamflow Service portal & REST API. https://geoglows.ecmwf.int/
[^geoglows2]: GEOGLOWS 2.0 ECMWF Streamflow Model (10-day ensemble; NASA GIS item). 2024–2025. https://gis.earthdata.nasa.gov/portal/home/item.html?id=06ae694c709f4efea943881c04e33cf4
[^geoglows-india]: India — GEOGloWS Stories (Aquaveo + World Bank NHP). ~2024. https://stories.geoglows.org/east-asia/india
[^cwc-bulletin]: Reservoir Storage Monitoring System — weekly bulletin list, CWC (rsms.cwc.gov.in). 2026. https://rsms.cwc.gov.in/frameWork/web/bulletin-report-page
[^cwc-rsb]: Reservoir Level & Storage Bulletin, CWC. https://www.cwc.gov.in/reservoir-level-storage-bulletin
[^cwc-ff]: Flood Forecasting / Hydrological Observation, CWC. https://www.cwc.gov.in/flood-forecasting-hydrological-observation
[^grealm]: Global Reservoirs and Lakes Monitor (G-REALM), USDA International Climate Hub. https://www.climatehubs.usda.gov/hubs/international/tools/global-reservoirs-and-lakes-monitor-g-realm
[^dahiti]: Water Level Time Series from Satellite Altimetry — DAHITI (DGFI-TUM). https://dahiti.dgfi.tum.de/en/products/water-level-altimetry/
[^hydroweb]: Hydroweb — LEGOS water levels of rivers and lakes (Theia/LEGOS). https://www.legos.omp.eu/en/hydroweb-2/
[^growl]: A global dataset of reservoir in-situ water levels (GROWL) (Zhang et al., Scientific Data). 2026. https://www.nature.com/articles/s41597-026-07091-9