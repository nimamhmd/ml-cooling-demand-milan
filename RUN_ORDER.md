# Run Order

This file documents the recommended order in which to execute the scripts in this repository to reproduce the analytical pipeline of the thesis. Each step depends on the outputs of the steps that precede it.

The full pipeline takes approximately 8–14 hours to run end-to-end on a single workstation with 16 GB RAM and a quad-core CPU, depending on the climate-ensemble extraction throughput.

---

## Stage 1 — Data Preprocessing (`01_data_preprocessing/`)

Prepare the CENED+2 EPC database, the DBT2012 cadastral inventory, and the integrated building-level analytical dataset.

| Step | Script | Purpose |
|---|---|---|
| 1.1 | `stage10_7_epc_inspection.ipynb` | First-pass inspection of the raw CENED+2 export |
| 1.2 | `stage10_8_epc_cleaning.ipynb` | Field selection, type coercion, deduplication |
| 1.3 | `10_9_EPC_Geocoding_and_Spatial_Matching.ipynb` | Geocoding of EPCs |
| 1.4 | `spatial_join_epc_to_buildings.py` | Spatial join EPC → DBT2012 cadastral polygons |
| 1.5 | `stage10.8_epc_spatial_merge.ipynb.ipynb` | EPC-to-building merge and EDIFC_ID-key checks |
| 1.6 | `merge_epc_batches.py` | Batched EPC merging |
| 1.7 | `merge_epc_into_buildings.py` | Final EPC-to-building aggregation |
| 1.8 | `aggregate_epc_to_building_level.py` | Building-level EPC aggregation |
| 1.9 | `recover_spatial_join_v2.py` | Recovery routine for unmatched EPCs |
| 1.10 | `stage9_buildings.ipynb` | Build the 53,041-building cadastral inventory |
| 1.11 | `prepare_stage1_stage2_datasets.py` *(in folder 03)* | Build the Phase 1 and Phase 2 ML-ready feature matrices |
| 1.12 | `audit_ml_dataset.py`, `dataset_validation.py`, `inspect_ml_dataset.py`, `check_representativeness.py` | Dataset audit and representativeness checks |

**Output:** ML-ready building-level dataset (19,063 EPC-labelled buildings + 53,041 cadastral buildings).

---

## Stage 2 — Phase 1 Cooling-Presence Classification (`02_phase1_classification/`)

Train four classifier classes and select the best performer for the city-wide spatial application.

| Step | Script | Purpose |
|---|---|---|
| 2.1 | `stage1_train_baselines.py` | Train Logistic Regression and Random Forest baselines |
| 2.2 | `train_stage1_cooling_classifier.py` | Train and tune XGBoost (selected as final Phase 1 model) |
| 2.3 | `Step_05_Stage1_Modelling.ipynb` | Modelling notebook with MLP comparison |
| 2.4 | `Step_09_Final_Evaluation.ipynb` | Held-out and spatial cross-validation evaluation; SHAP and LIME audits |

**Output:** Tuned XGBoost classifier (held-out ROC-AUC = 0.972) applied to the full 53,041-building cadastral stock.

---

## Stage 3 — Climate Data Extraction and Preprocessing (`04_climate_projection/`)

Extract ERA5-Land historical and NEX-GDDP-CMIP6 future climate data, then compute CDD22 (cooling-degree-days above 22 °C base).

| Step | Script | Purpose |
|---|---|---|
| 3.1 | `00_setup_gee_authentication.py` | Authenticate Google Earth Engine for climate-data extraction |
| 3.2 | `01_extract_nexgddp_milan.py` | Extract NEX-GDDP-CMIP6 daily fields for Milan |
| 3.3 | `preprocess_historical.py` | Preprocess ERA5-Land historical baseline (1990–2024) |
| 3.4 | `preprocess_future.py` | Preprocess CMIP6 future projections (2025–2100) |
| 3.5 | `02_validate_and_build_cdd.py` | Validate climate fields and compute per-building CDD22 |
| 3.6 | `Step_01_Compute_Climate_Indices.ipynb` | Climate-indices notebook with diagnostics |
| 3.7 | `attach_climate_to_buildings.py` | Attach climate data to building polygons |
| 3.8 | `stage10_climate.ipynb`, `eda_climate.ipynb.ipynb` | Exploratory analysis of climate data |

**Output:** Per-building, per-scenario, per-time-window CDD22 dataset (12 GCMs × 2 SSPs × 5 time windows = 120 climate-scenario realisations).

---

## Stage 4 — Phase 2 Regression and Demand Projection (`03_phase2_regression/` + `04_climate_projection/`)

Train the Phase 2 four-learner stacking ensemble for cooling intensity (β), then propagate to demand projections.

| Step | Script | Purpose |
|---|---|---|
| 4.1 | `Step_10_Stage2_Cooling_Intensity.ipynb` | Build β target, train stacking ensemble (XGBoost + LightGBM + CatBoost + MLP), evaluate with SHAP and LIME; this notebook also contains the ISO 31000 / MCDA decision-support analysis used in §4.9 of the thesis |
| 4.2 | `03_update_stage2_demand_projections.py` | Forward-propagate the trained Phase 2 model through the climate ensemble to produce V1, V2, V3 demand variants |

**Output:** Per-building, per-scenario projected demand for 53,041 buildings (V1, V2, V3 variants).

---

## Stage 5 — Uncertainty Decomposition (`05_uncertainty_sobol/`)

Quantify the relative contribution of each uncertainty source (climate-model identity, AC adoption rate, regression error, inter-annual variability) to projected demand variance.

| Step | Script | Purpose |
|---|---|---|
| 5.1 | `04_update_stage2_uncertainty.py` | Sobol-Saltelli total-order sensitivity decomposition for V1 and V3 demand variants |

**Output:** Sobol total-order indices (S_T) for each uncertainty source; bootstrap 95% confidence intervals.

---

## Stage 6 — Spatial Analysis (`06_spatial_analysis/`)

Aggregate building-level results to 88 NIL-scale spatial clusters and produce vulnerability maps.

| Step | Script | Purpose |
|---|---|---|
| 6.1 | `06_fix_maps_4_7_and_4_8.py` | NIL-cluster aggregation, top-15 high-demand clusters, vulnerability maps |

**Output:** NIL-cluster demand summary, top-15 high-demand cluster identification, district-level vulnerability maps.

---

## Stage 7 — Visualisation (`08_visualisation/`)

Produce all figures appearing in the thesis (Figures 3.1–5.2).

| Step | Script | Purpose |
|---|---|---|
| 7.1 | `05_update_stage2_visualizations.py` | Generate all thesis figures |

**Output:** PNG/PDF figures used in Chapters 3, 4, and 5 of the thesis.

---

## Notes

- The exact filename of each input/output is documented in the docstring at the top of each script.
- Several scripts (notably the climate-extraction routines) call external APIs (Copernicus CDS, Google Earth Engine) that require authentication credentials. These are not committed to the repository; see `data/DATA_ACCESS.md` for setup instructions.
- The decision-support tables and the ISO 31000 risk register presented in §4.9 of the thesis were generated inside `Step_10_Stage2_Cooling_Intensity.ipynb`; the `07_decision_support/` folder is reserved for any future modular rewrite of this analysis.
- Repeating the full pipeline on a fresh dataset requires re-running every stage in the order listed above; partial reruns are supported as long as the upstream artefacts are present on disk.
