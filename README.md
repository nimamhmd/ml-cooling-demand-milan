# Explainable Machine Learning Framework for Building-Level Cooling Demand Projection under Climate Change Scenarios: The Case of Milan

**MSc Thesis — Politecnico di Torino, Department of Management and Production Engineering (DIGEP)**

| | |
|---|---|
| **Author** | Nima Mohammadi |
| **Supervisor** | Prof. Timur Narbaev |
| **Programme** | MSc Building Engineering |
| **Institution** | Politecnico di Torino, Turin, Italy |
| **Defence** | July 2026 |

---

## Overview

This repository contains the Python code developed for the MSc thesis listed above. The thesis builds a two-stage explainable machine-learning pipeline that:

1. **Phase 1 — Cooling Presence Classification.** Predicts whether each residential building in Milan has an active summer cooling system. Four classifier classes are evaluated (Logistic Regression, Random Forest, XGBoost, Multilayer Perceptron) on 19,063 buildings labelled through Energy Performance Certificates (EPCs). **Tuned XGBoost is selected as the final Phase 1 model** on the basis of held-out and spatial cross-validation performance, and applied city-wide to the 53,041-building cadastral stock.

2. **Phase 2 — Cooling Intensity Regression.** Predicts a per-building climate-normalised cooling-intensity coefficient (β) using a four-learner stacking ensemble (XGBoost, LightGBM, CatBoost, MLP). The coefficient is then forward-propagated through a 12-model NEX-GDDP-CMIP6 climate ensemble to produce city-wide residential cooling demand projections for Milan under SSP2-4.5 and SSP5-8.5 through 2100.

Both phases include a full explainability audit using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations). Uncertainty in the projections is decomposed using the Sobol-Saltelli variance-decomposition method. Results are aggregated to 88 NIL-scale spatial clusters for municipal planning use, and a decision-support demonstration is provided following ISO 31000:2018 and ISO 50001:2018 process standards.

---

## Key Results

| Result | Value |
|---|---|
| Phase 1 held-out ROC-AUC (tuned XGBoost) | 0.972 |
| Phase 1 spatial cross-validation gap | 0.003 ROC-AUC points |
| Phase 2 held-out R² (log1p-β scale) | 0.689 [95% CI: 0.644, 0.731] |
| City-wide cooling demand, ERA5 baseline | 38.6 GWh yr⁻¹ |
| City-wide cooling demand, SSP5-8.5 / 2080–2100 (V1) | 139.5 GWh yr⁻¹ [93.0, 180.9] |
| Percentage increase, SSP5-8.5 / 2080–2100 | +261.8% |
| Dominant uncertainty source (Sobol S_T, V1) | Climate-model identity: 94.3% |
| Buildings in top-15 high-demand NIL-scale clusters | 29.2% of cooled stock, 35.2% of projected demand |

---

## Repository Structure

```
ml-cooling-demand-milan/
│
├── README.md                          ← This file
├── RUN_ORDER.md                       ← Recommended script execution order
├── requirements.txt                   ← Python dependencies
├── LICENSE                            ← MIT Licence
│
├── 01_data_preprocessing/             EPC cleaning, cadastral matching, EPC-to-building
│                                      merging, dataset audit and representativeness checks.
│
├── 02_phase1_classification/          Stage 1 cooling-presence classifier training
│                                      (LR, RF, XGBoost, MLP), final XGBoost evaluation.
│
├── 03_phase2_regression/              Stage 2 cooling-intensity (β coefficient) stacking
│                                      ensemble training and Phase 1/Phase 2 dataset preparation.
│                                      Note: this folder also contains the embedded ISO 31000 /
│                                      MCDA decision-support analysis used in §4.9 of the thesis.
│
├── 04_climate_projection/             CMIP6 (NEX-GDDP) extraction, ERA5-Land processing,
│                                      CDD22 computation, demand-projection propagation,
│                                      Google Earth Engine setup.
│
├── 05_uncertainty_sobol/              Sobol-Saltelli total-order sensitivity decomposition
│                                      across V1 and V3 demand-variant configurations.
│
├── 06_spatial_analysis/               NIL-cluster aggregation, vulnerability maps,
│                                      district-level cohort decomposition.
│
├── 08_visualisation/                  Visualisation scripts for thesis figures.
│
└── data/
    └── DATA_ACCESS.md                 ← How to obtain the datasets used in this thesis
```

Note: Phase 1 evaluates four classifier classes (LR, RF, XGBoost, MLP); tuned XGBoost is the selected final model. Phase 2 is a four-learner stacking ensemble (XGBoost + LightGBM + CatBoost + MLP). The decision-support workflow (ISO 31000, MCDA, PDCA) is presented in §4.9 of the thesis; its supporting code is embedded inside the Phase 2 cooling-intensity notebook rather than as a separate module.

---

## Data Availability

The raw datasets used in this thesis cannot be redistributed in this repository. Access instructions are provided in `data/DATA_ACCESS.md` and summarised below.

**CENED+2 — Lombardy Regional EPC Database.** The CENED+2 database is maintained by Regione Lombardia. Research access can be requested through the regional energy agency (AREXPO / Struttura Certificazione Energetica). The thesis used a snapshot dated 28 November 2025 (file: `Database_CENED+2_-_Certificazione_ENergetica_degli_EDifici_20251128.csv`). After filtering to the Municipality of Milan, the working subset contained **342,684 EPC certificates**; after deduplication and matching to DBT2012 cadastral building polygons via the EDIFC_ID key, **19,063 buildings carried at least one valid EPC label** and were retained as the Phase 1 supervised training corpus.

**DBT2012 — Milan Building Cadastre.** The Database Topografico (DBT2012) building footprint layer is maintained by the Comune di Milano and is available through the Geoportale del Comune di Milano (geoportale.comune.milano.it). The cadastral inventory comprises 53,041 residential building polygons.

**ERA5-Land — Climate Reanalysis.** ERA5-Land hourly 2-metre air temperature data (0.1° resolution, 1990–2024) is publicly available through the Copernicus Climate Change Service (C3S) Climate Data Store at cds.climate.copernicus.eu. No access restrictions apply.

**NEX-GDDP-CMIP6 — CMIP6 Downscaled Climate Projections.** The NASA Earth Exchange Global Daily Downscaled Projections (NEX-GDDP-CMIP6) dataset is publicly available through NASA at www.nasa.gov/nex and via Google Earth Engine. The 12 GCMs used in this thesis are listed in Table 3.6 of the thesis document.

---

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/nimamhmd/ml-cooling-demand-milan.git
cd ml-cooling-demand-milan
pip install -r requirements.txt
```

Python 3.10 or higher is recommended. All scripts were developed and tested on Python 3.11.

---

## Execution Order

The recommended order in which to run the scripts is documented in `RUN_ORDER.md`. In brief, the pipeline runs from EPC data ingestion (folder 01), through Phase 1 classifier training (folder 02), Phase 2 regression and decision-support work (folder 03), climate projection (folder 04), uncertainty decomposition (folder 05), spatial vulnerability mapping (folder 06), and finally figure generation (folder 08).

---

## Requirements

The full list of dependencies is in `requirements.txt`. Key packages:

```
numpy
pandas
scipy
scikit-learn
xgboost
lightgbm
catboost
torch
shap
lime
SALib
geopandas
rasterio
xarray
netCDF4
earthengine-api
matplotlib
seaborn
joblib
tqdm
```

---

## How to Cite

If you use this code in your research, please cite the thesis as follows:

```
Mohammadi, N. (2026). Explainable Machine Learning Framework for Building-Level
Cooling Demand Projection under Climate Change Scenarios: The Case of Milan.
MSc Thesis, Politecnico di Torino, Turin, Italy.
Supervisor: Prof. T. Narbaev.
```

---

## Licence

This repository is released under the MIT Licence. See the `LICENSE` file for details.
You are free to use, modify, and distribute the code with attribution.

---

## Contact

**Nima Mohammadi**
MSc Building Engineering, Politecnico di Torino
nima.mohammadi@studenti.polito.it
