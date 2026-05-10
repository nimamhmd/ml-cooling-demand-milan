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

This repository contains all Python code developed for the MSc thesis listed above. The thesis builds a two-stage explainable machine-learning pipeline that:

1. **Phase 1 — Cooling Presence Classification.** Predicts whether each residential building in Milan has an active summer cooling system, using a stacking ensemble of four classifiers (Logistic Regression, Random Forest, XGBoost, LightGBM) trained on 19,063 Energy Performance Certificate (EPC) records merged with cadastral geometry and ERA5-Land climate reanalysis data.

2. **Phase 2 — Cooling Intensity Regression.** Predicts a per-building climate-normalised cooling-intensity coefficient (β) using a four-learner stacking ensemble (XGBoost, LightGBM, CatBoost, MLP). The coefficient is then forward-propagated through a 12-model CMIP6 climate ensemble to produce city-wide residential cooling demand projections for Milan under SSP2-4.5 and SSP5-8.5 through 2100.

Both stages include a full explainability audit using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations). Uncertainty in the projections is decomposed using the Sobol-Saltelli variance-decomposition method. Results are aggregated to 88 NIL-scale spatial clusters for municipal planning use, and a decision-support demonstration is provided following ISO 31000:2018 and ISO 50001:2018 process standards.

---

## Key Results

| Result | Value |
|---|---|
| Phase 1 held-out ROC-AUC | 0.972 |
| Phase 1 spatial cross-validation gap | 0.003 ROC-AUC points |
| Phase 2 held-out R² (log1p-β scale) | 0.689 [95% CI: 0.644, 0.731] |
| City-wide cooling demand, ERA5 baseline | 38.6 GWh yr⁻¹ |
| City-wide cooling demand, SSP5-8.5 / 2080–2100 (V1) | 139.5 GWh yr⁻¹ [93.0, 180.9] |
| Percentage increase, SSP5-8.5 / 2080–2100 | +261.8% |
| Dominant uncertainty source (Sobol S_T, V1) | Climate-model identity: 94.3% |
| Buildings in top-15 high-demand clusters | 29.2% of cooled stock, 35.2% of projected demand |

---

## Repository Structure

```
ml-cooling-demand-milan/
│
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies
├── LICENSE                          ← MIT Licence
│
├── 01_data_preprocessing/
│   ├── 01a_epc_cleaning.py          ← CENED+2 EPC data cleaning and field selection
│   ├── 01b_cadastre_matching.py     ← EDIFC_ID-based EPC-to-cadastre matching
│   ├── 01c_era5_extraction.py       ← ERA5-Land CDD22 computation (1990–2024)
│   └── 01d_feature_engineering.py  ← Interaction terms, heating_fraction proxy, log transforms
│
├── 02_phase1_classification/
│   ├── 02a_baseline_classifiers.py  ← Logistic Regression and Random Forest training
│   ├── 02b_xgboost_lgbm.py          ← XGBoost and LightGBM training
│   ├── 02c_stacking_ensemble.py     ← Phase 1 stacking ensemble and evaluation
│   ├── 02d_spatial_cv.py            ← Geographic cluster cross-validation diagnostic
│   └── 02e_explainability.py        ← SHAP and LIME audit for Phase 1
│
├── 03_phase2_regression/
│   ├── 03a_beta_construction.py     ← Per-building β coefficient construction (Eq. 3.4)
│   ├── 03b_regression_models.py     ← XGBoost, LightGBM, CatBoost, MLP training
│   ├── 03c_stacking_ensemble.py     ← Phase 2 stacking ensemble and evaluation
│   ├── 03d_functional_form.py       ← Linear vs log vs polynomial CDD robustness check
│   └── 03e_explainability.py        ← SHAP and LIME audit for Phase 2
│
├── 04_climate_projection/
│   ├── 04a_cmip6_download.py        ← CMIP6 ensemble retrieval (NEX-GDDP-CMIP6)
│   ├── 04b_cdd_computation.py       ← CDD22 computation for all 12 models × 5 scenarios
│   ├── 04c_demand_propagation.py    ← V1, V2, V3 demand variant computation
│   └── 04d_bootstrap_intervals.py   ← Bootstrap 95% CI for projected demand
│
├── 05_uncertainty_sobol/
│   ├── 05a_sobol_decomposition.py   ← Sobol-Saltelli total-order indices (V1 and V3)
│   └── 05b_sensitivity_analysis.py  ← Sensitivity to functional form and adoption rate
│
├── 06_spatial_analysis/
│   ├── 06a_nil_aggregation.py       ← K-means clustering to 88 NIL-scale spatial clusters
│   ├── 06b_vulnerability_maps.py    ← Per-building and district-level vulnerability maps
│   └── 06c_cohort_decomposition.py  ← Demand decomposition by construction era and energy class
│
├── 07_decision_support/
│   ├── 07a_risk_register.py         ← ISO 31000 district-level risk register construction
│   └── 07b_mcda_evaluation.py       ← Multi-criteria treatment evaluation and ranking
│
├── 08_visualisation/
│   └── figures.py                   ← All figures appearing in the thesis (Figures 3.1–5.2)
│
└── data/
    └── DATA_ACCESS.md               ← How to obtain the datasets used in this thesis
```

---

## Data Availability

The raw datasets used in this thesis cannot be redistributed in this repository. Access instructions for each source are as follows.

**CENED+2 — Lombardy Regional EPC Database**
The CENED+2 database is maintained by Regione Lombardia. Research access can be requested through the regional energy agency (AREXPO / Struttura Certificazione Energetica). The thesis used a snapshot extracted in 2024 covering 19,063 residential buildings in the Municipality of Milan.

**DBT2012 — Milan Building Cadastre**
The Database Topografico (DBT2012) building footprint layer is maintained by the Comune di Milano and is available through the Geoportale del Comune di Milano (geoportale.comune.milano.it). The EDIFC_ID field was used to match EPC records to cadastral polygons.

**ERA5-Land — Climate Reanalysis**
ERA5-Land hourly 2-metre air temperature data (0.1° resolution, 1990–2024) is publicly available through the Copernicus Climate Change Service (C3S) Climate Data Store at cds.climate.copernicus.eu. No access restrictions apply.

**NEX-GDDP-CMIP6 — CMIP6 Downscaled Climate Projections**
The NASA Earth Exchange Global Daily Downscaled Projections (NEX-GDDP-CMIP6) dataset is publicly available through NASA at www.nasa.gov/nex. The 12 GCMs used in this thesis are listed in Table 3.6 of the thesis document.

---

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/YourUsername/ml-cooling-demand-milan.git
cd ml-cooling-demand-milan
pip install -r requirements.txt
```

Python 3.10 or higher is recommended. All scripts were developed and tested on Python 3.11.

---

## Requirements

The full list of dependencies is in `requirements.txt`. Key packages:

```
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
torch
shap
lime
scipy
matplotlib
seaborn
geopandas
rasterio
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
