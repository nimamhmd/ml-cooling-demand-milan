# 07 — Decision Support

This folder is reserved for the decision-support layer of the thesis, which implements the ISO 31000:2018 risk-management process and the multi-criteria decision analysis (MCDA) applied to the projected cooling demand at the NIL-cluster scale, as described in §4.9 of the thesis.

## Current status

In the present version of the repository, **the code for this analysis is embedded inside** `03_phase2_regression/Step_10_Stage2_Cooling_Intensity.ipynb` rather than being maintained as a separate module. The notebook produces:

- The district-level risk register following ISO 31000:2018 likelihood-and-consequence scoring
- The multi-criteria treatment-evaluation matrix and weighting-scheme sensitivity
- The PDCA-aligned decision tables presented in §4.9 of the thesis

The reason for keeping these analyses inside the Phase 2 notebook is that the MCDA matrices depend directly on the projected demand quantities and per-cluster aggregations produced by the Phase 2 stacking ensemble, so co-locating them avoided an additional intermediate-data layer for the thesis work.

## Future direction

A modular rewrite that extracts the decision-support functions into standalone scripts (provisionally `07a_risk_register.py`, `07b_mcda_evaluation.py`) is one of the future-research directions named in §5.7 of the thesis. If undertaken, the rewritten scripts will be placed in this folder and `RUN_ORDER.md` will be updated accordingly.

For now, to reproduce the decision-support outputs of §4.9, run `03_phase2_regression/Step_10_Stage2_Cooling_Intensity.ipynb` end-to-end.
