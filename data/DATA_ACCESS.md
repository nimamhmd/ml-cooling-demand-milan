# Data Access

This file describes how to obtain the four datasets used in the thesis. None of the raw data is redistributed in this repository, in accordance with the terms of use of each data provider.

---

## 1. CENED+2 — Lombardy Regional Energy Performance Certificate Database

**Maintained by:** Regione Lombardia (Italy)
**Coverage in thesis:** Municipality of Milan, snapshot dated 28 November 2025
**File used:** `Database_CENED+2_-_Certificazione_ENergetica_degli_EDifici_20251128.csv`
**Volume at extraction:** 342,684 EPC certificates for the Municipality of Milan
**Volume after cadastral matching:** 19,063 buildings with at least one valid EPC label

**Access:**
Research access to CENED+2 microdata can be requested through the regional energy agency (AREXPO / Struttura Certificazione Energetica). The public aggregate dataset is available at:
https://www.dati.regione.lombardia.it/

For research access to the building-level microdata, contact:
Regione Lombardia — Struttura Certificazione Energetica

**Variables used in this thesis:**
Cooling-system presence (`climatizzazione_estiva`), energy-class label, total non-renewable primary-energy intensity, heating and cooling components, year of construction, gross heated floor area, EDIFC_ID matching key, and ancillary descriptive fields. The complete field list is documented in §3.6 of the thesis.

---

## 2. DBT2012 — Database Topografico of the Municipality of Milan

**Maintained by:** Comune di Milano (Italy)
**Coverage in thesis:** All residential building polygons within the municipal boundary
**Volume:** 53,041 residential building polygons

**Access:**
The DBT2012 vector layer is publicly available through the Geoportale del Comune di Milano:
https://geoportale.comune.milano.it/

The relevant layer is `EDIFC_DBT2012` (or its current equivalent). Each polygon carries the `EDIFC_ID` field used in this thesis as the matching key against the CENED+2 EPC database.

**Licence:**
The DBT2012 layer is distributed under the Italian *Licenza Italiana di Riuso* (IODL 2.0) by the Comune di Milano.

---

## 3. ERA5-Land — Climate Reanalysis (Historical Baseline)

**Maintained by:** Copernicus Climate Change Service (C3S) / European Centre for Medium-Range Weather Forecasts (ECMWF)
**Coverage in thesis:** 0.1° spatial resolution, hourly 2-metre air temperature, 1990–2024 (35 calendar years)
**Variable used:** `2m_temperature` (t2m)

**Access:**
ERA5-Land is freely available, after free registration, from the Copernicus Climate Data Store (CDS):
https://cds.climate.copernicus.eu/

The dataset identifier in the CDS catalogue is `reanalysis-era5-land`. The download is via the CDS API (`cdsapi` Python package) or the web interface. The thesis used the Python API to extract a regional subset around Milan.

**Licence:**
ERA5-Land is distributed under the Copernicus Open Licence. Acknowledgement of Copernicus in any publication is required.

---

## 4. NEX-GDDP-CMIP6 — Downscaled Climate Projections (Future Scenarios)

**Maintained by:** NASA Earth Exchange (NEX), in collaboration with the World Climate Research Programme (WCRP)
**Coverage in thesis:** 0.25° spatial resolution, daily values, 1990–2100, two scenarios (SSP2-4.5 and SSP5-8.5), 12 GCMs
**Variables used:** daily mean 2-metre air temperature (`tas`) and its daily maximum and minimum (`tasmax`, `tasmin`)

**Access:**
NEX-GDDP-CMIP6 is freely available through several mirrors:
- NASA Earth Exchange portal: https://www.nasa.gov/nex
- Amazon Open Data: https://registry.opendata.aws/nex-gddp-cmip6/
- Google Earth Engine: `ImageCollection("NASA/GDDP-CMIP6")`

This thesis used the Google Earth Engine interface for extraction. Authentication setup is documented in `04_climate_projection/00_setup_gee_authentication.py`.

**The 12 GCMs used in this thesis are enumerated in the climate-projection methodology section of the thesis (Chapter 3).**

**Licence:**
NEX-GDDP-CMIP6 is publicly available with no access restrictions. Acknowledgement of NASA NEX and the participating modelling centres in any publication is required.

---

## Reproducibility note

The data-preprocessing scripts in folder `01_data_preprocessing/` and the climate-extraction scripts in folder `04_climate_projection/` assume the raw datasets have been downloaded to a local working directory. The exact directory layout expected by the scripts is documented in their respective module docstrings.

For questions about data access or to request the cleaned analytical dataset for replication purposes, contact the author: nima.mohammadi@studenti.polito.it
