# EarthSignal: The Invisible Dust Signal

**NASA Space Apps Challenge 2026 · Be an Earth System Trend Detective!**

> A trend can be real long before it becomes detectable. EarthSignal shows when and where changes in Saharan dust rise above natural variability.

**Live app:** https://earthsignal-dust1234.streamlit.app/

## Research question

When and where do long-term changes in Saharan dust transport become detectable above background variability?

## What EarthSignal does

EarthSignal is an interactive web app that estimates the **Time of Emergence (ToE)** of dust trends across North Africa and the tropical Atlantic. Users can:

* Choose a dust variable, season (annual, JJA, DJF) and analysis start year (1980 or 2003)
* Switch between three map layers: **Estimated ToE**, **Projected ToE** and **Trend**
* Change the signal-to-noise threshold (k = 2 main, k = 1 exploratory)
* Turn the False Discovery Rate (FDR) filter on or off
* Track how much of the study area has emerged by a chosen year
* Inspect a preset region or any single grid point, with its time series, signal vs. noise curve and detection status

## Data

* **NASA MERRA-2** monthly aerosol diagnostics, M2TMNXAER v5.12.4
  * Dataset page (NASA GES DISC): https://disc.gsfc.nasa.gov/datasets/M2TMNXAER_5.12.4/summary
  * Dataset DOI: https://doi.org/10.5067/FH9A0MLJPC7N
  * About MERRA-2 (NASA GMAO): https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
  * NASA Earthdata login: https://urs.earthdata.nasa.gov
* Variables: dust AOD (DUEXTTAU), total AOD (TOTEXTTAU), dust surface concentration (DUSMASS), dust column mass (DUCMASS), dust zonal flux (DUFLUXU)
* Domain: 0 to 40°N, 100°W to 40°E
* Period: January 1980 to September 2026
* Accessed through NASA Earthdata with [earthaccess](https://github.com/nsidc/earthaccess) and cloud OPeNDAP

MERRA-2 assimilates satellite aerosol observations (AVHRR, MODIS, MISR) and ground based AERONET data. Because the observing system changed around 2000, the app offers two start years: **1980** (full record) and **2003** (modern satellite era).

## Method

1. **Seasonal means:** annual, summer (JJA) and winter (DJF, labelled by its January year).
2. **Trend:** Sen's slope.
3. **Significance:** Mann-Kendall test with an AR(1) based variance correction inspired by Yue and Wang (2004).
4. **Multiple testing:** Benjamini-Hochberg FDR correction across all valid grid cells of each map, and separately across the four preset regions.
5. **Noise (σ):** standard deviation of the detrended series.
6. **Signal:** S(t) = trend slope from the start year to year t, multiplied by the years elapsed.
7. **Time of Emergence:** the first year in which SNR(t) = |S(t)| / σ ≥ k holds and stays true to the end of the record. ToE is estimated retrospectively and requires at least 10 years of data.
8. **Projected ToE:** t0 + k·σ / |slope per year|. This is a conditional extrapolation if the current trend continues, not a forecast. It is labelled low confidence when the trend is not significant.

## Key findings

Dust AOD, annual mean, 2003 to 2025, k = 2, FDR q < 0.05:

| Region | Result |
|---|---|
| Sahara source | Decline emerged in **2019** (summer: 2022) |
| Sahel | Significant decline, only just emerging (2025) |
| Atlantic corridor | Trend not significant; projected around 2038 (low confidence) |
| Caribbean | Trend not significant; projected around 2079 (low confidence) |

The decline is detectable at the source first, while downstream it remains hidden in natural variability.

Results also depend strongly on the analysis period. Using annual dust AOD with k = 2 and FDR, 41.4% of the study area has emerged since 1980, compared with 5.5% since 2003. The two windows differ in length and observing history, so this difference should not be read as evidence that the physical trend weakened.

## Repository structure

~~~
app.py                     Streamlit web app
earthsignal_analysis.py    Trend, significance, FDR and ToE calculations
download_merra2.py         Subsets and downloads MERRA-2 data from NASA Earthdata
results/                   Precomputed fields, regional series and metadata used by the app
requirements.txt           Python dependencies for the app
.streamlit/config.toml     App theme
~~~

## Run locally

~~~
pip install -r requirements.txt
streamlit run app.py
~~~

## Reproduce the analysis

You need a free NASA Earthdata account with the GES DISC application approved.

~~~
pip install earthaccess
python download_merra2.py
python earthsignal_analysis.py
~~~

The first script downloads subsets to merra2/ and the second writes results/.

## Limitations

* MERRA-2 is a reanalysis, so its dust fields depend on both the model and the assimilated observations.
* ToE is retrospective: a 2019 ToE means the signal is detectable in hindsight using data through 2025, not that it could have been confirmed in real time in 2019.
* Projected ToE assumes a linear trend continues and is not a forecast.
* Results are sensitive to start year, season, threshold k and the FDR filter. The app is designed to make this sensitivity visible.

## Team

* **Ruth Zhou**, UC Irvine: data pipeline, analysis, app development
* **Nicholas Chua Minor**: [ROLE]

## References

* Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57, 289 to 300.
* Gelaro, R., et al. (2017). The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). *Journal of Climate*, 30, 5419 to 5454. https://doi.org/10.1175/JCLI-D-16-0758.1
* Global Modeling and Assimilation Office (2015). MERRA-2 tavgM_2d_aer_Nx (M2TMNXAER), version 5.12.4. NASA GES DISC. https://doi.org/10.5067/FH9A0MLJPC7N
* Hawkins, E., and Sutton, R. (2012). Time of emergence of climate signals. *Geophysical Research Letters*, 39, L01702. https://doi.org/10.1029/2011GL050087
* Randles, C. A., et al. (2017). The MERRA-2 aerosol reanalysis, 1980 onward. *Journal of Climate*, 30, 6823 to 6850. https://doi.org/10.1175/JCLI-D-16-0609.1
* Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association*, 63, 1379 to 1389.
* Yue, S., and Wang, C. (2004). The Mann-Kendall test modified by effective sample size to detect trend in serially correlated hydrological data series. *Water Resources Management*, 18, 201 to 218.

## Acknowledgments

Data provided by NASA's Global Modeling and Assimilation Office and the Goddard Earth Sciences Data and Information Services Center (GES DISC).
