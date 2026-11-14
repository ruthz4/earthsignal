import argparse, glob, json, os, re
import numpy as np, pandas as pd, xarray as xr
from scipy.stats import norm

VARIABLES = {
    "DUEXTTAU": ("Dust AOD (550 nm)", "1", 1.0),
    "TOTEXTTAU": ("Total AOD (550 nm)", "1", 1.0),
    "DUSMASS": ("Dust surface mass concentration", "ug m-3", 1e9),
    "DUCMASS": ("Dust column mass density", "g m-2", 1e3),
    "DUFLUXU": ("Dust column zonal mass flux", "g m-1 s-1", 1e3),
}
SEASONS = {"ANN": None, "JJA": [6, 7, 8], "DJF": [12, 1, 2]}
START_YEARS = [1980, 2003]
THRESHOLDS = [2, 1]
MIN_YEARS_FOR_TOE = 10
REGIONS = {
    "sahara_source": {"name": "Saharan source region", "bounds": [15, 30, -15, 30]},
    "sahel": {"name": "Sahel", "bounds": [10, 18, -18, 25]},
    "atlantic_corridor": {"name": "Tropical Atlantic transport corridor", "bounds": [10, 25, -50, -20]},
    "caribbean": {"name": "Caribbean", "bounds": [10, 25, -90, -60]},
}

def _time_from_name(path):
    m = re.search(r"(19|20)\d{4}", os.path.basename(path))
    return pd.Timestamp(year=int(m.group()[:4]), month=int(m.group()[4:6]), day=15) if m else None

def load_monthly(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.nc*"), recursive=True))
    if not files: raise SystemExit(f"No NetCDF files found in {input_dir}")
    pieces = []
    for f in files:
        ds = xr.open_dataset(f)
        ds = ds[[v for v in VARIABLES if v in ds]]
        if "time" not in ds.dims:
            ds = ds.expand_dims(time=[_time_from_name(f)])
        pieces.append(ds.load()); ds.close()
    ds = xr.concat(pieces, dim="time").sortby("time")
    ds = ds.assign_coords(time=pd.to_datetime(ds.time.values).to_period("M").to_timestamp() + pd.Timedelta(days=14))
    _, idx = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=idx)
    for v, (_, units, scale) in VARIABLES.items():
        if v in ds:
            ds[v] = ds[v].astype("float64") * scale
            ds[v].attrs = {"long_name": VARIABLES[v][0], "units": units}
    print(f"Loaded {len(files)} files, {ds.time.size} months ({str(ds.time.values[0])[:7]} to {str(ds.time.values[-1])[:7]}), grid {ds.lat.size} x {ds.lon.size}")
    return ds

def seasonal_series(da, season):
    months, years = da.time.dt.month, da.time.dt.year
    if season == "ANN":
        counts = da.time.groupby(years).count()
        out = da.groupby(years).mean("time").where(counts == 12, drop=True)
    else:
        sel = da.where(months.isin(SEASONS[season]), drop=True)
        yr = sel.time.dt.year
        if season == "DJF": yr = xr.where(sel.time.dt.month == 12, yr + 1, yr)
        sel = sel.assign_coords(season_year=("time", yr.values))
        counts = sel.time.groupby(sel.season_year).count()
        out = sel.groupby("season_year").mean("time").where(counts == 3, drop=True).rename({"season_year": "year"})
    if "group" in out.dims: out = out.rename({"group": "year"})
    if "year" not in out.dims: out = out.rename({out.dims[0]: "year"})
    return out.transpose("year", ...)

def sens_slope(y, t):
    i, j = np.triu_indices(len(t), k=1)
    return np.nanmedian((y[j] - y[i]) / (t[j] - t[i]).reshape((-1,) + (1,) * (y.ndim - 1)), axis=0)

def ols_detrend(y, t):
    tc = t - t.mean()
    b = (tc[:, None] * (y - y.mean(0))).sum(0) / (tc ** 2).sum()
    return b, y - y.mean(0) - b * tc[:, None]

def mann_kendall_p(y, t):
    """Two-sided MK p-value, AR(1)-based variance correction inspired by Yue and Wang (2004)."""
    n = len(t); i, j = np.triu_indices(n, k=1)
    s = np.sign(y[j] - y[i]).sum(0)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    _, r = ols_detrend(y, t)
    r1 = np.clip((r[1:] * r[:-1]).sum(0) / np.maximum((r ** 2).sum(0), 1e-300), -0.99, 0.99)
    k = np.arange(1, n)
    factor = np.maximum(1 + 2 * ((n - k)[:, None] * (r1[None, :] ** k[:, None])).sum(0) / n, 1.0)
    var_s = var_s * factor
    z = np.where(s > 0, (s - 1) / np.sqrt(var_s), np.where(s < 0, (s + 1) / np.sqrt(var_s), 0.0))
    return 2 * (1 - norm.cdf(np.abs(z)))

def sn_curve(y, t, noise):
    sn = np.full(y.shape, np.nan)
    for e in range(MIN_YEARS_FOR_TOE - 1, len(t)):
        tt, yy = t[: e + 1], y[: e + 1]; tc = tt - tt.mean()
        b = (tc[:, None] * (yy - yy.mean(0))).sum(0) / (tc ** 2).sum()
        sn[e] = b * (tt[-1] - tt[0]) / noise
    return sn

def time_of_emergence(sn, years, k):
    """Retrospective ToE: first year after which S/N >= k with the SAME SIGN as the final signal holds every later year."""
    final_sign = np.sign(sn[-1])
    above = np.where(np.isnan(sn), False, sn * final_sign >= k)
    stays = np.flip(np.logical_and.accumulate(np.flip(above, 0), axis=0), 0) & ~np.isnan(sn)
    toe = years[np.argmax(stays, axis=0)].astype(float)
    toe[~stays.any(0)] = np.nan
    return toe

def bh_fdr(p):
    """Benjamini-Hochberg q-values; scope = all finite p in the array."""
    p = np.asarray(p, dtype=float); q = np.full(p.shape, np.nan)
    ok = np.isfinite(p); m = ok.sum()
    if m == 0: return q
    ps = p[ok]; order = np.argsort(ps)
    ranked = np.minimum.accumulate((ps[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[order] = np.clip(ranked, 0, 1); q[ok] = out
    return q

def analyse(y2d, years):
    t = years.astype(float); good = np.isfinite(y2d).all(0)
    out = {k: np.full(y2d.shape[1], np.nan) for k in ["slope", "pvalue", "qvalue", "noise"] + [f"toe_k{k}" for k in THRESHOLDS]}
    sn_all = np.full(y2d.shape, np.nan)
    if good.any():
        y = y2d[:, good]
        out["slope"][good] = sens_slope(y, t) * 10.0
        out["pvalue"][good] = mann_kendall_p(y, t)
        out["qvalue"] = bh_fdr(out["pvalue"])
        _, resid = ols_detrend(y, t)
        noise = resid.std(0, ddof=1); noise = np.where(noise > 0, noise, np.nan)
        out["noise"][good] = noise
        sn = sn_curve(y, t, noise); sn_all[:, good] = sn
        for k in THRESHOLDS: out[f"toe_k{k}"][good] = time_of_emergence(sn, years, k)
    return out, sn_all

def region_mean(da, b):
    sub = da.sel(lat=slice(b[0], b[1]), lon=slice(b[2], b[3]))
    return sub.weighted(np.cos(np.deg2rad(sub.lat))).mean(("lat", "lon"))

def _save(ds, path):
    ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})

def _clean(x):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), 6) for v in np.atleast_1d(x)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", default="results")
    args = ap.parse_args(); os.makedirs(args.output, exist_ok=True)
    ds = load_monthly(args.input)
    regions_out = {rid: {"name": r["name"], "bounds": r["bounds"], "data": {}} for rid, r in REGIONS.items()}
    for var in [v for v in VARIABLES if v in ds]:
        for season in SEASONS:
            ser = seasonal_series(ds[var], season); ser.name = var; ser.attrs = ds[var].attrs
            _save(ser.astype("float32").to_dataset(), os.path.join(args.output, f"series_{var}_{season}.nc"))
            for start in START_YEARS:
                s = ser.sel(year=slice(start, None)); years = s.year.values
                if len(years) < MIN_YEARS_FOR_TOE + 2: continue
                res, _ = analyse(s.values.reshape(len(years), -1), years)
                fields = xr.Dataset({k: (("lat", "lon"), v.reshape(s.shape[1:])) for k, v in res.items()}, coords={"lat": s.lat, "lon": s.lon})
                fields["slope"].attrs = {"units": f"{ds[var].attrs['units']} per decade", "method": "Sen's slope"}
                fields["pvalue"].attrs = {"method": "Mann-Kendall, AR(1)-based variance correction inspired by Yue and Wang (2004)"}
                fields["qvalue"].attrs = {"method": "Benjamini-Hochberg FDR across all valid grid cells of this map"}
                fields.attrs = {"variable": var, "season": season, "start_year": int(start), "end_year": int(years[-1]), "source": "MERRA-2 M2TMNXAER v5.12.4"}
                _save(fields.astype("float32"), os.path.join(args.output, f"fields_{var}_{season}_{start}.nc"))
                key = f"{var}_{season}_{start}"
                for rid, r in REGIONS.items():
                    rm = region_mean(s, r["bounds"]).values
                    rres, rsn = analyse(rm[:, None], years)
                    regions_out[rid]["data"][key] = {
                        "years": [int(y) for y in years], "values": _clean(rm), "sn": _clean(rsn[:, 0]),
                        "slope_per_decade": _clean(rres["slope"])[0], "pvalue": _clean(rres["pvalue"])[0],
                        "noise": _clean(rres["noise"])[0], **{f"toe_k{k}": _clean(rres[f"toe_k{k}"])[0] for k in THRESHOLDS}}
                rids = list(REGIONS)
                rq = bh_fdr([np.nan if regions_out[r]["data"][key]["pvalue"] is None else regions_out[r]["data"][key]["pvalue"] for r in rids])
                for r, qv in zip(rids, rq): regions_out[r]["data"][key]["qvalue"] = _clean(qv)[0]
                v_ = np.isfinite(res["pvalue"])
                print(f"  {var:9s} {season} from {start}: p<0.05 {(res['pvalue'][v_] < 0.05).mean()*100:5.1f}% | FDR q<0.05 {(res['qvalue'][v_] < 0.05).mean()*100:5.1f}% | emerged k=2 & q<0.05 {(np.isfinite(res['toe_k2']) & (res['qvalue'] < 0.05))[v_].mean()*100:5.1f}%")
    json.dump(regions_out, open(os.path.join(args.output, "regions.json"), "w"))
    json.dump({"variables": {v: {"long_name": VARIABLES[v][0], "units": VARIABLES[v][1]} for v in VARIABLES if v in ds},
               "seasons": list(SEASONS), "start_years": START_YEARS, "thresholds": THRESHOLDS,
               "regions": {rid: {"name": r["name"], "bounds_S_N_W_E": r["bounds"]} for rid, r in REGIONS.items()},
               "observing_system_break": 2000, "source": "NASA MERRA-2 M2TMNXAER v5.12.4 (GES DISC)"},
              open(os.path.join(args.output, "metadata.json"), "w"), indent=2)
    print(f"Done. Results in {args.output}/")

if __name__ == "__main__":
    main()
