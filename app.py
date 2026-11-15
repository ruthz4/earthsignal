import json, os
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from scipy.stats import norm

st.set_page_config(page_title="EarthSignal", page_icon="🌍", layout="wide")
DATA_DIR = next((d for d in [os.environ.get("EARTHSIGNAL_DATA", ""), "results", "mock_results"] if d and os.path.isdir(d)), None)
if DATA_DIR is None:
    st.error("No results folder found. Put the analysis output in ./results."); st.stop()

VAR_LABELS = {"DUEXTTAU": "Dust AOD (550 nm)", "TOTEXTTAU": "Total AOD (550 nm)",
              "DUSMASS": "Dust surface concentration", "DUCMASS": "Dust column mass",
              "DUFLUXU": "Dust zonal flux (negative = westward)"}
SEASON_LABELS = {"JJA": "Summer (JJA)", "DJF": "Winter (DJF)", "ANN": "Annual mean"}
START_LABELS = {1980: "1980 (full record)", 2003: "2003 (consistent satellite era)"}
BREAK_YEAR, HORIZON = 2000, 2100

@st.cache_data
def load_meta():
    return json.load(open(os.path.join(DATA_DIR, "metadata.json"))), json.load(open(os.path.join(DATA_DIR, "regions.json")))

@st.cache_data
def load_fields(var, season, start):
    ds = xr.open_dataset(os.path.join(DATA_DIR, f"fields_{var}_{season}_{start}.nc")).load()
    return {k: ds[k].values for k in ds.data_vars}, ds.lat.values, ds.lon.values, int(ds.attrs.get("end_year", 2025))

@st.cache_data
def load_point_series(var, season, lat, lon):
    da = xr.open_dataset(os.path.join(DATA_DIR, f"series_{var}_{season}.nc"))[var]
    p = da.sel(lat=lat, lon=lon, method="nearest").load()
    return p.year.values, p.values.astype(float), float(p.lat), float(p.lon)

def projected_toe(start, slope_per_decade, noise, k):
    b = np.abs(np.asarray(slope_per_decade, dtype=float)) / 10.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return start + k * np.asarray(noise, dtype=float) / b

def series_stats(years, y, k_list=(1, 2), min_years=10):
    t = years.astype(float); n = len(t); i, j = np.triu_indices(n, k=1)
    slope = np.median((y[j] - y[i]) / (t[j] - t[i]))
    tc = t - t.mean(); b = (tc * (y - y.mean())).sum() / (tc ** 2).sum()
    resid = y - y.mean() - b * tc; noise = resid.std(ddof=1)
    s = np.sign(y[j] - y[i]).sum()
    r1 = np.clip((resid[1:] * resid[:-1]).sum() / (resid ** 2).sum(), -0.99, 0.99)
    kk = np.arange(1, n)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0 * max(1 + 2 * ((n - kk) * r1 ** kk).sum() / n, 1.0)
    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    sn = np.full(n, np.nan)
    for e in range(min_years - 1, n):
        tt, yy = t[: e + 1], y[: e + 1]; c = tt - tt.mean()
        sn[e] = (c * (yy - yy.mean())).sum() / (c ** 2).sum() * (tt[-1] - tt[0]) / noise
    toe = {}
    fs = np.sign(sn[~np.isnan(sn)][-1]) if (~np.isnan(sn)).any() else 0
    for k in k_list:
        above = np.where(np.isnan(sn), False, sn * fs >= k)
        stays = np.flip(np.logical_and.accumulate(np.flip(above))) & ~np.isnan(sn)
        toe[k] = int(years[np.argmax(stays)]) if stays.any() else None
    return {"slope_per_decade": slope * 10, "pvalue": p, "noise": noise, "sn": sn, "toe": toe}

meta, regions = load_meta()
var_units = {v: (meta["variables"][v]["units"] if meta["variables"][v]["units"] != "1" else "unitless") for v in meta["variables"]}

st.sidebar.title("EarthSignal")
st.sidebar.caption("When does Saharan dust change become detectable?")
var = st.sidebar.selectbox("Variable", list(meta["variables"]), format_func=lambda v: VAR_LABELS.get(v, v))
season = st.sidebar.radio("Season", ["JJA", "DJF", "ANN"], format_func=lambda s: SEASON_LABELS[s])
start = st.sidebar.radio("Analysis start year", meta["start_years"], format_func=lambda s: START_LABELS.get(s, str(s)),
    help="MERRA-2 starts in 1980, but began assimilating MODIS aerosol observations in 2000 (Terra) and 2002 (Aqua). 2003 is the first full year with a consistent satellite observing system.")
layer = st.sidebar.radio("Map layer", ["Estimated ToE", "Projected ToE", "Trend"])
k = st.sidebar.radio("Signal-to-noise threshold (k)", sorted(meta["thresholds"], reverse=True), horizontal=True,
    format_func=lambda x: f"{x} (main)" if x == 2 else f"{x} (exploratory)",
    help="ToE is the first year when trend signal / natural variability reaches k (same sign as the final signal) and stays above it. Main findings use k = 2.")
only_sig = st.sidebar.checkbox("Only show where trend passes FDR (q < 0.05)", value=True,
    help="Mann-Kendall p-values are corrected for testing thousands of grid cells at once (Benjamini-Hochberg false discovery rate).")
fields, lat, lon, end_year = load_fields(var, season, start)
if layer != "Trend":
    hi = HORIZON if layer == "Projected ToE" else end_year
    asof = st.sidebar.slider("Emerged by year", start + 10, hi, hi, help="Drag to see where the signal had (or would have) emerged by a given year.")
st.sidebar.divider()
mode = st.sidebar.radio("Time series for", ["Preset region", "Single grid point"])
if mode == "Preset region":
    rid = st.sidebar.selectbox("Region", list(regions), format_func=lambda r: regions[r]["name"])
else:
    plat = st.sidebar.slider("Latitude (°N)", 0.0, 40.0, 15.0, 0.5)
    plon = st.sidebar.slider("Longitude (°E, west negative)", -100.0, 40.0, -30.0, 0.625)

hc1, hc2 = st.columns([3, 1], vertical_alignment="center")
with hc1:
    st.title("EarthSignal: The Invisible Dust Signal")
    st.markdown("*A trend can be real long before it becomes detectable. EarthSignal shows when and where changes in Saharan dust rise above natural variability.*")
with hc2:
    if os.path.exists("mascot.svg"):
        st.html('<div style="display:flex;justify-content:center">' + open("mascot.svg").read() + '</div>')
with st.expander("🌍 Before you explore: Why does dust matter?", expanded=False):
    st.markdown("""
Every year, winds carry Saharan dust thousands of kilometers across the Atlantic, sometimes reaching the Caribbean and the Amazon. Along the way, dust affects air quality, ecosystems, and even how sunlight interacts with the atmosphere.

But here's something we find fascinating: **dust doesn't behave the same way every year.**

Some years are much dustier than others. So when we see dust gradually increasing or decreasing over decades, how can we tell whether it's a long-term change or just the usual ups and downs?

That's the question behind **EarthSignal**.

### When does a change become detectable?

Imagine trying to hear someone speaking in a noisy room. At first, their voice gets lost in the background. But as it grows louder, you can finally distinguish it from the noise.

Scientists call a related concept **Time of Emergence (ToE)**.

EarthSignal estimates when a long-term change in dust becomes large enough to stand out from natural year-to-year variability, and remains above that threshold through the end of the record.

Because we look back at the full record, the estimated emergence year is not necessarily the year scientists could have detected the change in real time.

### Try exploring

1. **Start in the Sahara.** Select *Annual mean*, *2003*, and *Saharan source region*. Our analysis estimates that a declining dust signal emerged around 2019.
2. **Follow the dust.** Switch to the *Caribbean*. Is the change equally detectable farther from the source?
3. **Change the starting year.** Compare *1980* and *2003*. Notice how the results change when you use a different observation period.

### Reading the map

- **Brown → yellow:** Earlier to more recent estimated emergence years.
- **Dark gray:** Significant trend, but no detected emergence.
- **Light gray:** No significant trend.
- Use the **Emerged by year** slider to explore retrospectively estimated emergence years.

**A few terms:** *AOD (aerosol optical depth)* measures how much sunlight is blocked by particles in the air; higher means dustier. *FDR* is a correction that avoids false alarms when thousands of map cells are tested at once. *k* is how many times larger than normal year-to-year variability a change must be to count as emerged (our main results use k = 2).

**One thing to remember:** not detecting a change doesn't mean nothing is changing. Sometimes the signal is simply too small compared with the background variability to distinguish confidently.
""")

st.markdown(f"**{VAR_LABELS.get(var, var)}**, {SEASON_LABELS[season]}, from **{start}** to {end_year} · NASA MERRA-2 reanalysis (M2TMNXAER), 0–40°N, 100°W–40°E")

step = 2
LON, LAT = np.meshgrid(lon[::step], lat[::step]); sl = (slice(None, None, step), slice(None, None, step))
qfull = fields.get("qvalue", fields["pvalue"])
pval, qval, slope_f, noise_f = fields["pvalue"][sl], qfull[sl], fields["slope"][sl], fields["noise"][sl]
valid = np.isfinite(slope_f)
proj_flag = np.zeros_like(valid)
if layer == "Trend":
    val = slope_f; lim = np.nanpercentile(np.abs(val), 98) or 1
    colors = dict(colorscale="RdBu_r", cmin=-lim, cmax=lim, colorbar=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top", len=0.6, thickness=14, title_side="top", title=f"{var_units[var]}<br>per decade"))
    mask = valid
    hover = [f"{la:.1f}°N {lo:.1f}°E<br>trend {v:.3g}/decade<br>p = {p:.3f}, q = {q:.3f}"
             for la, lo, v, p, q in zip(LAT[mask], LON[mask], val[mask], pval[mask], qval[mask])]
else:
    obs = fields[f"toe_k{k}"][sl].copy()
    if layer == "Projected ToE":
        proj = projected_toe(start, slope_f, noise_f, k)
        proj_flag = ~np.isfinite(obs) & np.isfinite(proj) & (proj > end_year) & ((qval < 0.05) | (not only_sig))
        val = np.where(np.isfinite(obs), obs, np.where(proj_flag, np.ceil(proj), np.nan)); cmax = HORIZON
    else:
        val = obs; cmax = end_year
    if only_sig:
        val = np.where(qval < 0.05, val, np.nan)
    val = np.where(val <= asof, val, np.nan)
    colors = dict(colorscale=[[0, "#5B2C06"], [0.4, "#C8650E"], [0.75, "#F2A51A"], [1, "#FFD43B"]], cmin=start + 10, cmax=cmax, colorbar=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top", len=0.6, thickness=14, title_side="top", title="Time of Emergence (year)", tickvals=[t for t in range(int(np.ceil((start + 10) / 5) * 5), cmax + 1, 5 if cmax <= end_year else 10) if cmax - t >= 2] + [cmax]))
    mask = np.isfinite(val)
    hover = [f"{la:.1f}°N {lo:.1f}°E<br>{'conditional projection' if pf else 'emerged'} {int(v)}<br>p = {p:.3f}, q = {q:.3f}"
             for la, lo, v, p, q, pf in zip(LAT[mask], LON[mask], val[mask], pval[mask], qval[mask], proj_flag[mask])]

fig = go.Figure()
if layer != "Trend":
    nm = ~mask & valid
    ns = (nm & (qval >= 0.05)) if only_sig else np.zeros_like(nm)
    ny = nm & ~ns
    fig.add_trace(go.Scattergeo(lon=LON[ns], lat=LAT[ns], mode="markers", name="Trend not significant (FDR q ≥ 0.05)",
                                marker=dict(size=6, symbol="square", color="#cfd5dd"), hoverinfo="skip"))
    fig.add_trace(go.Scattergeo(lon=LON[ny], lat=LAT[ny], mode="markers",
                                name=(f"Significant trend, not emerged by {asof}" if only_sig else f"Not emerged by {asof}"),
                                marker=dict(size=6, symbol="square", color="#7f8ea3"), hoverinfo="skip"))
fig.add_trace(go.Scattergeo(lon=LON[mask], lat=LAT[mask], mode="markers", showlegend=False,
                            marker=dict(size=6, symbol="square", color=val[mask], **colors), text=hover, hoverinfo="text"))
if layer == "Projected ToE":
    pm = mask & proj_flag
    fig.add_trace(go.Scattergeo(lon=LON[pm][::2], lat=LAT[pm][::2], mode="markers", name=f"Conditional projection (after {end_year})",
                                marker=dict(size=2, color="#1f3f6e"), hoverinfo="skip"))
if layer == "Trend":
    sigm = mask & (qval < 0.05)
    fig.add_trace(go.Scattergeo(lon=LON[sigm][::3], lat=LAT[sigm][::3], mode="markers", name="FDR q < 0.05",
                                marker=dict(size=2, color="black"), hoverinfo="skip"))
if mode == "Preset region":
    s_, n_, w_, e_ = regions[rid]["bounds"]
    bx = np.r_[np.linspace(w_, e_, 30), np.full(30, e_), np.linspace(e_, w_, 30), np.full(30, w_)]
    by = np.r_[np.full(30, s_), np.linspace(s_, n_, 30), np.full(30, n_), np.linspace(n_, s_, 30)]
    fig.add_trace(go.Scattergeo(lon=bx, lat=by, mode="lines", line=dict(color="black", width=2), name=regions[rid]["name"]))
else:
    fig.add_trace(go.Scattergeo(lon=[plon], lat=[plat], mode="markers", name="Selected point", marker=dict(size=12, color="black", symbol="x")))
fig.update_geos(lataxis_range=[0, 40], lonaxis_range=[-100, 40], showcoastlines=True, coastlinecolor="#444",
                showland=True, landcolor="#f4f1ea", showocean=True, oceancolor="#dcEEF8",
                projection_type="equirectangular", fitbounds="locations")
fig.update_layout(height=560, margin=dict(l=0, r=0, t=40, b=70), legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", itemsizing="constant", font=dict(size=13)))
st.plotly_chart(fig, width="stretch")
if layer == "Projected ToE":
    st.caption(f"Cells after {end_year} (marked with dots) are **conditional projections, not forecasts**: shown only where the trend passes FDR (q < 0.05), "
               "assuming it continues at the same rate and natural variability stays the same. Dark gray: significant trend not emerged by the selected year; light gray: trend not significant after FDR; blank: no data. Uncheck the FDR box to also see low-confidence projections from non-significant trends.")
elif layer == "Estimated ToE":
    st.caption("Estimated Time of Emergence under the selected detection criterion. "
               + ("Dark gray: significant trend that has not emerged by the selected year; light gray: trend not significant after FDR (q ≥ 0.05); blank: no data."
                  if only_sig else "Gray: not emerged by the selected year (significance not required); blank: no data."))

if layer != "Trend":
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], qfull.shape)
    ok_cells = np.isfinite(fields["slope"])
    toe_full = fields[f"toe_k{k}"]
    passes = (qfull < 0.05) if only_sig else np.ones_like(ok_cells)
    yrs_c = np.arange(start + 10, (HORIZON if layer == "Projected ToE" else end_year) + 1)
    tot = w[ok_cells].sum()
    obs_frac = [100 * w[ok_cells & passes & (toe_full <= yy)].sum() / tot for yy in yrs_c]
    cfig = go.Figure()
    cfig.add_trace(go.Scatter(x=yrs_c[yrs_c <= end_year], y=np.array(obs_frac)[yrs_c <= end_year], mode="lines",
                              line=dict(color="#C98A0B", width=2.5), name="Emerged (observed)"))
    if layer == "Projected ToE":
        pfull = projected_toe(start, fields["slope"], fields["noise"], k)
        pyear = np.where(np.isfinite(toe_full), toe_full, np.where(((qfull < 0.05) | (not only_sig)) & (pfull > end_year), np.ceil(pfull), np.nan))
        proj_frac = [100 * w[ok_cells & passes & (pyear <= yy)].sum() / tot for yy in yrs_c]
        fut = yrs_c >= end_year
        cfig.add_trace(go.Scatter(x=yrs_c[fut], y=np.array(proj_frac)[fut], mode="lines",
                                  line=dict(color="#C98A0B", width=2.5, dash="dash"), name="If significant trends continue"))
    cfig.add_vline(x=asof, line_color="#3B82C4", line_width=2, annotation_text=str(asof), annotation_position="top")
    cfig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="% of area emerged",
                       xaxis_title="year", legend=dict(orientation="h", y=1.15), yaxis=dict(rangemode="tozero"))
    st.markdown(f"**Share of the study area where the change has emerged (k = {k}{', FDR q < 0.05' if only_sig else ''})** · the blue line follows the *Emerged by year* slider")
    st.plotly_chart(cfig, width="stretch")

if mode == "Preset region":
    d = regions[rid]["data"][f"{var}_{season}_{start}"]
    years = np.array(d["years"]); y = np.array([np.nan if v is None else v for v in d["values"]], dtype=float)
    place = regions[rid]["name"]; q_here = d.get("qvalue", d.get("pvalue"))
else:
    yrs, vals, glat, glon = load_point_series(var, season, plat, plon)
    keep = yrs >= start; years, y = yrs[keep], vals[keep]
    place = f"Grid point {glat:.1f}°N, {glon:.2f}°E"
    q_here = float(qfull[np.abs(lat - glat).argmin(), np.abs(lon - glon).argmin()])
ok = np.isfinite(y)
stats = series_stats(years[ok], y[ok]) if ok.sum() > 12 else None
if stats:
    if q_here is None or not np.isfinite(q_here):
        q_here = stats["pvalue"]
    sig = q_here < 0.05
    toe_year = stats["toe"][k]
    shown_toe = toe_year if (toe_year and (sig or not only_sig)) else None
    pj = float(projected_toe(start, stats["slope_per_decade"], stats["noise"], k))
    proj_year = int(np.ceil(pj)) if (shown_toe is None and np.isfinite(pj) and pj > years[-1]) else None

c1, c2 = st.columns([1.6, 1])
with c1:
    st.subheader(place)
    ts = go.Figure()
    if stats:
        x_end = min(max(proj_year or years[-1], years[-1]), HORIZON)
        xs = np.arange(years[0], x_end + 1)
        trend = np.nanmedian(y) + stats["slope_per_decade"] / 10 * (xs - np.median(years))
        obs_part = xs <= years[-1]
        ts.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[trend + stats["noise"], (trend - stats["noise"])[::-1]],
                                fill="toself", fillcolor="rgba(245,205,90,0.25)", line=dict(width=0), name="± natural variability (1σ)", hoverinfo="skip"))
        ts.add_trace(go.Scatter(x=xs[obs_part], y=trend[obs_part], mode="lines", line=dict(color="#C98A0B", width=2.5), name="Trend (Sen's slope)"))
        if (~obs_part).any():
            ts.add_trace(go.Scatter(x=xs[~obs_part], y=trend[~obs_part], mode="lines", line=dict(color="#C98A0B", width=2.5, dash="dash"), name="Trend continued (conditional projection)"))
    ts.add_trace(go.Scatter(x=years, y=y, mode="lines+markers", line=dict(color="#5F5E5A", width=1), marker=dict(size=4), name=SEASON_LABELS[season]))
    if start < BREAK_YEAR:
        ts.add_vline(x=BREAK_YEAR, line_dash="dot", line_color="gray", annotation_text="obs. system change", annotation_position="top left")
    if stats and shown_toe:
        ts.add_vline(x=shown_toe, line_color="#D85A30", line_width=2, annotation_text=f"ToE {shown_toe}", annotation_position="top right")
    if stats and proj_year and proj_year <= HORIZON:
        ts.add_vline(x=proj_year, line_color="#D85A30", line_width=2, line_dash="dash", annotation_text=(f"conditional projection {proj_year}" if sig else f"low-confidence projection {proj_year}"), annotation_position="top left")
    ts.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=var_units[var], legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(ts, width="stretch")
with c2:
    st.subheader("Signal vs. noise")
    if stats:
        sn = go.Figure()
        sn.add_trace(go.Scatter(x=years[ok], y=stats["sn"], mode="lines", line=dict(color="#534AB7", width=2)))
        for sgn in (1, -1): sn.add_hline(y=sgn * k, line_dash="dot", line_color="#D85A30")
        sn.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, yaxis_title="signal / noise", xaxis_title="end year of trend window")
        st.plotly_chart(sn, width="stretch")

if stats:
    rowA, rowB = st.columns(3), st.columns(3)
    m1, m2, m3 = rowA; m4, m5 = rowB[0], rowB[1]
    m1.metric("Trend per decade", f"{stats['slope_per_decade']:+.3g}", help=var_units[var])
    m2.metric("p (raw) / q (FDR)", f"{stats['pvalue']:.3f} / {q_here:.3f}",
              help="Raw Mann-Kendall p-value and Benjamini-Hochberg FDR-adjusted q-value (scope: all grid cells of this map, or the 4 preset regions).")
    m3.metric("Natural variability (1σ)", f"{stats['noise']:.3g}")
    if shown_toe:
        m4.metric(f"Detection status (k={k})", f"Emerged in {shown_toe}",
                  help="S/N reached k with the same sign as the final signal and stayed there to the end of the record, and the trend passes FDR (q < 0.05).")
    elif sig:
        m4.metric(f"Detection status (k={k})", "Significant, not yet emerged",
                  help="The trend passes FDR, but the signal has not yet stayed above k times natural variability.")
        if proj_year:
            m5.metric("Expected emergence", f"~{proj_year}" if proj_year <= HORIZON else f"Beyond {HORIZON}",
                      help="Conditional projection: the year S/N would reach k if the current trend continued unchanged. Not a forecast.")
    else:
        m4.metric(f"Detection status (k={k})", "No significant trend",
                  help="The trend does not pass FDR (q >= 0.05): any change is still hidden in natural variability.")
        if proj_year:
            m5.metric("Expected emergence (low confidence)", f"~{proj_year}" if proj_year <= HORIZON else f"Beyond {HORIZON}",
                      help="Year the signal would reach k if the weak, non-significant trend continued unchanged. Not statistically supported; shown for exploration only.")
    label = VAR_LABELS.get(var, var); direction = "increased" if stats["slope_per_decade"] > 0 else "decreased"
    if sig and shown_toe:
        msg = f"Since {start}, {label} over **{place}** has {direction} significantly, and under the selected criterion the change became detectable above natural variability in **{shown_toe}**."
    elif sig:
        msg = (f"Since {start}, {label} over **{place}** has {direction} significantly, but has not yet stayed above {k}× natural variability."
               + (f" If the trend continues unchanged, it would emerge around **{proj_year}** (conditional projection)." if proj_year and proj_year <= HORIZON else ""))
    else:
        msg = (f"Since {start}, there is no statistically significant trend (after FDR correction) in {label} over **{place}**: "
               "any change is still hidden in natural variability."
               + (f" If this weak trend continued unchanged, it would emerge around **{proj_year}**, but this projection is not statistically supported." if proj_year and proj_year <= HORIZON
                  else (f" Even if this weak trend continued unchanged, it would not emerge before {HORIZON}." if proj_year else "")))
    st.info(msg)

with st.expander("Method and data"):
    st.markdown("""
- **Data**: NASA MERRA-2 monthly aerosol diagnostics (M2TMNXAER v5.12.4, GES DISC), 0.5° × 0.625°, 1980 to present.
- **Seasonal means**: JJA (peak trans-Atlantic transport), DJF (labelled by January year), annual. Only complete seasons are used, so annual means end in the last complete year while JJA and DJF can include the current year (the colour scale and slider end at the last analysed year).
- **Trend**: Sen's slope. **Significance**: Mann-Kendall test with an AR(1)-based variance correction inspired by Yue and Wang (2004). Because thousands of grid cells are tested at once, p-values are adjusted with the Benjamini-Hochberg false discovery rate (FDR); scope = all valid grid cells of each map (variable × season × start year), or the 4 preset regions (each region-mean series is tested with its own Mann-Kendall test, then the 4 p-values are FDR-corrected together). FDR controls the expected share of false discoveries; it does not guarantee every significant cell is correct, and spatial dependence is only partly covered by the method's assumptions.
- **Noise**: standard deviation of the detrended interannual series, from the same period as the signal. With no control simulation for reanalysis this is a pragmatic choice; for 1980 starts it may be inflated by the 1980s Sahel drought and by the ~2000 observing-system change.
- **Estimated Time of Emergence (under the selected detection criterion)**: S(t) is the OLS linear trend fitted from the start year to year t, multiplied by elapsed years. ToE is the first year when S(t)/noise ≥ k *with the same sign as the final signal* and stays there through the end of the record (a retrospective definition), and the trend passes FDR. Main findings use k = 2; k = 1 is exploratory. The displayed trend is Sen's slope while S/N uses OLS slopes; the two are usually close.
- **Projected ToE**: a conditional extrapolation based on the historical trend and background variability, for places where the signal has not yet emerged: start year + k × noise / |trend per year|. Projections from trends that pass FDR are conditional projections; projections from statistically insignificant trends are shown for exploration only and are not statistically supported. Projections beyond 2100 are reported as highly uncertain. None of these are forecasts.
- **Caveat**: MERRA-2 assimilated only AVHRR ocean AOD before ~2000 and added MODIS, MISR and AERONET afterwards. Trends that span 2000 may partly reflect this change; compare the 1980 and 2003 start years.
""")
st.caption(f"Data folder: {DATA_DIR} · EarthSignal · NASA Space Apps Challenge 2026")
