"""EarthSignal: animated Time of Emergence map.

Run locally (does not change the deployed app):
    streamlit run animate_toe.py
"""
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

st.set_page_config(page_title="EarthSignal · Emergence animation", page_icon="🌍", layout="wide")
DATA_DIR = next((d for d in [os.environ.get("EARTHSIGNAL_DATA", ""), "results", "mock_results"]
                 if d and os.path.isdir(d)), None)
if DATA_DIR is None:
    st.error("No results folder found. Put the analysis output in ./results."); st.stop()

VAR_LABELS = {"DUEXTTAU": "Dust AOD (550 nm)", "TOTEXTTAU": "Total AOD (550 nm)",
              "DUSMASS": "Dust surface concentration", "DUCMASS": "Dust column mass",
              "DUFLUXU": "Dust zonal flux"}
SEASON_LABELS = {"ANN": "Annual mean", "JJA": "Summer (JJA)", "DJF": "Winter (DJF)"}
COLORSCALE = [[0, "#5B2C06"], [0.4, "#C8650E"], [0.75, "#F2A51A"], [1, "#FFD43B"]]
REGIONS = {"Sahara source": (15, 30, -15, 30), "Sahel": (10, 18, -18, 25),
           "Atlantic corridor": (10, 25, -50, -20), "Caribbean": (10, 25, -90, -60)}


@st.cache_data
def load(var, season, start):
    ds = xr.open_dataset(os.path.join(DATA_DIR, f"fields_{var}_{season}_{start}.nc")).load()
    return ({k: ds[k].values for k in ds.data_vars}, ds.lat.values, ds.lon.values,
            int(ds.attrs.get("end_year", 2025)))


def box(s, n, w, e, m=25):
    return (np.r_[np.linspace(w, e, m), np.full(m, e), np.linspace(e, w, m), np.full(m, w)],
            np.r_[np.full(m, s), np.linspace(s, n, m), np.full(m, n), np.linspace(n, s, m)])


def build_figure(fields, lat, lon, end_year, start, k, only_sig, show_regions, step=2, speed=700):
    toe = fields[f"toe_k{k}"].astype(float)
    q = fields.get("qvalue", fields["pvalue"])
    valid = np.isfinite(fields["slope"])
    if only_sig:
        toe = np.where(q < 0.05, toe, np.nan)

    # area-weighted share of valid cells emerged by each year (full resolution)
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], toe.shape)
    wsum = w[valid].sum()
    first = start + 10
    years = list(range(first, end_year + 1))
    share = {y: 100 * w[valid & (toe <= y)].sum() / wsum for y in years}

    sl = (slice(None, None, step), slice(None, None, step))
    LON, LAT = np.meshgrid(lon[::step], lat[::step])
    T, V = toe[sl], valid[sl]

    def frame_traces(y):
        em = V & (T <= y)
        new = V & (T == y)
        return [
            go.Scattergeo(lon=LON[em], lat=LAT[em], mode="markers", hoverinfo="skip",
                          marker=dict(size=5, symbol="square", color=T[em], colorscale=COLORSCALE,
                                      cmin=first, cmax=end_year,
                                      colorbar=dict(title="Year<br>emerged", x=1.0, y=0.5, len=0.75, thickness=14))),
            go.Scattergeo(lon=LON[new], lat=LAT[new], mode="markers", hoverinfo="skip",
                          marker=dict(size=9, symbol="square-open", color="#FFFFFF",
                                      line=dict(width=1.5, color="#FFFFFF"))),
        ]

    def title(y):
        return (f"<span style='font-size:40px'><b>{y}</b></span>"
                f"<span style='font-size:18px'>   ·   {share[y]:.1f}% of the area has emerged</span>")

    fig = go.Figure()
    # trace 0: static background of all valid cells
    fig.add_trace(go.Scattergeo(lon=LON[V], lat=LAT[V], mode="markers", hoverinfo="skip",
                                marker=dict(size=5, symbol="square", color="#d5dbe3"), showlegend=False))
    for tr in frame_traces(first):          # traces 1 and 2: animated
        tr.showlegend = False
        fig.add_trace(tr)
    if show_regions:
        for name, (s, n, wl, e) in REGIONS.items():
            bx, by = box(s, n, wl, e)
            fig.add_trace(go.Scattergeo(lon=bx, lat=by, mode="lines", hoverinfo="skip", showlegend=False,
                                        line=dict(color="#111827" if name == "Sahara source" else "#475569",
                                                  width=2)))
            fig.add_trace(go.Scattergeo(lon=[(wl + e) / 2], lat=[n + 1.6], mode="text", text=[name],
                                        hoverinfo="skip", showlegend=False,
                                        textfont=dict(size=12, color="#1F2937")))

    fig.frames = [go.Frame(name=str(y), data=frame_traces(y), traces=[1, 2],
                           layout=dict(title_text=title(y))) for y in years]

    fig.update_geos(lataxis_range=[0, 40], lonaxis_range=[-100, 40], showcoastlines=True,
                    coastlinecolor="#444", showland=True, landcolor="#f4f1ea", showocean=True,
                    oceancolor="#e3eef6", projection_type="equirectangular", showframe=False)
    fig.update_layout(
        title=dict(text=title(first), x=0.02, y=0.97),
        height=560, margin=dict(l=10, r=10, t=80, b=70), paper_bgcolor="white",
        updatemenus=[dict(type="buttons", direction="left", x=0.02, y=-0.02, xanchor="left", yanchor="top",
                          pad=dict(r=10, t=10), showactive=False, buttons=[
            dict(label="▶  Play", method="animate",
                 args=[None, dict(frame=dict(duration=speed, redraw=True), fromcurrent=True,
                                  transition=dict(duration=0))]),
            dict(label="❚❚  Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.18, len=0.8, y=-0.02, yanchor="top", currentvalue=dict(visible=False),
                      pad=dict(t=10), steps=[dict(label=str(y), method="animate",
                                                  args=[[str(y)], dict(mode="immediate",
                                                                       frame=dict(duration=0, redraw=True),
                                                                       transition=dict(duration=0))])
                                             for y in years])])
    return fig, share


# ---------------- page ----------------
st.title("Watch the signal emerge")
st.caption("Each frame shows the cells whose long-term change has risen above natural variability by that year. "
           "White outlines mark cells that emerged that year. ToE is estimated retrospectively from the full record.")

c1, c2, c3, c4, c5 = st.columns([2, 1.3, 1.3, 1, 1.3])
var = c1.selectbox("Variable", list(VAR_LABELS), format_func=VAR_LABELS.get)
season = c2.selectbox("Season", list(SEASON_LABELS), format_func=SEASON_LABELS.get)
start = c3.selectbox("Start year", [2003, 1980])
k = c4.selectbox("k", [2, 1])
only_sig = c5.checkbox("FDR q < 0.05", value=True)
show_regions = st.checkbox("Show the four dust-path regions", value=True)

fields, lat, lon, end_year = load(var, season, start)
fig, share = build_figure(fields, lat, lon, end_year, start, k, only_sig, show_regions)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"NASA MERRA-2 (M2TMNXAER) · {VAR_LABELS[var]} · {SEASON_LABELS[season]} · {start} to {end_year} · "
           f"k = {k}{' · FDR q < 0.05' if only_sig else ''}")
