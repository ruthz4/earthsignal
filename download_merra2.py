import argparse, os, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import earthaccess

VARS = ["DUEXTTAU", "TOTEXTTAU", "DUSMASS", "DUCMASS", "DUFLUXU"]
J0, J1, I0, I1 = 180, 260, 128, 352   # 0-40N, 100W-40E on the MERRA-2 grid
GOOD = (b"\x89HDF", b"CDF\x01", b"CDF\x02")

def opendap_link(g):
    for u in g["umm"].get("RelatedUrls", []):
        url = u.get("URL", "")
        if "opendap" in url.lower() or "OPENDAP" in u.get("Subtype", ""):
            return url
    return None

def subset_url(base):
    if "opendap.earthdata.nasa.gov" in base:      # cloud OPeNDAP (DAP4)
        ce = ";".join([f"/{v}[0:1:0][{J0}:1:{J1}][{I0}:1:{I1}]" for v in VARS]
                      + [f"/lat[{J0}:1:{J1}]", f"/lon[{I0}:1:{I1}]", "/time[0:1:0]"])
        return base + ".dap.nc4?dap4.ce=" + urllib.parse.quote(ce, safe="/;:")
    ce = ",".join([f"{v}[0:0][{J0}:{J1}][{I0}:{I1}]" for v in VARS]
                  + [f"lat[{J0}:{J1}]", f"lon[{I0}:{I1}]", "time[0:0]"])
    return base + ".nc4?" + ce

def fetch_opendap(s, name, base, out):
    r = s.get(subset_url(base), timeout=180)
    if r.status_code == 200 and r.content[:4] in GOOD:
        open(out, "wb").write(r.content); return "ok"
    return f"HTTP {r.status_code}"

def fetch_full(s, name, link, out):
    import xarray as xr
    tmp = out + ".full.tmp"
    with s.get(link, stream=True, timeout=600) as r:
        if r.status_code != 200: return f"HTTP {r.status_code}"
        with open(tmp, "wb") as f:
            for c in r.iter_content(1 << 20): f.write(c)
    ds = xr.open_dataset(tmp)[VARS].sel(lat=slice(0, 40), lon=slice(-100, 40))
    ds.load().to_netcdf(out); ds.close(); os.remove(tmp); return "ok"

def job(s, g, outdir, full, tries=3):
    link = [l for l in g.data_links() if l.endswith(".nc4")][0]
    name = os.path.basename(link)
    out = os.path.join(outdir, name.replace(".nc4", "_subset.nc4"))
    if os.path.exists(out) and os.path.getsize(out) > 10_000: return name, "skip"
    base = None if full else opendap_link(g)
    msg = "no OPeNDAP link" if base is None and not full else ""
    for k in range(tries):
        try:
            st = fetch_full(s, name, link, out) if (full or base is None) else fetch_opendap(s, name, base, out)
            if st == "ok": return name, "ok"
            msg = st
            if st.startswith(("HTTP 401", "HTTP 403")): break
        except Exception as e: msg = str(e)[:150]
        time.sleep(3 * (k + 1))
    return name, f"FAILED ({msg})"

ap = argparse.ArgumentParser()
ap.add_argument("--output", default="merra2")
ap.add_argument("--test", action="store_true")
ap.add_argument("--full", action="store_true", help="download full files and subset locally")
ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()
os.makedirs(a.output, exist_ok=True)
earthaccess.login(strategy="netrc")
s = earthaccess.get_requests_https_session()
gs = earthaccess.search_data(short_name="M2TMNXAER", version="5.12.4")
gs = sorted(gs, key=lambda g: g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
print(f"Found {len(gs)} granules")
if a.test:
    gs = gs[:1] + gs[-1:]
    print("OPeNDAP link of first granule:", opendap_link(gs[0]))
    if not a.full and gs: print("Request URL:", subset_url(opendap_link(gs[0]) or "NONE")[:300])
done, failed = 0, []
with ThreadPoolExecutor(a.workers) as pool:
    for f in as_completed([pool.submit(job, s, g, a.output, a.full) for g in gs]):
        n, st = f.result(); done += 1
        if st not in ("ok", "skip"): failed.append(n); print(f"  {n}: {st}")
        if done % 25 == 0 or done == len(gs): print(f"  {done}/{len(gs)} done, {len(failed)} failed")
print("Rerun to retry failed files." if failed else f"All files saved in {a.output}/")
