import os, json, time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.floors import load_floors, rooms_by_floor
from src.merge_policy import suggest_merges_to_common
from src.policy import derive_commands
from src.open_meteo import geocode_city, forecast_hours, outline_bullets
from src.hf_llm import summarize

# ------------------------------------------------------------
# Site configuration and theme
# ------------------------------------------------------------
st.set_page_config(page_title="Smart Building Operations Center", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
:root { --bg:#0b1020; --panel:#0f172a; --ink:#e5e7eb; --muted:#94a3b8; --line:#1f2937; --accent:#1fb6ff; }
html, body, [class^="css"] { font-family: Inter, ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #0b1020 0%, #0f172a 100%); color: var(--ink); }
.block-container { padding-top: 0.5rem; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.18); }
.kpi { display:flex; gap:12px; }
.kpi .box { flex:1; background: var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px; text-align:center; }
.kpi .box h3 { margin: 0; color: var(--muted); font-weight:600; font-size:0.9rem; }
.kpi .box p { margin: 6px 0 0 0; font-size:1.5rem; font-weight:700; }
.stButton>button { background: var(--accent); color: #081018; border: none; border-radius: 10px; font-weight: 700; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
.small { color: var(--muted); font-size: 0.85rem; }
.hr { height:1px; background: var(--line); margin: 0.5rem 0 1rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# Paths and helpers
# ------------------------------------------------------------
FLOORS_JSON = "data/floors.json"
BATCH_DIR = "outputs/synth"
LIVE_DIR = "outputs/live"

def _read_csv(path):
    return pd.read_csv(path, parse_dates=["timestamp"]) if os.path.exists(path) else None

def _live_snapshots():
    occ = _read_csv(os.path.join(LIVE_DIR, "occupancy_live.csv"))
    sen = _read_csv(os.path.join(LIVE_DIR, "sensors_live.csv"))
    er  = _read_csv(os.path.join(LIVE_DIR, "energy_room_live.csv"))
    ef  = _read_csv(os.path.join(LIVE_DIR, "energy_floor_live.csv"))
    eb  = _read_csv(os.path.join(LIVE_DIR, "energy_building_live.csv"))
    return occ, sen, er, ef, eb

def _batch_dataset():
    occ = _read_csv(os.path.join(BATCH_DIR, "occupancy.csv"))
    sen = _read_csv(os.path.join(BATCH_DIR, "sensors.csv"))
    er  = _read_csv(os.path.join(BATCH_DIR, "energy_room.csv"))
    ef  = _read_csv(os.path.join(BATCH_DIR, "energy_floor.csv"))
    eb  = _read_csv(os.path.join(BATCH_DIR, "energy_building.csv"))
    return occ, sen, er, ef, eb

def _file_mtime(path):
    try:
        ts = os.path.getmtime(path)
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return "n/a"

# ------------------------------------------------------------
# Load floors
# ------------------------------------------------------------
floors_idx, floors_list = load_floors(FLOORS_JSON)
floors = [f["floor_id"] for f in floors_list]
floor_name = {f["floor_id"]: f["name"] for f in floors_list}

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.title("Controls")
mode = st.sidebar.radio("Data mode", ["Live", "Batch"], index=0)

# Manual refresh button: reloads data without full-page timers
do_refresh = st.sidebar.button("Refresh data")

# Weather options
city = st.sidebar.text_input("Weather city", value="Bengaluru")
hrs = st.sidebar.slider("Forecast horizon (hours)", 12, 72, 36, step=6)

# Batch assist
if mode == "Batch" and st.sidebar.button("Generate 24h synthetic batch"):
    from src.synth import generate_batch
    os.makedirs(BATCH_DIR, exist_ok=True)
    generate_batch(FLOORS_JSON, BATCH_DIR, hours=24, step_min=5)
    st.sidebar.success("Batch data generated.")

# ------------------------------------------------------------
# Data loading according to mode
# ------------------------------------------------------------
def _load_data():
    if mode == "Live":
        return _live_snapshots()
    else:
        return _batch_dataset()

occ, sen, er, ef, eb = _load_data()
if occ is None or sen is None or er is None:
    st.warning("No dataset found. For Live mode, run the live generator. For Batch mode, click 'Generate 24h synthetic batch'.")
    st.stop()

# Show freshness indicators (Live only)
if mode == "Live":
    st.markdown(
        f"<div class='small'>Live snapshots: "
        f"occupancy { _file_mtime(os.path.join(LIVE_DIR,'occupancy_live.csv')) } | "
        f"sensors { _file_mtime(os.path.join(LIVE_DIR,'sensors_live.csv')) } | "
        f"energy-room { _file_mtime(os.path.join(LIVE_DIR,'energy_room_live.csv')) }</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Select timepoint (Batch) or use now (Live)
# ------------------------------------------------------------
if mode == "Batch":
    ts_list = sorted(occ["timestamp"].unique())
    if "ts_idx" not in st.session_state:
        st.session_state["ts_idx"] = len(ts_list)-1
    st.session_state["ts_idx"] = st.slider("Timeline", 0, len(ts_list)-1, st.session_state["ts_idx"])
    ts_sel = ts_list[st.session_state["ts_idx"]]
    occ_now = occ[occ["timestamp"] == ts_sel]
    sen_now = sen[sen["timestamp"] == ts_sel]
    er_now  = er[er["timestamp"] == ts_sel]
else:
    ts_sel = occ["timestamp"].max()
    occ_now, sen_now, er_now = occ, sen, er

# If user pressed Refresh, reload just-in-time (without losing tab state)
if do_refresh:
    occ, sen, er, ef, eb = _load_data()
    if mode == "Batch":
        # keep same ts_sel index if possible
        if "ts_idx" in st.session_state:
            ts_list = sorted(occ["timestamp"].unique())
            st.session_state["ts_idx"] = min(st.session_state["ts_idx"], len(ts_list)-1)
            ts_sel = ts_list[st.session_state["ts_idx"]]
        occ_now = occ[occ["timestamp"] == ts_sel]
        sen_now = sen[sen["timestamp"] == ts_sel]
        er_now  = er[er["timestamp"] == ts_sel]
    else:
        ts_sel = occ["timestamp"].max()
        occ_now, sen_now, er_now = occ, sen, er

# ------------------------------------------------------------
# Build maps for policies
# ------------------------------------------------------------
counts_by_floor = {fid: {} for fid in floors}
for _, r in occ_now.iterrows():
    counts_by_floor[r["floor_id"]][r["room_id"]] = int(r["count"])

sensors_by_floor = {}
for _, r in sen_now.iterrows():
    sensors_by_floor.setdefault(r["floor_id"], {})[r["room_id"]] = {
        "co2": float(r["co2"]), "lux": float(r["lux"]), "noise": float(r["noise"]),
        "motion": int(r["motion"]), "door": int(r["door"]), "rh": float(r["rh"])
    }

energy_room_snapshot = {}
for _, r in er_now.iterrows():
    energy_room_snapshot.setdefault(r["floor_id"], {})[r["room_id"]] = float(r["kw"])

merge_plan = suggest_merges_to_common(floors_list, counts_by_floor)
commands = derive_commands(floors_list, counts_by_floor, merge_plan["assignments"], sensors_by_floor)

# ------------------------------------------------------------
# Header and KPIs
# ------------------------------------------------------------
st.markdown("<h1>Smart Building Operations Center</h1><p style='color:#94a3b8'>Multi-floor occupancy, sensors, per-room energy, policy, weather, and AI brief</p>", unsafe_allow_html=True)

total_occ = int(occ_now["count"].sum())
active_rooms = int((occ_now["count"] > 0).sum())
saved_area = float(merge_plan.get("saved_area_m2", 0))
sug = len(merge_plan.get("suggestions", []))

st.markdown(
    f"""
<div class='kpi'>
  <div class='box'><h3>Total Occupancy</h3><p>{total_occ}</p></div>
  <div class='box'><h3>Active Rooms</h3><p>{active_rooms}</p></div>
  <div class='box'><h3>Merge Suggestions</h3><p>{sug}</p></div>
  <div class='box'><h3>Potential Saved Area</h3><p>{saved_area:.1f} m²</p></div>
</div>
""", unsafe_allow_html=True
)

st.write(" ")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Floor Plans", "Trends", "Actions", "Weather", "AI Brief"])

# Floor Plans
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if "floor_sel" not in st.session_state:
        st.session_state["floor_sel"] = floors[0]
    st.session_state["floor_sel"] = st.selectbox("Choose floor", options=floors,
                                                 index=floors.index(st.session_state["floor_sel"]),
                                                 format_func=lambda x: f"{x} — {floor_name[x]}")
    floor_sel = st.session_state["floor_sel"]
    rlist = rooms_by_floor(floors_list, floor_sel)

    maxx = max(r["x"]+r["w"] for r in rlist)
    maxy = max(r["y"]+r["h"] for r in rlist)
    fig = go.Figure()
    for r in rlist:
        rid = r["room_id"]
        occv = counts_by_floor.get(floor_sel, {}).get(rid, 0)
        color = "rgba(31, 182, 255, 0.75)" if occv>0 else "rgba(239,68,68,0.75)"
        fig.add_shape(type="rect", x0=r["x"], y0=r["y"], x1=r["x"]+r["w"], y1=r["y"]+r["h"],
                      line=dict(color="rgba(255,255,255,0.6)", width=1.2), fillcolor=color)
        kw = energy_room_snapshot.get(floor_sel, {}).get(rid, 0.0)
        fig.add_trace(go.Scatter(x=[r["x"]+r["w"]/2], y=[r["y"]+r["h"]/2],
                                 text=[f"{rid}<br>{occv} ppl<br>{kw:.2f} kW"], mode="text",
                                 textfont=dict(color="white", size=12)))
    fig.update_xaxes(visible=False, range=[0, maxx+0.5])
    fig.update_yaxes(visible=False, range=[maxy+0.5, -0.5])
    fig.update_layout(height=540, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Trends
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if mode == "Live":
        def tail_csv(path, n=400):
            p = os.path.join(LIVE_DIR, path)
            if not os.path.exists(p): return None
            df = pd.read_csv(p, parse_dates=["timestamp"])
            return df.tail(n)
        occ_hist = tail_csv("occupancy_log.csv")
        ef_hist  = tail_csv("energy_floor_log.csv")
        eb_hist  = tail_csv("energy_building_log.csv")
    else:
        occ_hist = occ.groupby("timestamp", as_index=False)["count"].sum()
        ef_hist  = _read_csv(os.path.join(BATCH_DIR, "energy_floor.csv"))
        eb_hist  = _read_csv(os.path.join(BATCH_DIR, "energy_building.csv"))
        if ef_hist is not None: ef_hist = ef_hist.tail(400)
        if eb_hist is not None: eb_hist = eb_hist.tail(400)

    if occ_hist is not None and len(occ_hist):
        fig1 = px.area(occ_hist, x="timestamp", y="count", title="Total Occupancy (recent)")
        fig1.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig1, use_container_width=True)

    if ef_hist is not None and len(ef_hist):
        fig2 = px.line(ef_hist, x="timestamp", y="meter_kw", color="floor_id", title="Energy by Floor (kW)")
        fig2.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    if eb_hist is not None and len(eb_hist):
        fig3 = px.area(eb_hist, x="timestamp", y="meter_kw", title="Building Energy (kW)")
        fig3.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Actions
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Merge suggestions")
    if not merge_plan["suggestions"]:
        st.info("No merges suggested at this time.")
    else:
        for s in merge_plan["suggestions"]:
            tos = ", ".join([m["to"] for m in s["moves"]])
            st.write(f"{s['floor_id']} / {s['from']} → {tos} (zone {s['zone']})")

    st.subheader("Derived commands")
    if commands:
        dfc = pd.DataFrame(commands)
        st.dataframe(dfc.sort_values(["floor_id","room_id"]), use_container_width=True, height=320)
    else:
        st.caption("No commands.")
    st.markdown("</div>", unsafe_allow_html=True)

# Weather
with tab4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button("Fetch forecast"):
        try:
            lat, lon, name = geocode_city(city)
            fc = forecast_hours(lat, lon, hours=hrs, tz="auto")
            bullets = outline_bullets(fc)
            st.session_state["wx_bullets"] = bullets
            st.caption(f"Forecast for {name}")
            figw = px.line(fc, x="timestamp", y=["temp","dew","cloud","precip","wind"], title="Next hours")
            figw.update_layout(legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
            st.plotly_chart(figw, use_container_width=True)
            for b in bullets: st.write("- " + b)
        except Exception as e:
            st.error(str(e))
    st.markdown("</div>", unsafe_allow_html=True)

# AI Brief (Hugging Face)
with tab5:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.caption("Uses Hugging Face Inference API. Set HUGGINGFACE_API_TOKEN and optionally HF_MODEL.")
    if st.button("Generate JSON brief"):
        bullets = st.session_state.get("wx_bullets", ["No forecast fetched"])
        js = summarize(counts_by_floor, sensors_by_floor, merge_plan, energy_room_snapshot, bullets, window=str(ts_sel))
        st.code(js, language="json")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='color:#94a3b8; text-align:center;'>© Smart Building Operations Center — demo</p>", unsafe_allow_html=True)
