import os, json, time, math, tempfile, shutil
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from .floors import load_floors

def _diurnal_prob(hour: float, is_common: bool) -> float:
    peak1 = math.exp(-((hour - 11.0) ** 2) / (2 * 2.2 ** 2))
    peak2 = math.exp(-((hour - 15.0) ** 2) / (2 * 2.5 ** 2))
    base = 0.02 if not is_common else 0.05
    scale = 0.55 if is_common else 0.45
    return max(0.0, min(0.98, base + scale * (0.55 * peak1 + 0.45 * peak2)))

def _safe_write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), suffix=".tmp") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)

def run_live(out_dir: str = "outputs/live", floors_json: str = "data/floors.json",
             tick_seconds: int = 3,
             seed: int = 1337,
             ambient_kw_per_m2: float = 0.02,
             per_person_watts: float = 120.0,
             hvac_kw_per_degC_per_m2: float = 0.01,
             setpoint_cool_c: float = 24.0):
    """
    Live loop producing stable snapshot CSVs, rewritten atomically each tick:
      - occupancy_live.csv
      - sensors_live.csv
      - energy_room_live.csv
      - energy_floor_live.csv
      - energy_building_live.csv
    Also appends to rolling history files for plots:
      - occupancy_log.csv, sensors_log.csv, energy_room_log.csv, energy_floor_log.csv, energy_building_log.csv
    """
    np.random.seed(seed)
    idx, floors_list = load_floors(floors_json)
    os.makedirs(out_dir, exist_ok=True)

    # inertia state
    prev_counts = {k: 0 for k in idx.keys()}

    # synthetic outdoor temperature diurnal
    def outdoor_temp(now_local_hour: float, mean=28.0, swing=6.0) -> float:
        return mean + swing * math.sin((now_local_hour - 6) / 24 * 2 * math.pi) + np.random.normal(0, 0.4)

    while True:
        now = datetime.now(timezone.utc)
        now_local = now.astimezone()
        hour_local = now_local.hour + now_local.minute / 60.0

        occ_rows = []
        sen_rows = []
        energy_room_rows = []

        T_out = outdoor_temp(hour_local)

        for F in floors_list:
            fid = F["floor_id"]
            for r in F["rooms"]:
                rid = r["room_id"]
                cap = int(r["capacity"])
                p = _diurnal_prob(hour_local, r.get("is_common", False))
                draw = np.random.binomial(cap, p)
                count = int(round(0.6 * prev_counts[(fid, rid)] + 0.4 * draw))
                prev_counts[(fid, rid)] = count

                # sensors
                co2 = float(np.clip(np.random.normal(450 + 35*count, 40), 400, 2000))
                daytime = 7 <= now_local.hour <= 18
                lux = float(np.clip(np.random.normal(700 if daytime else 120, 80), 0, 1500))
                noise = float(np.clip(35 + 3 * (max(count,0) ** 0.5) + np.random.normal(0, 2), 30, 85))
                motion = int(count > 0)
                door = int(np.random.rand() < (0.1 if count > 0 else 0.02))
                rh = float(np.clip(np.random.normal(50, 6), 30, 70))

                occ_rows.append({"timestamp": now.isoformat(), "floor_id": fid, "room_id": rid, "count": count})
                sen_rows.append({"timestamp": now.isoformat(), "floor_id": fid, "room_id": rid,
                                 "co2": co2, "lux": lux, "noise": noise, "motion": motion, "door": door, "rh": rh})

                # per-room power model (kW)
                area = r["area_m2"]
                ambient_kw = ambient_kw_per_m2 * area
                ppl_kw = (per_person_watts * count) / 1000.0
                cooling_kw = max(T_out - setpoint_cool_c, 0.0) * hvac_kw_per_degC_per_m2 * area * (1.0 if count > 0 else 0.3)
                # daylight dimming reduces lighting share when lux high and occupied
                lighting_kw = 0.008 * area * (0.5 if (lux >= 600 and count > 0) else 1.0)
                # door open penalty
                door_penalty_kw = 0.1 if (door == 1 and count > 0) else 0.0

                room_kw = ambient_kw + ppl_kw + cooling_kw + lighting_kw + door_penalty_kw
                energy_room_rows.append({"timestamp": now.isoformat(), "floor_id": fid, "room_id": rid, "kw": room_kw})

        # aggregate per floor and building
        occ_df = pd.DataFrame(occ_rows)
        sen_df = pd.DataFrame(sen_rows)
        er_df = pd.DataFrame(energy_room_rows)

        ef_df = er_df.groupby(["timestamp","floor_id"], as_index=False)["kw"].sum().rename(columns={"kw":"meter_kw"})
        eb_df = er_df.groupby(["timestamp"], as_index=False)["kw"].sum().rename(columns={"kw":"meter_kw"})

        # atomically write current snapshots
        _safe_write_csv(occ_df, os.path.join(out_dir, "occupancy_live.csv"))
        _safe_write_csv(sen_df, os.path.join(out_dir, "sensors_live.csv"))
        _safe_write_csv(er_df,  os.path.join(out_dir, "energy_room_live.csv"))
        _safe_write_csv(ef_df,  os.path.join(out_dir, "energy_floor_live.csv"))
        _safe_write_csv(eb_df,  os.path.join(out_dir, "energy_building_live.csv"))

        # append to logs
        def append(df, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            mode = "a" if os.path.exists(path) else "w"
            header = not os.path.exists(path)
            df.to_csv(path, mode=mode, header=header, index=False)
        append(occ_df, os.path.join(out_dir, "occupancy_log.csv"))
        append(sen_df, os.path.join(out_dir, "sensors_log.csv"))
        append(er_df,  os.path.join(out_dir, "energy_room_log.csv"))
        append(ef_df,  os.path.join(out_dir, "energy_floor_log.csv"))
        append(eb_df,  os.path.join(out_dir, "energy_building_log.csv"))

        time.sleep(tick_seconds)
