import os, json, argparse, math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from .floors import load_floors

def diurnal_prob(hour: float, is_common: bool) -> float:
    peak1 = math.exp(-((hour - 11.0) ** 2) / (2 * 2.2 ** 2))
    peak2 = math.exp(-((hour - 15.0) ** 2) / (2 * 2.5 ** 2))
    base = 0.02 if not is_common else 0.05
    scale = 0.55 if is_common else 0.45
    return max(0.0, min(0.98, base + scale * (0.55 * peak1 + 0.45 * peak2)))

def simulate_temp(ts: pd.DatetimeIndex, t_mean=28.0, swing=6.0) -> np.ndarray:
    hours = (ts.view("int64") // 3_600_000_000_000) % 24
    diurnal = swing * np.sin((hours - 6) / 24 * 2 * np.pi)
    noise = np.random.normal(0, 0.7, size=len(ts))
    return t_mean + diurnal + noise

def generate_batch(floors_json: str, out_dir: str, hours: int = 24, step_min: int = 5, start_iso: str = None):
    os.makedirs(out_dir, exist_ok=True)
    idx, floors_list = load_floors(floors_json)

    start = pd.Timestamp(start_iso, tz="UTC") if start_iso else pd.Timestamp(datetime.now(timezone.utc))
    ts = pd.date_range(start=start, periods=int(hours*60/step_min), freq=f"{step_min}min", tz="UTC")

    prev = {k: 0 for k in idx.keys()}
    occ_rows, sen_rows, er_rows = [], [], []

    for t in ts:
        hl = t.tz_convert(None).hour + t.tz_convert(None).minute/60.0
        T_out = 28.0 + 6.0 * math.sin((hl - 6) / 24 * 2 * math.pi) + np.random.normal(0, 0.4)

        for F in floors_list:
            fid = F["floor_id"]
            for r in F["rooms"]:
                rid = r["room_id"]
                cap = int(r["capacity"])
                p = diurnal_prob(hl, r.get("is_common", False))
                draw = np.random.binomial(cap, p)
                cnt = int(round(0.5*prev[(fid, rid)] + 0.5*draw))
                prev[(fid, rid)] = cnt

                co2 = float(np.clip(np.random.normal(450 + 35*cnt, 40), 400, 2000))
                daytime = 7 <= t.tz_convert(None).hour <= 18
                lux = float(np.clip(np.random.normal(700 if daytime else 120, 80), 0, 1500))
                noise = float(np.clip(35 + 3 * (max(cnt,0) ** 0.5) + np.random.normal(0,2), 30, 85))
                motion = int(cnt > 0)
                door = int(np.random.rand() < (0.1 if cnt > 0 else 0.02))
                rh = float(np.clip(np.random.normal(50, 6), 30, 70))

                occ_rows.append({"timestamp": t.isoformat(), "floor_id": fid, "room_id": rid, "count": cnt})
                sen_rows.append({"timestamp": t.isoformat(), "floor_id": fid, "room_id": rid,
                                 "co2":co2, "lux":lux, "noise":noise, "motion":motion, "door":door, "rh":rh})

                # per-room energy (kW)
                area = r["area_m2"]
                ambient_kw = 0.02 * area
                ppl_kw = 0.001 * 120.0 * cnt
                cooling_kw = max(T_out - 24.0, 0.0) * 0.01 * area * (1.0 if cnt > 0 else 0.3)
                lighting_kw = 0.008 * area * (0.5 if (lux >= 600 and cnt > 0) else 1.0)
                door_penalty_kw = 0.1 if (door == 1 and cnt > 0) else 0.0
                room_kw = ambient_kw + ppl_kw + cooling_kw + lighting_kw + door_penalty_kw

                er_rows.append({"timestamp": t.isoformat(), "floor_id": fid, "room_id": rid, "kw": room_kw})

    occ_df = pd.DataFrame(occ_rows)
    sen_df = pd.DataFrame(sen_rows)
    er_df  = pd.DataFrame(er_rows)
    ef_df  = er_df.groupby(["timestamp","floor_id"], as_index=False)["kw"].sum().rename(columns={"kw":"meter_kw"})
    eb_df  = er_df.groupby(["timestamp"], as_index=False)["kw"].sum().rename(columns={"kw":"meter_kw"})

    occ_df.to_csv(os.path.join(out_dir, "occupancy.csv"), index=False)
    sen_df.to_csv(os.path.join(out_dir, "sensors.csv"), index=False)
    er_df.to_csv(os.path.join(out_dir, "energy_room.csv"), index=False)
    ef_df.to_csv(os.path.join(out_dir, "energy_floor.csv"), index=False)
    eb_df.to_csv(os.path.join(out_dir, "energy_building.csv"), index=False)
    print(f"Wrote synthetic batch to {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors_json", default="data/floors.json")
    ap.add_argument("--out_dir", default="outputs/synth")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--step_min", type=int, default=5)
    ap.add_argument("--start", type=str, default=None)
    args = ap.parse_args()
    generate_batch(args.floors_json, args.out_dir, args.hours, args.step_min, args.start)

if __name__ == "__main__":
    main()
