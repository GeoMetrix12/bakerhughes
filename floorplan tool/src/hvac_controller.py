import os, glob, time, json, argparse
import pandas as pd
from typing import Dict, Any
from .merge_policy import load_rooms, suggest_merges_to_common
from .iot import IoTSink

def latest_counts_from_glob(csv_glob: str) -> Dict[str, int]:
    files = glob.glob(csv_glob)
    if not files: return {}
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass
    if not dfs: return {}
    all_df = pd.concat(dfs, ignore_index=True)
    if "timestamp" not in all_df.columns: return {}
    all_df["timestamp"] = pd.to_datetime(all_df["timestamp"], utc=True, errors="coerce")
    all_df = all_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    latest = all_df.groupby("room_id").tail(1)
    return {r.room_id: int(r["count"]) for _, r in latest.iterrows()}

def latest_sensors(sensors_csv: str) -> Dict[str, Any]:
    if not os.path.exists(sensors_csv): return {}
    df = pd.read_csv(sensors_csv)
    out = {}
    for _, r in df.iterrows():
        out[r["room_id"]] = {
            "co2": float(r.get("co2", 450)), "lux": float(r.get("lux", 200)),
            "noise": float(r.get("noise", 40)), "motion": int(r.get("motion", 0)),
            "door": int(r.get("door", 0)), "rh": float(r.get("rh", 50.0))
        }
    return out

def derive_commands(rooms, counts, merge_plan, sensors):
    """
    Smarter policy:
      - If count == 0 → lights:off, fan:off, hvac:off
      - If occupied:
          - lights:on (dim if lux>600)
          - hvac:comfort; if merge suggests inactive → eco
          - if CO2>1200 → hvac:vent_boost
          - if door open → hvac:eco + advice
    """
    assignments = merge_plan["assignments"]
    cmds = []
    for rid, meta in rooms.items():
        c = counts.get(rid, 0)
        sens = sensors.get(rid, {})
        inactive_suggested = (assignments.get(rid, {}).get("active", True) is False)
        co2 = sens.get("co2", 450.0); lux = sens.get("lux", 200.0)
        door = sens.get("door", 0)

        if c == 0:
            cmds += [
                (rid, "lights", "off", {"reason": "vacant"}),
                (rid, "fan", "off", {"reason": "vacant"}),
                (rid, "hvac", "off", {"reason": "vacant"})
            ]
        else:
            light_payload = {"level": "dim"} if lux >= 600 else {}
            cmds.append((rid, "lights", "on", light_payload))
            cmds.append((rid, "fan", "on", {}))
            if door == 1:
                cmds.append((rid, "hvac", "eco", {"warning": "door_open"}))
                cmds.append((rid, "advice", "note", {"msg": "Close door to maintain cooling efficiency"}))
            else:
                if inactive_suggested:
                    cmds.append((rid, "hvac", "eco", {"note": "merge-suggested"}))
                else:
                    payload = {}
                    if co2 >= 1200:
                        payload["vent_boost"] = True
                    cmds.append((rid, "hvac", "comfort", payload))
    return cmds

def run_loop(rooms_json, occ_glob, sensors_csv, state_path, merges_out, commands_log, interval=5):
    rooms = load_rooms(rooms_json)
    sink = IoTSink(commands_log)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    os.makedirs(os.path.dirname(merges_out), exist_ok=True)

    while True:
        counts = latest_counts_from_glob(occ_glob)
        sensors = latest_sensors(sensors_csv)
        merge_plan = suggest_merges_to_common(rooms, counts)
        with open(merges_out, "w") as f:
            json.dump(merge_plan, f, indent=2)
        cmds = derive_commands(rooms, counts, merge_plan, sensors)
        last_cmds = [sink.send(r, d, c, p) for (r, d, c, p) in cmds]
        state = {"counts": counts, "merge_plan": merge_plan, "sensors": sensors, "last_commands": last_cmds}
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        time.sleep(interval)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms_json", default="data/rooms.json")
    ap.add_argument("--occ_glob", default="outputs/occupancy/*_counts.csv")
    ap.add_argument("--sensors_csv", default="outputs/sensors/sensors_latest.csv")
    ap.add_argument("--state_path", default="outputs/hvac/state.json")
    ap.add_argument("--merges_out", default="outputs/hvac/merges.json")
    ap.add_argument("--commands_log", default="outputs/hvac/commands.log")
    ap.add_argument("--interval", type=int, default=3)
    args = ap.parse_args()
    run_loop(args.rooms_json, args.occ_glob, args.sensors_csv, args.state_path, args.merges_out, args.commands_log, args.interval)

if __name__ == "__main__":
    main()
