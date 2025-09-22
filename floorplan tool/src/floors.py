import json
from typing import Dict, List, Tuple

def load_floors(path: str = "data/floors.json") -> Tuple[Dict[tuple, dict], List[dict]]:
    data = json.load(open(path, "r"))
    floors_list = data["floors"]
    idx = {}
    for f in floors_list:
        fid = f["floor_id"]
        for r in f["rooms"]:
            r2 = r.copy(); r2["floor_id"] = fid
            idx[(fid, r["room_id"])] = r2
    return idx, floors_list

def rooms_by_floor(floors_list: List[dict], floor_id: str) -> List[dict]:
    for f in floors_list:
        if f["floor_id"] == floor_id:
            return f["rooms"]
    return []
