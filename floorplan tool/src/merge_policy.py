from typing import Dict, List

def suggest_merges_to_common(floors_list: list, counts_by_floor: Dict[str, Dict[str, int]]) -> dict:
    all_assignments = {}
    all_suggestions: List[dict] = []
    saved_area = 0.0

    for F in floors_list:
        fid = F["floor_id"]
        # zone groups
        by_zone = {}
        for r in F["rooms"]:
            by_zone.setdefault(r["zone"], []).append(r)

        assignments = {r["room_id"]: {"assigned": counts_by_floor.get(fid, {}).get(r["room_id"], 0),
                                      "active": counts_by_floor.get(fid, {}).get(r["room_id"], 0) > 0}
                       for r in F["rooms"]}

        for zone, rooms in by_zone.items():
            targets = [r for r in rooms if r.get("is_common", False)]
            targets.sort(key=lambda x: x["capacity"])
            sources = [r for r in rooms if not r.get("is_common", False)]
            sources.sort(key=lambda x: counts_by_floor.get(fid, {}).get(x["room_id"], 0))

            for src in sources:
                c = counts_by_floor.get(fid, {}).get(src["room_id"], 0)
                if c == 0:
                    continue
                remaining = c
                moves = []
                for tgt in targets:
                    cap = tgt["capacity"]
                    cur = assignments[tgt["room_id"]]["assigned"]
                    free = max(cap - cur, 0)
                    if free <= 0:
                        continue
                    move = min(remaining, free)
                    if move > 0:
                        assignments[tgt["room_id"]]["assigned"] = cur + move
                        remaining -= move
                        moves.append({"to": tgt["room_id"], "count": move})
                    if remaining == 0:
                        break
                if remaining == 0:
                    assignments[src["room_id"]]["assigned"] = 0
                    assignments[src["room_id"]]["active"] = False
                    all_suggestions.append({
                        "floor_id": fid,
                        "action": "merge_to_common",
                        "from": src["room_id"],
                        "moves": moves,
                        "zone": zone
                    })

        for r in F["rooms"]:
            rid = r["room_id"]
            if counts_by_floor.get(fid, {}).get(rid, 0) > 0 and assignments[rid]["active"] is False:
                saved_area += r["area_m2"]

        all_assignments[fid] = assignments

    return {"assignments": all_assignments, "suggestions": all_suggestions, "saved_area_m2": saved_area}
