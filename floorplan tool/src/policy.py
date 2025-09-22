from typing import Dict, Any, List

def derive_commands(floors_list: list,
                    counts: Dict[str,Dict[str,int]],
                    assignments: Dict[str,Dict[str,dict]],
                    sensors: Dict[str,Dict[str,dict]]) -> List[dict]:
    cmds = []
    for F in floors_list:
        fid = F["floor_id"]
        for r in F["rooms"]:
            rid = r["room_id"]
            c = counts.get(fid, {}).get(rid, 0)
            sens = sensors.get(fid, {}).get(rid, {}) if sensors else {}
            inactive = (assignments.get(fid, {}).get(rid, {}).get("active", True) is False)
            co2 = sens.get("co2", 450.0); lux = sens.get("lux", 200.0); door = sens.get("door", 0)

            if c == 0:
                cmds += [
                    {"floor_id": fid, "room_id": rid, "device":"lights","command":"off","payload":{"reason":"vacant"}},
                    {"floor_id": fid, "room_id": rid, "device":"fan","command":"off","payload":{"reason":"vacant"}},
                    {"floor_id": fid, "room_id": rid, "device":"hvac","command":"off","payload":{"reason":"vacant"}}
                ]
            else:
                light_payload = {"level":"dim"} if lux >= 600 else {}
                cmds.append({"floor_id": fid,"room_id": rid,"device":"lights","command":"on","payload":light_payload})
                cmds.append({"floor_id": fid,"room_id": rid,"device":"fan","command":"on","payload":{}})
                if door == 1:
                    cmds.append({"floor_id": fid,"room_id": rid,"device":"hvac","command":"eco","payload":{"warning":"door_open"}})
                    cmds.append({"floor_id": fid,"room_id": rid,"device":"advice","command":"note","payload":{"msg":"Close door to maintain efficiency"}})
                else:
                    if inactive:
                        cmds.append({"floor_id": fid,"room_id": rid,"device":"hvac","command":"eco","payload":{"note":"merge-suggested"}})
                    else:
                        payload = {"vent_boost": True} if co2 >= 1200 else {}
                        cmds.append({"floor_id": fid,"room_id": rid,"device":"hvac","command":"comfort","payload":payload})
    return cmds
