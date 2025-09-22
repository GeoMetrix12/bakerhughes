import os, time, json
import cv2
import numpy as np
import pandas as pd
from typing import Optional

from cv.config import ModelConfig, TrackerConfig, StreamConfig, OutputConfig, ZonesConfig, SyntheticConfig
from src_cv.yolo_onnx import YOLOOnnxDetector
from src_cv.iou_tracker import IOUTracker
from src_cv.people_counter import draw_overlay, counts_per_zone
from src_cv.synthetic_stream import SyntheticPeople

def load_zones_from_json(path: str, default: ZonesConfig) -> ZonesConfig:
    if path and os.path.exists(path):
        import json
        js = json.load(open(path, "r"))
        return ZonesConfig(zones=[tuple(z) for z in js.get("zones", default.zones)])
    return default

def ensure_dirs(out_csv: str, out_video: Optional[str]):
    d = os.path.dirname(out_csv)
    if d: os.makedirs(d, exist_ok=True)
    if out_video:
        d2 = os.path.dirname(out_video)
        if d2: os.makedirs(d2, exist_ok=True)

def write_snap_and_log(snap_csv, log_csv, ts, zone_counts, total, frame_idx):
    snap_df = pd.DataFrame([{"timestamp": ts, **zone_counts, "total": total, "frame": frame_idx}])
    snap_df.to_csv(snap_csv, index=False)
    # append to log
    mode = "a" if os.path.exists(log_csv) else "w"
    header = not os.path.exists(log_csv)
    snap_df.to_csv(log_csv, mode=mode, header=header, index=False)

def run(
    use_synthetic: bool = False,
    zones_json: Optional[str] = None,
    source: str = "0",
    onnx_path: str = "models/yolov8n.onnx",
    write_annotated: bool = True,
    out_video_path: str = "outputs/cv/annotated.mp4",
    fps: int = 25
):
    mcfg = ModelConfig(onnx_path=onnx_path)
    tcfg = TrackerConfig()
    scfg = StreamConfig(source=source, write_annotated=write_annotated, out_video_path=out_video_path, fps=fps)
    ocfg = OutputConfig()
    zcfg = load_zones_from_json(zones_json, ZonesConfig())
    syn = SyntheticConfig(enable=use_synthetic)

    detector = None if use_synthetic else YOLOOnnxDetector(mcfg)
    tracker = IOUTracker(max_age=tcfg.max_age, iou_match_thresh=tcfg.iou_match_thresh, min_hits=tcfg.min_hits)

    ensure_dirs(ocfg.snap_csv, scfg.out_video_path if scfg.write_annotated else None)

    if use_synthetic:
        synth = SyntheticPeople(syn, zcfg)
        W, H = syn.width, syn.height
        writer = None
    else:
        cap = cv2.VideoCapture(0 if source == "0" else source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if scfg.write_annotated:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(scfg.out_video_path, fourcc, scfg.fps, (W, H))
        else:
            writer = None

    frame_idx = 0
    try:
        while True:
            ts = pd.Timestamp.utcnow().isoformat()

            if use_synthetic:
                frame, dets = synth.step()  # dets: (x1,y1,x2,y2,score)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                dets = detector.detect_persons(frame)

            tracks = tracker.update(dets)
            # counts
            zcounts = counts_per_zone(tracks, zcfg.zones)
            total = int(sum(zcounts.values()))
            write_snap_and_log(ocfg.snap_csv, ocfg.log_csv, ts, zcounts, total, frame_idx)

            # draw overlay
            vis = frame.copy()
            # draw zones and tracks
            vis = draw_overlay(vis, tracks, zcfg.zones)
            # overlay HUD
            hud = f"Total: {total} | Zones: " + ", ".join([f"{k}:{v}" for k,v in zcounts.items()])
            cv2.putText(vis, hud, (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(vis)

            # Preview window (press q to quit)
            cv2.imshow("People Counting", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    finally:
        if not use_synthetic:
            cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Examples:
    # Synthetic demo (no camera/model): python -m src_cv.run_cv --synthetic 1
    # Webcam with YOLO ONNX:           python -m src_cv.run_cv --source 0 --onnx models/yolov8n.onnx
    import argparse, pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=0, help="1 to enable synthetic mode")
    ap.add_argument("--zones_json", type=str, default="cv/zones.json", help="zones override JSON")
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--onnx", type=str, default="models/yolov8n.onnx")
    ap.add_argument("--write_annotated", type=int, default=1)
    ap.add_argument("--out", type=str, default="outputs/cv/annotated.mp4")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    run(
        use_synthetic=bool(args.synthetic),
        zones_json=args.zones_json if args.zones_json else None,
        source=args.source,
        onnx_path=args.onnx,
        write_annotated=bool(args.write_annotated),
        out_video_path=args.out,
        fps=args.fps
    )
