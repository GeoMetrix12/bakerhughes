import os
import cv2
import time
import argparse
import pandas as pd
from datetime import datetime
from typing import Optional

def _try_import_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except Exception:
        return None

def count_people_frame_ultralytics(model, frame, conf=0.35) -> int:
    res = model(frame, conf=conf, verbose=False)
    n = 0
    for r in res:
        for c in r.boxes.cls:
            if int(c) == 0:
                n += 1
    return n

def count_people_frame_hog(hog, frame) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects, _ = hog.detectMultiScale(gray, winStride=(4,4), padding=(8,8), scale=1.05)
    return len(rects)

def run_stream(source: str, room_id: str, out_csv: str, model_path: Optional[str] = None, conf: float = 0.35, show: bool = False):
    YOLO = _try_import_ultralytics()
    yolo_model = YOLO(model_path or "yolov8n.pt") if YOLO else None
    hog = None if YOLO else cv2.HOGDescriptor()
    if hog and not YOLO:
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []
    last_write = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count = count_people_frame_ultralytics(yolo_model, frame, conf) if yolo_model else count_people_frame_hog(hog, frame)
            ts = datetime.utcnow().isoformat()
            rows.append({"timestamp": ts, "room_id": room_id, "count": int(count)})

            if show:
                txt = f"{room_id}: {count}"
                cv2.putText(frame, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                cv2.imshow(f"Occupancy - {room_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if time.time() - last_write > 5 and rows:
                df = pd.DataFrame(rows)
                mode = 'a' if os.path.exists(out_csv) else 'w'
                header = not os.path.exists(out_csv)
                df.to_csv(out_csv, mode=mode, header=header, index=False)
                rows.clear()
                last_write = time.time()
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        if rows:
            df = pd.DataFrame(rows)
            mode = 'a' if os.path.exists(out_csv) else 'w'
            header = not os.path.exists(out_csv)
            df.to_csv(out_csv, mode=mode, header=header, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True, help="video file path, RTSP URL, or 'webcam'")
    ap.add_argument("--room_id", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--model_path", type=str, default=None, help="YOLO .pt path; defaults to yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    run_stream(args.source, args.room_id, args.out_csv, args.model_path, args.conf, args.show)

if __name__ == "__main__":
    main()
