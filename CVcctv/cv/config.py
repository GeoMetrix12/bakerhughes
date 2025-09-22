import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ModelConfig:
    onnx_path: str = os.getenv("YOLO_ONNX_PATH", "models/yolov8n.onnx")
    input_size: Tuple[int, int] = (640, 640)
    score_thresh: float = 0.3
    nms_thresh: float = 0.45
    person_class_id: int = 0  # in COCO, 'person' is class 0

@dataclass
class TrackerConfig:
    max_age: int = 15          # frames to keep a lost track
    iou_match_thresh: float = 0.3
    min_hits: int = 2          # warm-up before counting as a stable track

@dataclass
class StreamConfig:
    source: str = os.getenv("VIDEO_SOURCE", "0")  # "0" for default webcam, or path to file
    write_annotated: bool = True
    out_video_path: str = "outputs/cv/annotated.mp4"
    fps: int = 25

@dataclass
class OutputConfig:
    snap_csv: str = "outputs/live/cv_people_counts_live.csv"
    log_csv: str = "outputs/live/cv_people_counts_log.csv"

@dataclass
class ZonesConfig:
    # Simple rectangles per zone: (name, x1,y1,x2,y2) in pixel coords of the input frames
    # For demo; you can switch to polygons later.
    zones: List[Tuple[str, int, int, int, int]] = (
        ("Lobby",  20,  20, 300, 300),
        ("OpenArea", 320,  20, 620, 260),
        ("Corridor", 20, 320, 620, 470),
    )

@dataclass
class SyntheticConfig:
    enable: bool = False       # If True, ignore model and synthesize detections
    num_people: int = 12
    width: int = 640
    height: int = 480
    speed_px: int = 4
    jitter_px: int = 2
