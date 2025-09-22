import cv2
import numpy as np
from typing import List, Tuple
from cv.config import ModelConfig

class YOLOOnnxDetector:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.net = cv2.dnn.readNetFromONNX(cfg.onnx_path)
        # Try CUDA if available and OpenCV built with it
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        except Exception:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int,int]]:
        h, w = frame.shape[:2]
        inp_w, inp_h = self.cfg.input_size
        # Letterbox
        r = min(inp_w / w, inp_h / h)
        nw, nh = int(w * r), int(h * r)
        canvas = np.full((inp_h, inp_w, 3), 114, dtype=np.uint8)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas[0:nh, 0:nw] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1/255.0, (inp_w, inp_h), swapRB=True, crop=False)
        return blob, r, (nw, nh)

    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
        """
        Returns list of detections as (x1,y1,x2,y2,score) only for class 'person'.
        Assumes YOLOv8-style ONNX output: [N, 84] -> cx,cy,w,h, conf, 80 class probs.
        """
        blob, r, (nw, nh) = self._preprocess(frame)
        self.net.setInput(blob)
        out = self.net.forward()  # shape: (1,N,84) for yolov8; some exports are (N,84)
        if out.ndim == 3:
            out = out[0]
        boxes, scores = [], []
        for det in out:
            conf = det[4]
            cls_scores = det[5:]
            cls_id = int(np.argmax(cls_scores))
            cls_conf = cls_scores[cls_id]
            score = conf * cls_conf
            if cls_id == self.cfg.person_class_id and score >= self.cfg.score_thresh:
                cx, cy, w, h = det[0], det[1], det[2], det[3]
                # undo letterbox scaling back to original frame
                x = (cx - w/2)
                y = (cy - h/2)
                # scale from input to original
                x /= 1.0; y /= 1.0; w /= 1.0; h /= 1.0  # identity; coordinates already in input scale
                # because we letterboxed at top-left, actual scale is r; need to map back:
                x = x * (1/r); y = y * (1/r); w = w * (1/r); h = h * (1/r)
                x1, y1 = int(max(0, x)), int(max(0, y))
                x2, y2 = int(min(frame.shape[1]-1, x + w)), int(min(frame.shape[0]-1, y + h))
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(score))

        # NMS
        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.cfg.score_thresh, self.cfg.nms_thresh)
        result = []
        if len(idxs) > 0:
            for i in (idxs.flatten() if hasattr(idxs, "flatten") else idxs):
                x1,y1,x2,y2 = boxes[i]
                result.append((x1,y1,x2,y2, scores[i]))
        return result
