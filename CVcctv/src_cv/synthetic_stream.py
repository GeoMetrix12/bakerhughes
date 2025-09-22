import numpy as np
import cv2
from typing import List, Tuple
from cv.config import SyntheticConfig, ZonesConfig

class SyntheticPeople:
    def __init__(self, cfg: SyntheticConfig, zones: ZonesConfig):
        self.cfg = cfg
        self.zones = zones
        self.W, self.H = cfg.width, cfg.height
        rng = np.random.default_rng(1337)
        self.cx = rng.integers(30, self.W-30, size=cfg.num_people).astype(np.int32)
        self.cy = rng.integers(30, self.H-30, size=cfg.num_people).astype(np.int32)
        self.vx = rng.integers(-cfg.speed_px, cfg.speed_px+1, size=cfg.num_people).astype(np.int32)
        self.vy = rng.integers(-cfg.speed_px, cfg.speed_px+1, size=cfg.num_people).astype(np.int32)

    def step(self):
        rng = np.random.default_rng()
        self.vx += rng.integers(-self.cfg.jitter_px, self.cfg.jitter_px+1, size=self.cfg.num_people)
        self.vy += rng.integers(-self.cfg.jitter_px, self.cfg.jitter_px+1, size=self.cfg.num_people)
        self.vx = np.clip(self.vx, -self.cfg.speed_px, self.cfg.speed_px)
        self.vy = np.clip(self.vy, -self.cfg.speed_px, self.cfg.speed_px)

        self.cx += self.vx
        self.cy += self.vy
        self.cx = np.clip(self.cx, 20, self.W-20)
        self.cy = np.clip(self.cy, 20, self.H-20)

        boxes = []
        for x, y in zip(self.cx, self.cy):
            w, h = 26, 52
            boxes.append((int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2), 0.9))
        # render frame
        frame = np.full((self.H, self.W, 3), 18, dtype=np.uint8)
        # zones
        for name, x1,y1,x2,y2 in self.zones.zones:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (60,60,60), -1)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (120,120,120), 1)
        for (x1,y1,x2,y2,_) in boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)
        return frame, boxes
