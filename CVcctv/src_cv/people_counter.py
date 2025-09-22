from typing import List, Tuple, Dict
import numpy as np
import cv2

def in_rect(pt, rect):
    x, y = pt
    x1,y1,x2,y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def box_center(box):
    x1,y1,x2,y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)

def draw_overlay(frame, tracks, zones, color_person=(31,182,255), color_text=(255,255,255)):
    # zones
    for name, x1,y1,x2,y2 in zones:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (80,120,200), 1)
        cv2.putText(frame, name, (x1+4,y1+16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,200,255), 1, cv2.LINE_AA)
    # tracks
    for t in tracks:
        x1,y1,x2,y2 = t.box
        cv2.rectangle(frame, (x1,y1), (x2,y2), color_person, 2)
        cx, cy = box_center(t.box)
        cv2.circle(frame, (cx,cy), 2, color_person, -1)
        cv2.putText(frame, f"ID {t.tid}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1, cv2.LINE_AA)
    return frame

def counts_per_zone(tracks, zones) -> Dict[str, int]:
    c = {name: 0 for (name, *_rest) in zones}
    for t in tracks:
        cx, cy = box_center(t.box)
        for name, x1,y1,x2,y2 in zones:
            if in_rect((cx,cy), (x1,y1,x2,y2)):
                c[name] += 1
                break
    return c
