#!/usr/bin/env python3
"""Fit a look image to a W x H render frame without cutting off the head (never crop the top of the head).

usage: prep_look.py IN.png OUT.jpg [W=768] [H=512]
Crops the source to the target aspect, keeping the top of the head: the crop band is placed so the estimated top of
the hair (face box top - 0.3 x face height; YuNet boxes start near the hairline) sits about 6% below the frame top. Cuts from the bottom first. Exits 1
when the head cannot fit (then the look needs regenerating with more headroom). Face detector: OpenCV YuNet
(path in $YUNET_ONNX, default ~/.cache/yunet.onnx; download face_detection_yunet_2023mar.onnx from github.com/opencv/opencv_zoo).
"""
import os, sys
import cv2

src, out = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 768
H = int(sys.argv[4]) if len(sys.argv) > 4 else 512
img = cv2.imread(src); h, w = img.shape[:2]
det = cv2.FaceDetectorYN.create(os.environ.get("YUNET_ONNX", os.path.expanduser("~/.cache/yunet.onnx")), "", (w, h), 0.6)
_, faces = det.detect(img)
if faces is None or len(faces) == 0:
    sys.exit("no face found")
x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4]
head_top = max(0.0, y - 0.3 * fh)
ta = W / H
if w / h > ta:  # too wide: crop the sides around the face
    cw, ch = int(round(h * ta)), h
    cx = int(min(max(x + fw / 2 - cw / 2, 0), w - cw)); cy = 0
else:           # too tall: crop top/bottom, keep the head
    cw, ch = w, int(round(w / ta))
    cy = int(max(0, min(head_top - 0.06 * ch, h - ch))); cx = 0
crop = img[cy:cy + ch, cx:cx + cw]
margin = (head_top - cy) / ch
print(f"{os.path.basename(src)}: crop {cw}x{ch} at ({cx},{cy}); hair top {margin:.0%} below the frame top")
if head_top < cy or margin < 0.05:
    print("FAIL: the head would touch or leave the top edge; regenerate the look with more headroom"); sys.exit(1)
cv2.imwrite(out, cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
