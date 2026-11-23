"""Phase 3, step 3 — run YOLO-pose on every video and save COCO-17 keypoints.

Accepts either layout (mixed is fine), searched recursively under --data:

    <data>/<class>/<video>.mp4                      (the Kaggle videos)
    <data>/<class>/<video_folder>/frame_00001.jpg   (videos already split into frames)

The class of a video is the name of the folder it sits in. If --data points
straight at one class folder (e.g. .../bench_press), that folder's name is used.

Output: one file per video, <out>/<class>/<video>.npz with
    kp     (T, 17, 3) float32   x, y normalised 0..1 to the original frame, and score;
                                rows are NaN where no person was found
    label  class name as written on disk
Already-processed videos are skipped, so the script can be stopped and restarted.

Example:
    python training/extract_keypoints.py --data /data/gym/verified_data --out /data/gym/keypoints --device 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def find_items(root: Path):
    """Yield (kind, path) for each video file and each folder of frames."""
    items = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VID_EXT:
            items.append(("video", p))
        elif p.is_dir() and any(c.suffix.lower() in IMG_EXT for c in p.iterdir() if c.is_file()):
            items.append(("frames", p))
    if not items and any(c.suffix.lower() in IMG_EXT for c in root.iterdir()):
        items.append(("frames", root))
    return items


def best_person(result):
    """(17, 3) array for the biggest detected person, or None."""
    kps = result.keypoints
    if kps is None or kps.xyn is None or len(kps.xyn) == 0 or kps.conf is None:
        return None
    boxes = result.boxes.xywh.cpu().numpy()
    i = int(np.argmax(boxes[:, 2] * boxes[:, 3]))
    xy = kps.xyn[i].cpu().numpy()
    conf = kps.conf[i].cpu().numpy()
    return np.concatenate([xy, conf[:, None]], 1).astype(np.float32)


def frames_from_folder(folder: Path, step: int):
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT)
    return [str(f) for f in files[::step]]


def frames_from_video(path: Path, target_fps: float):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / target_fps))
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(fr)  # BGR numpy, which ultralytics expects
        i += 1
    cap.release()
    return frames, step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="dataset root")
    ap.add_argument("--out", required=True, type=Path, help="where to write keypoint .npz files")
    ap.add_argument("--pose-model", default="yolo11n-pose.pt",
                    help="ultralytics pose weights; use the same one the website will run")
    ap.add_argument("--fps", type=float, default=6.0, help="sampling rate (website sends ~6 fps)")
    ap.add_argument("--frames-fps", type=float, default=30.0,
                    help="frame rate the frame folders were extracted at (used to pick every Nth image)")
    ap.add_argument("--device", default=None, help="'0' for first GPU, 'cpu', 'mps' (Apple)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.pose_model)
    items = find_items(args.data)
    if not items:
        sys.exit(f"No videos or frame folders found under {args.data}")
    print(f"Found {len(items)} videos/frame folders")
    folder_step = max(1, round(args.frames_fps / args.fps))

    t0, done, skipped, failed = time.time(), 0, 0, 0
    for n, (kind, path) in enumerate(items, 1):
        if path == args.data:
            label, rel = path.name, path.name
        else:
            label = path.parent.name if path.parent != args.data else args.data.name
            rel = path.relative_to(args.data).with_suffix("").as_posix()
        # flatten the relative path into the file name so videos with the same
        # name in different sub-folders (data_btc / data_crawl) don't collide
        out = args.out / label / (rel.replace("/", "__") + ".npz")
        if out.exists():
            skipped += 1
            continue

        if kind == "frames":
            sources, step = frames_from_folder(path, folder_step), folder_step
        else:
            sources, step = frames_from_video(path, args.fps)
        if not sources:
            print(f"  ! could not read {path}")
            failed += 1
            continue

        kp = np.full((len(sources), 17, 3), np.nan, dtype=np.float32)
        for s in range(0, len(sources), args.batch):
            results = model.predict(sources[s:s + args.batch], device=args.device, imgsz=args.imgsz,
                                    verbose=False)
            for j, r in enumerate(results):
                p = best_person(r)
                if p is not None:
                    kp[s + j] = p

        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, kp=kp, label=label, step=step, source=str(path))
        done += 1
        if n % 20 == 0 or n == len(items):
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[{n}/{len(items)}] {rate:.2f} videos/s, last: {rel}")

    print(f"Done: {done} new, {skipped} already there, {failed} unreadable, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
