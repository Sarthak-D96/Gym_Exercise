"""Learn what correct form looks like from the dataset -> form_profiles.json for the website.

The dataset has no "good form / bad form" labels, but its videos are demonstrations
of each exercise. So every training video is replayed through the website's own
rep counter and form measurements (recognition/pipeline/rep_counter.py, posture.py),
and the spread of each measurement over the dataset's reps becomes the normal
range the website grades against (Tukey's fence p75 + 1.5 IQR, capped at the 95th
percentile, so some sloppy reps in the dataset don't widen it; see posture.py).

Per exercise:
  1. Calibrate the rep-counting angles: the typical lowest and highest joint angle
     of each video (10th / 90th percentile), median over videos. Replaces the rough
     hand-set low/high angles in gym_config.py when the data supports it.
  2. Replay with those angles and collect every rep's depth, lockout, tempo, torso
     lean, hip line, elbow drift and left/right symmetry.
  3. Check on the held-out test videos (--test-keypoints): how many of their reps the
     grader would call good / minor / not counted. These are demonstration videos,
     so "not counted" there is mostly false rejection: keep it low.

Example (normally run by run_all.py, step "profiles"):
    python training/build_form_profiles.py --keypoints gym/keypoints --test-keypoints gym/keypoints_test \
        --exercises gym/exercises.json --fps 5 --out gym/form_profiles.json
Website:  python manage.py install_form_profiles form_profiles.json   (install_model does it too)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from common import KEYPOINT_NAMES, slugify
from recognition.pipeline import posture, rep_counter
from recognition.pipeline.types import ExerciseSpec, Keypoint

MIN_SPAN = 25.0      # calibrated low/high must be at least this far apart
MAX_SHIFT = 35.0     # ... and within this many degrees of the configured ones
MOVING_VIDEO = 20.0  # a video whose angle varies less than this has no usable reps


def load_videos(root: Path, folder_to_class: dict[str, str]) -> dict[str, list[np.ndarray]]:
    by_class: dict[str, list[np.ndarray]] = defaultdict(list)
    files = sorted(root.rglob("*.npz")) if root and root.exists() else []
    print(f"Reading {len(files)} keypoint files from {root} ...", flush=True)
    for f in files:
        cls = folder_to_class.get(slugify(f.parent.name))
        if cls is None:
            with np.load(f, allow_pickle=False) as d:
                cls = folder_to_class.get(slugify(str(d["label"])))
        if cls is None:
            continue
        with np.load(f, allow_pickle=False) as d:
            by_class[cls].append(d["kp"].astype(np.float32))
    return by_class


def frames(kp: np.ndarray):
    for row in kp:
        if np.isnan(row).any():
            yield None
        else:
            yield {n: Keypoint(float(row[i, 0]), float(row[i, 1]), float(row[i, 2])) for i, n in enumerate(KEYPOINT_NAMES)}


def replay(spec: ExerciseSpec, kp: np.ndarray, fps: float):
    """Feed one video through the website's rep counter. Yields (RepUpdate, frame index)."""
    state = rep_counter.RepState()
    for i, f in enumerate(frames(kp)):
        if f is not None:
            yield rep_counter.update(spec, state, f, i * 1000.0 / fps, strict=True), i


def make_spec(slug: str, web: dict, low=None, high=None, profile=None) -> ExerciseSpec:
    return ExerciseSpec(slug, web.get("name", slug), tuple(web["angle_points"]),
                        float(web["low_angle"] if low is None else low), float(web["high_angle"] if high is None else high),
                        web.get("start_phase", "high"), form_profile=profile)


def calibrate(spec: ExerciseSpec, videos: list[np.ndarray], fps: float) -> dict:
    lows, highs = [], []
    for kp in videos:
        angles = [u.angle for u, _ in replay(spec, kp, fps) if u.tracked and u.angle is not None]
        if len(angles) < 8:
            continue
        lo, hi = np.percentile(angles, [10, 90])
        if hi - lo >= MOVING_VIDEO:
            lows.append(lo)
            highs.append(hi)
    out = {"config_low": spec.low_angle, "config_high": spec.high_angle, "videos": len(lows)}
    if len(lows) >= 3:
        low, high = 5 * round(float(np.median(lows)) / 5), 5 * round(float(np.median(highs)) / 5)
        ok = (high - low >= MIN_SPAN and abs(low - spec.low_angle) <= MAX_SHIFT
              and abs(high - spec.high_angle) <= MAX_SHIFT)
        out.update(suggested_low=low, suggested_high=high, applied=ok)
        if ok:
            out.update(low=low, high=high)
    if "low" not in out:
        out.update(low=spec.low_angle, high=spec.high_angle, applied=False)
    return out


def percentiles(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    p = np.percentile(v, [5, 25, 50, 75, 95])
    return {"p5": round(p[0], 2), "p25": round(p[1], 2), "p50": round(p[2], 2), "p75": round(p[3], 2),
            "p95": round(p[4], 2), "n": int(len(v))}


def collect(spec: ExerciseSpec, videos: list[np.ndarray], fps: float):
    """Measurements of every rep that reached the target (holds: of every frame)."""
    values: dict[str, list[float]] = defaultdict(list)
    reps_per_video = []
    for kp in videos:
        n = 0
        for u, _ in replay(spec, kp, fps):
            if posture.is_static(spec):
                if u.tracked:
                    for k, v in u.frame.items():
                        if k != "body_dir":
                            values[k].append(v)
            elif u.rep_metrics is not None and not u.partial:
                n += 1
                for k, v in u.rep_metrics.items():
                    if k != "body_dir":
                        values[k].append(v)
        reps_per_video.append(n)
    return values, reps_per_video


def evaluate(spec: ExerciseSpec, videos: list[np.ndarray], fps: float) -> dict:
    """Grade the held-out videos' reps with the learned profile."""
    c = Counter()
    for kp in videos:
        for u, _ in replay(spec, kp, fps):
            if u.grade is None:
                continue
            c["attempts"] += 1
            c["partial" if u.partial else u.grade.severity] += 1
            for i in u.grade.issues:
                c[f"issue:{i.code}"] += 1
    if c["attempts"]:
        c["counted_share"] = round((c["good"] + c["minor"]) / c["attempts"], 3)
    return dict(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keypoints", required=True, type=Path, help="extract_keypoints.py output (training videos)")
    ap.add_argument("--test-keypoints", type=Path, help="held-out videos, only for the check in step 3")
    ap.add_argument("--exercises", required=True, type=Path,
                    help='JSON {slug: {"folders": [...], "website": {...}}} for the classes to profile')
    ap.add_argument("--fps", type=float, required=True, help="frame rate of the keypoint files (config.FPS)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    exercises = json.loads(args.exercises.read_text())
    folder_to_class = {slugify(f): slug for slug, ex in exercises.items() for f in [slug, *ex.get("folders", [])]}
    train = load_videos(args.keypoints, folder_to_class)
    test = load_videos(args.test_keypoints, folder_to_class) if args.test_keypoints else {}

    result = {"created": datetime.now().isoformat(timespec="seconds"), "fps": args.fps, "exercises": {}}
    print(f"\n{'exercise':22s} {'videos':>6s} {'reps':>6s}  angles (config -> data)   test: counted / attempts")
    for slug, ex in exercises.items():
        web, videos = ex["website"], train.get(slug, [])
        if not videos:
            print(f"{slug:22s} {0:6d}  no videos, skipped")
            continue
        base = make_spec(slug, web)
        cal = calibrate(base, videos, args.fps) if not posture.is_static(base) else {
            "low": base.low_angle, "high": base.high_angle, "applied": False, "videos": len(videos)}
        spec = make_spec(slug, web, cal["low"], cal["high"])
        values, reps = collect(spec, videos, args.fps)
        metrics = {k: percentiles(v) for k, v in values.items() if len(v) >= 3}
        entry = {"angles": cal, "videos": len(videos), "reps": int(sum(reps)),
                 "videos_without_reps": int(sum(1 for r in reps if r == 0)), "metrics": metrics}
        if test.get(slug):
            profiled = make_spec(slug, web, cal["low"], cal["high"], {"metrics": metrics})
            entry["test"] = evaluate(profiled, test[slug], args.fps)
        result["exercises"][slug] = entry
        t = entry.get("test", {})
        ang = (f"{cal['config_low']:.0f}/{cal['config_high']:.0f} -> {cal['low']:.0f}/{cal['high']:.0f}"
               if cal.get("applied") else f"{base.low_angle:.0f}/{base.high_angle:.0f} (kept)")
        test_txt = (f"{t.get('good', 0) + t.get('minor', 0)} / {t['attempts']}" if t.get("attempts") else "–")
        print(f"{slug:22s} {len(videos):6d} {entry['reps']:6d}  {ang:24s}  {test_txt}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.out}")
    print("Website: python manage.py install_form_profiles form_profiles.json  (install_model does it too)")


if __name__ == "__main__":
    main()
