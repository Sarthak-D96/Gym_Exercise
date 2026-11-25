"""Phase 3, steps 2 and 4–6 — clean keypoints, split by video, cut 16-frame windows.

Reads the .npz files from extract_keypoints.py and writes to --out:
    train.npz, val.npz, test.npz   X (N, 16, 42) features, kp (N, 16, 17, 3) raw keypoints,
                                   y (N,) class index, video (N,) video id
    classes.json                   class slugs, in label-index order
    splits.json                    video id -> split (same split for every experiment)
    summary.json                   counts per split / class, dropped videos

The split is done per video (never per frame or per window), stratified by class.
Clips cut from one source video must stay together: if their names show that
(e.g. 'squat_12_part3'), pass --group-regex so they share one group.

Example:
    python training/build_dataset.py --keypoints /data/gym/keypoints --out /data/gym/prepared
    # use the dataset's own noisy 'test' folder as the test set:
    python training/build_dataset.py --keypoints /data/gym/keypoints --test-keypoints /data/gym/keypoints_test \
        --out /data/gym/prepared
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import check_against_website, features_np, slugify


def clean(kp: np.ndarray, min_conf: float, max_gap: int):
    """Mark low-confidence frames missing, fill short gaps. Returns (kp, valid mask)."""
    kp = kp.copy()
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        score = np.nanmean(kp[..., 2], axis=1)
    valid = ~np.isnan(score) & (score >= min_conf)
    idx = np.flatnonzero(valid)
    if len(idx) >= 2:
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                for k in range(1, gap + 1):
                    t = k / (gap + 1)
                    kp[a + k] = (1 - t) * kp[a] + t * kp[b]
                    valid[a + k] = True
    return kp, valid


def runs(mask: np.ndarray):
    """Start/end of each run of True values."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def windows(kp, valid, size, stride, min_frames):
    out = []
    for s, e in runs(valid):
        seg = kp[s:e]
        if len(seg) >= size:
            out += [seg[i:i + size] for i in range(0, len(seg) - size + 1, stride)]
        elif len(seg) >= min_frames:  # short clip: repeat the last frame to fill the window
            out.append(np.concatenate([seg, np.repeat(seg[-1:], size - len(seg), 0)]))
    return out


def load_dir(root: Path, args, aliases):
    videos, dropped = [], Counter()
    files = sorted(root.rglob("*.npz"))
    if args.classes:
        # extract_keypoints.py writes <out>/<class folder>/<video>.npz: skip other classes
        # without opening them (opening is slow on network drives such as Google Drive)
        files = [f for f in files if aliases.get(slugify(f.parent.name), slugify(f.parent.name)) in args.classes]
    print(f"Loading {len(files)} keypoint files from {root} ...", flush=True)
    for n, f in enumerate(files, 1):
        if n % 100 == 0:
            print(f"  {n}/{len(files)}", flush=True)
        d = np.load(f, allow_pickle=False)
        label = slugify(str(d["label"]))
        label = aliases.get(label, label)
        if args.classes and label not in args.classes:
            continue
        vid = f.relative_to(root).with_suffix("").as_posix()
        kp, valid = clean(d["kp"], args.min_conf, args.max_gap)
        if valid.mean() < args.min_valid:
            dropped["mostly_missing"] += 1
            continue
        w = windows(kp, valid, args.window, args.stride, args.min_frames)
        if not w:
            dropped["too_short"] += 1
            continue
        m = re.search(args.group_regex, Path(vid).name) if args.group_regex else None
        group = f"{label}/{m.group(1) if m else Path(vid).name}"
        videos.append({"id": vid, "label": label, "group": group, "windows": np.stack(w)})
    return videos, dropped


def split_groups(videos, val_frac, test_frac, seed):
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for v in videos:
        by_class[v["label"]].append(v["group"])
    assign = {}
    for label, groups in by_class.items():
        groups = sorted(set(groups))
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(round(n * test_frac)) if n >= 3 else 0
        n_val = int(round(n * val_frac)) if n >= 3 else 0
        if test_frac > 0 and n >= 3:
            n_test = max(n_test, 1)
        if val_frac > 0 and n >= 3:
            n_val = max(n_val, 1)
        for i, g in enumerate(groups):
            assign[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return assign


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keypoints", required=True, type=Path)
    ap.add_argument("--test-keypoints", type=Path, help="optional separate test set (e.g. the dataset's test/)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--classes", nargs="*", help="keep only these class slugs (default: all)")
    ap.add_argument("--alias", nargs="*", default=[], metavar="OLD=NEW",
                    help="rename/merge classes, e.g. barbell_biceps_curl=bicep_curl push_up=push_up")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--max-gap", type=int, default=2)
    ap.add_argument("--min-valid", type=float, default=0.5, help="drop videos with fewer good frames than this")
    ap.add_argument("--min-frames", type=int, default=8, help="shortest clip that still gives a (padded) window")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--group-regex", default=None,
                    help=r"regex whose group(1) is the source-video id, e.g. '^(.*)_part\d+$'")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    aliases = dict(a.split("=", 1) for a in args.alias)
    if args.classes:
        args.classes = [slugify(c) for c in args.classes]

    videos, dropped = load_dir(args.keypoints, args, aliases)
    if not videos:
        raise SystemExit("No usable videos found.")
    if args.test_keypoints:
        test_videos, d2 = load_dir(args.test_keypoints, args, aliases)
        dropped.update({f"test_{k}": v for k, v in d2.items()})
        assign = split_groups(videos, args.val / (1 - args.test) if args.test < 1 else args.val, 0.0, args.seed)
        for v in videos:
            v["split"] = assign[v["group"]]
        for v in test_videos:
            v["split"] = "test"
            v["id"] = "test/" + v["id"]
        videos += test_videos
    else:
        assign = split_groups(videos, args.val, args.test, args.seed)
        for v in videos:
            v["split"] = assign[v["group"]]

    classes = sorted({v["label"] for v in videos})
    cidx = {c: i for i, c in enumerate(classes)}

    print(f"Loaded {len(videos)} videos; checking features and writing {args.out} ...", flush=True)
    sample = np.concatenate([v["windows"].reshape(-1, 17, 3) for v in videos[:50]])
    check_against_website(sample)
    print("Feature check passed: training features == website features.py")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "dropped_videos": dict(dropped), "splits": {}}
    for split in ("train", "val", "test"):
        vs = [v for v in videos if v["split"] == split]
        if not vs:
            print(f"  ! {split} split is empty")
            kp = np.zeros((0, args.window, 17, 3), np.float32)
            y, vid = np.zeros(0, np.int64), np.zeros(0, dtype="<U1")
        else:
            kp = np.concatenate([v["windows"] for v in vs]).astype(np.float32)
            y = np.concatenate([[cidx[v["label"]]] * len(v["windows"]) for v in vs]).astype(np.int64)
            vid = np.concatenate([[v["id"]] * len(v["windows"]) for v in vs])
        X = features_np(kp)
        np.savez_compressed(args.out / f"{split}.npz", X=X, kp=kp, y=y, video=vid)
        per_class = Counter(classes[i] for i in y)
        summary["splits"][split] = {"videos": len(vs), "windows": int(len(y)), "windows_per_class": dict(per_class)}
        print(f"{split:5s}: {len(vs):5d} videos, {len(y):7d} windows")

    (args.out / "classes.json").write_text(json.dumps(classes, indent=2))
    (args.out / "splits.json").write_text(json.dumps({v["id"]: v["split"] for v in videos}, indent=2))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Dropped videos: {dict(dropped)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
