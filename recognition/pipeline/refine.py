"""4.4 Validate & Refine — turn noisy per-frame predictions into a stable label."""

from collections import Counter


def refine(history: list[list], label: str | None, conf: float, *, window: int, min_confidence: float):
    """
    `history` is a list of [label, confidence] pairs (JSON-friendly), newest last.
    Returns (smoothed_label | None, smoothed_confidence, new_history).

    Majority vote over the last `window` frames; confidence is the mean
    confidence of the winning label across the window (frames that voted for
    another label count as 0). Below `min_confidence` the result is "uncertain".
    """
    history = (history + [[label, conf]])[-window:]
    votes = Counter(h[0] for h in history if h[0])
    if not votes:
        return None, 0.0, history
    best, _ = votes.most_common(1)[0]
    smoothed = sum(h[1] for h in history if h[0] == best) / len(history)
    if smoothed < min_confidence:
        return None, smoothed, history
    return best, smoothed, history
