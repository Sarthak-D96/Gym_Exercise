import math

from .types import Keypoint


def angle(a: Keypoint, b: Keypoint, c: Keypoint) -> float:
    """Angle ABC in degrees (0..180), measured at keypoint b."""
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def midpoint(a: Keypoint, b: Keypoint) -> Keypoint:
    return Keypoint((a.x + b.x) / 2, (a.y + b.y) / 2, min(a.score, b.score))


def tilt_from_vertical(bottom: Keypoint, top: Keypoint) -> float:
    """Angle in degrees between segment bottom->top and the upward vertical."""
    dx, dy = top.x - bottom.x, bottom.y - top.y
    if dx == 0 and dy == 0:
        return 0.0
    return abs(math.degrees(math.atan2(dx, dy)))
