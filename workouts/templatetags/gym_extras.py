from django import template

register = template.Library()


@register.filter
def duration(seconds):
    """Format seconds as H:MM:SS or M:SS."""
    try:
        s = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "–"
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@register.filter
def percent(value):
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return "–"


@register.filter
def level_badge(level):
    return {"good": "badge-good", "warning": "badge-warn", "info": "badge-info"}.get(level, "")
