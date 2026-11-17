"""Launch-day Twitch viewer forecaster for the GTA VI category (v4).

Design, in the order the data flows:

1. Pool selection by unlock type (kept from the previous entry): a
   launch day on Twitch is driven by rolling midnight-local unlocks, so
   the shape worth learning comes from past launches whose unlock note
   says the same -- console launches that rolled out at midnight local,
   with no simultaneous global unlock.  The target game's own
   pre-release category history is never training data.  Only if the
   rolling pool ever came back too small does it widen, first to the
   already-launched comparables, then to every complete one.

2. Centre by local intercepts + one smooth size slope.  A joint
   least-squares fit of every pool curve's log ratio
   log(viewers_h / viewers_1) on a quadratic basis in the hour, with the
   basis also interacted with (log10 size - target log10 size), gives a
   single smooth slope curve: how the whole day shape tilts with launch
   size.  Each hour's centre is then that hour's pooled mean ratio plus
   the smooth slope times the target's size offset.  (Per-hour
   independent slopes were noisy with eight pool curves, and a
   size-kernel weighting threw away most of the pool; both measurably
   lost to this combination on the kit's practice launches and on
   leave-one-out over the pool itself.)

3. Quantiles from pooled standardised residuals.  Each pool curve's
   residual against its own fitted line is divided by its hour's
   residual spread (the spread curve itself smoothed over neighbouring
   hours, since eight residuals per hour estimate it poorly); the
   standardised residuals are pooled within three hour blocks (early,
   mid, late day) and their weighted quantiles, rescaled per hour, give
   the five forecast quantiles.  A final 0.95 factor on the spread
   around the median is the calibration the practice launches and the
   mimicked final (each practice launch scored with the other three in
   its training pool) both preferred.

Validation (local grader, exact contest pinball formula, real
practice answers): mean loss 0.1564 over the kit's four practice
launches -- per case 0.1869 / 0.1380 / 0.1033 / 0.1973 -- against
0.1975 for the previous entry (per-hour kernel local-linear fits),
0.2496 for the median shape on the same pool, and 0.3871 for the house
persistence baseline.  Leave-one-out over the eight pool curves gives
0.3444 (previous entry 0.3410), and the mimicked final -- each
practice launch scored with the other three added to its pool, the
shape of the real final case -- gives 0.1562 against the previous
entry's 0.1700.  The one soft spot is unchanged: the fourth practice
launch's late-day surge still escapes the 90% band (coverage 0.70
there, 1.00 on the other three).

Everything is computed from the case file at run time: no hard-coded
curve, no network, standard library only, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = list(range(2, 25))
BLOCKS = [(2, 8), (9, 16), (17, 24)]

# Fallback pool rule (used only if the rolling-midnight pool ever came
# back too small): a comparable counts as "already launched" when its
# hour 1 is at least this share of its own launch-day peak.
MIN_HOUR1_SHARE = 0.12
# The fitted size slope is clipped to this absolute value per hour.
SLOPE_CLIP = 1.5
# Residual spreads are floored here (in log space) before standardising.
STD_FLOOR = 0.05
# The residual spread curve is averaged over this many neighbouring
# hours: with a handful of pool curves the per-hour estimate is noise.
STD_SMOOTH_WINDOW = 5
# Final calibration of the band around the median.
SPREAD = 0.95


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(value):
    scaled = math.floor(value * 100 + 0.5)
    out = scaled / 100
    return int(out) if float(out).is_integer() else out


def weighted_quantile_interp(pairs, tau):
    """Weighted quantile with linear interpolation on cumulative weight."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = tau * total
    running = 0.0
    prev_value, prev_cum = ordered[0][0], 0.0
    for value, weight in ordered:
        running += weight
        if running >= target:
            if running == prev_cum:
                return value
            frac = (target - prev_cum) / (running - prev_cum)
            return prev_value + frac * (value - prev_value)
        prev_value, prev_cum = value, running
    return ordered[-1][0]


def weighted_std(pairs):
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    mean = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    return math.sqrt(max(var, 0.0))


def unlock_is_rolling_midnight(comp):
    """True for a console launch that unlocked at midnight local, rolling.

    The case's comparables carry a free-text unlock note ending in a
    Tier tag.  GTA VI itself is a console, rolling-midnight launch with
    no early access; the closest analogues share that unlock shape.
    Simultaneous global unlocks (one worldwide hour) have a different
    day curve and are left out.
    """
    note = str(comp.get("unlock", "")).lower()
    if "tier: comparable" not in note:
        return False
    if "midnight local" not in note:
        return False
    if "single global" in note or "simultaneous" in note:
        return False
    return True


def is_target_history(comp):
    """True for the target game's own pre-release category history.

    It is not a launch at all, so its "release-day curve" would be a
    premiere day's sparse points.  It must never enter training.
    """
    note = str(comp.get("unlock", "")).lower()
    return "tier: target-history" in note or "not a launch" in note


def comparable_curve(comp):
    """One comparable's launch-day log ratios, or None if it is unusable.

    Returns (log10 of hour-1 viewers, {hour: log(viewers_h / viewers_1)},
    hour-1 share of the launch-day peak, rolling-midnight flag).
    """
    if "before release" in comp.get("name", "") or is_target_history(comp):
        return None
    try:
        start = parse_time(comp["day_start"])
    except (KeyError, ValueError):
        return None
    by_hour = {}
    for point in comp.get("hourly", []):
        try:
            offset = int(round((parse_time(point["from"]) - start).total_seconds() / 3600))
            viewers = float(point["viewers"])
        except (KeyError, ValueError, TypeError):
            continue
        if 0 <= offset < 24 and viewers > 0:
            by_hour[offset + 1] = viewers
    if 1 not in by_hour or len(by_hour) < 23:
        return None
    first = by_hour[1]
    peak = max(by_hour.values())
    if peak <= 0:
        return None
    ratios = {}
    for hour in range(2, 25):
        if hour in by_hour:
            ratios[hour] = math.log(by_hour[hour] / first)
    if len(ratios) < 20:
        return None
    return math.log10(first), ratios, first / peak, unlock_is_rolling_midnight(comp)


def build_pool(case):
    """The pool to learn from: rolling-midnight launches first.

    The rolling pool is the primary training set.  Only if it ever came
    back with fewer than three usable curves does the pool widen -- to
    the already-launched comparables, then to every complete one --
    rather than forecasting from a handful of launches.
    """
    complete = []
    for comp in case.get("comparables") or []:
        curve = comparable_curve(comp)
        if curve is not None:
            complete.append(curve)
    rolling = [c for c in complete if c[3]]
    if len(rolling) >= 3:
        return rolling
    launched = [c for c in complete if c[2] >= MIN_HOUR1_SHARE]
    return launched if len(launched) >= 8 else complete


def basis(hour):
    t = (hour - 2) / 22.0
    return [1.0, t, t * t]


def solve_linear(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            M[piv][col] = 1e-12
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        M[col] = [v / d for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                M[r] = [a - f * c for a, c in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def smooth_slope_curve(pool, log_size):
    """One smooth slope of log ratio on log10 size, per hour.

    Joint weighted least squares over every (curve, hour) point:
    ratio ~ sum_k a_k phi_k(h) + sum_k b_k phi_k(h) (size - target size).
    The slope at hour h is sum_k b_k phi_k(h), clipped for safety.
    """
    K = len(basis(2))
    P = 2 * K
    A = [[0.0] * P for _ in range(P)]
    b = [0.0] * P
    for size, ratios, _share, _roll in pool:
        dx = size - log_size
        for hour in HOURS:
            if hour not in ratios:
                continue
            phi = basis(hour)
            row = phi + [p * dx for p in phi]
            y = ratios[hour]
            for i in range(P):
                b[i] += row[i] * y
                for j in range(P):
                    A[i][j] += row[i] * row[j]
    for i in range(P):
        A[i][i] += 1e-8
    beta = solve_linear(A, b)
    slopes = {}
    for hour in HOURS:
        phi = basis(hour)
        s = sum(beta[K + k] * phi[k] for k in range(K))
        slopes[hour] = max(-SLOPE_CLIP, min(SLOPE_CLIP, s))
    return slopes


def hour_curves(pool, log_size):
    """Per-hour quantile log ratios: local intercepts, smooth slope,
    pooled standardised residuals for the band."""
    slopes = smooth_slope_curve(pool, log_size)
    centre, resid = {}, {}
    for hour in HOURS:
        rows = [(size, ratios[hour]) for size, ratios, _s, _r in pool if hour in ratios]
        if not rows:
            centre[hour] = None
            resid[hour] = []
            continue
        mean_x = sum(s for s, _ in rows) / len(rows)
        mean_y = sum(y for _, y in rows) / len(rows)
        slope = slopes[hour]
        centre[hour] = mean_y + slope * (log_size - mean_x)
        resid[hour] = [(y - (mean_y + slope * (s - mean_x)), 1.0) for s, y in rows]
    # Per-hour residual spread, smoothed over neighbouring hours.
    stds = {}
    for hour in HOURS:
        stds[hour] = max(STD_FLOOR, weighted_std(resid[hour])) if resid[hour] else STD_FLOOR
    half = STD_SMOOTH_WINDOW // 2
    smooth_stds = {}
    for hour in HOURS:
        ns = [stds[k] for k in range(hour - half, hour + half + 1) if k in stds]
        smooth_stds[hour] = sum(ns) / len(ns)
    # Standardised residuals pooled per hour block -> block quantiles.
    block_q = {}
    for lo, hi in BLOCKS:
        pooled = []
        for hour in range(lo, hi + 1):
            for v, w in resid[hour]:
                pooled.append((v / smooth_stds[hour], w))
        block_q[(lo, hi)] = {tau: weighted_quantile_interp(pooled, tau) for tau in LEVELS}
    raw = {}
    for hour in HOURS:
        if centre[hour] is None:
            raw[hour] = None
            continue
        lo, hi = next(b for b in BLOCKS if b[0] <= hour <= b[1])
        raw[hour] = [centre[hour] + smooth_stds[hour] * block_q[(lo, hi)][tau] for tau in LEVELS]
    fitted = [h for h in HOURS if raw[h] is not None]
    for hour in HOURS:
        if raw[hour] is None:
            nearest = min(fitted, key=lambda h: (abs(h - hour), h > hour))
            raw[hour] = list(raw[nearest])
    # Final calibration around the median; enforce monotonicity.
    out = {}
    for hour in HOURS:
        med = raw[hour][2]
        row = [med + (q - med) * SPREAD for q in raw[hour]]
        for j in range(1, len(LEVELS)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        out[hour] = row
    return out


def baseline_quantiles(p1, hour):
    """The house persistence shape, used only when no pool is available."""
    return [round2(p1 * math.exp(z * 0.15 * math.sqrt(hour - 1))) for z in Z]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    pool = build_pool(case)
    hours = []
    if not pool:
        for row in case["forecast"]:
            hours.append({"hour": row["hour"], "q": baseline_quantiles(p1, row["hour"])})
        return {"hours": hours}
    curves = hour_curves(pool, math.log10(p1))
    for row in case["forecast"]:
        hour = row["hour"]
        quantiles = []
        previous = 0.0
        for log_ratio in curves[hour]:
            value = p1 * math.exp(log_ratio)
            value = min(max(value, 0.01), 1_000_000_000.0)
            value = max(value, previous)
            quantiles.append(round2(value))
            previous = float(quantiles[-1])
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case, path.name)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
