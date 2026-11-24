"""boltgym: a small gym for one CAD skill, moving a hole pattern from one part to another.

Standard library only. See README.md.
"""
import json
import math
import random
import sys

# Normal-fit clearance hole for each metric thread size, in mm (the ISO 273 medium series;
# the same numbers are the normal column of ASME B18.2.8).
CLEARANCE = {3.0: 3.4, 4.0: 4.5, 5.0: 5.5, 6.0: 6.6, 8.0: 9.0, 10.0: 11.0}
STANDARD_DIAMETERS = sorted(CLEARANCE.values())
ANCHORS = ("bottom-left", "bottom-right", "top-left", "top-right")
SKILLS = ("transfer", "clearance")

POS_TOL = 0.05   # mm, a hole centre counts as correct inside this
DIA_TOL = 0.01   # mm
GRID = 0.5       # mm, every generated coordinate is a multiple of this
EXTRA_PENALTY = 0.10
RULE_PENALTY = 0.05


def snap(v):
    return round(v / GRID) * GRID


def assign(cost):
    """Minimum-cost assignment for a square matrix (Hungarian algorithm, O(n^3)).

    Returns a list where result[row] is the column given to that row.
    """
    n = len(cost)
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = [0] * n
    for j in range(1, n + 1):
        out[p[j] - 1] = j - 1
    return out


def assign_greedy(cost):
    """Each row takes the cheapest free column, in row order. Kept for comparison."""
    n = len(cost)
    free = set(range(n))
    out = []
    for i in range(n):
        j = min(free, key=lambda c: (cost[i][c], c))
        free.discard(j)
        out.append(j)
    return out


def pair_score(a, b):
    """Similarity of two holes (x, y, d) in [0, 1].

    The position term mixes a sharp part (so the last millimetre matters) with a
    long tail (so a hole 80 mm away still gets a signal about which way to go).
    """
    dist = math.hypot(a[0] - b[0], a[1] - b[1])
    pos = 0.5 * math.exp(-dist / 2.0) + 0.5 / (1.0 + dist / 25.0)
    dia = 1.0 / (1.0 + abs(a[2] - b[2]))
    return pos * dia


def rule_violations(holes, width, height):
    """Count manufacturing rule breaks.

    Edge rule: a hole centre must be at least one diameter from every plate edge.
    Spacing rule: two holes need at least 1 mm of material between them.
    """
    n = 0
    for (x, y, d) in holes:
        if x < d or y < d or x > width - d or y > height - d:
            n += 1
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            a, b = holes[i], holes[j]
            if math.hypot(a[0] - b[0], a[1] - b[1]) < (a[2] + b[2]) / 2.0 + 1.0:
                n += 1
    return n


def verify(placed, expected, width, height, matcher=assign):
    """Compare placed holes with the expected ones. Returns a dict with score and success.

    score is 1.0 for a perfect result and can go below 0 when penalties pile up.
    """
    ne, npl = len(expected), len(placed)
    m = max(ne, npl)
    matched = 0.0
    exact = npl == ne
    worst_pos = 0.0
    if m and ne:
        cost = [[1.0] * m for _ in range(m)]
        for i in range(ne):
            for j in range(npl):
                cost[i][j] = 1.0 - pair_score(expected[i], placed[j])
        cols = matcher(cost)
        for i in range(ne):
            j = cols[i]
            if j >= npl:
                exact = False
                continue
            e, q = expected[i], placed[j]
            matched += pair_score(e, q)
            dist = math.hypot(e[0] - q[0], e[1] - q[1])
            worst_pos = max(worst_pos, dist)
            if dist > POS_TOL or abs(e[2] - q[2]) > DIA_TOL:
                exact = False
    base = matched / ne if ne else 0.0
    extras = max(0, npl - ne)
    broken = rule_violations(placed, width, height)
    score = base - EXTRA_PENALTY * extras - RULE_PENALTY * broken
    return {
        "score": score,
        "success": bool(exact and broken == 0 and ne > 0),
        "extras": extras,
        "violations": broken,
        "worst_position_error": worst_pos,
    }


def to_target(hole, anchor, src, dst):
    """Keep a hole's offsets from the named corner when moving from src to dst (w, h)."""
    x, y, d = hole
    if "right" in anchor:
        x = dst[0] - (src[0] - x)
    if "top" in anchor:
        y = dst[1] - (src[1] - y)
    return (x, y, d)


# Generator settings. "default" is what the baselines were developed on. "heldout" was
# written afterwards for evaluation only: larger plates, a 0.1 mm coordinate grid, wider
# pitch and radius ranges, and a staggered two-row pattern the default never produces.
PROFILES = {
    "default": {"grid": GRID, "w": (90, 160), "h": (80, 140), "delta": (8, 40),
                "kinds": ("grid", "circle", "line"), "pitch": (14, 30), "line_pitch": (14, 24),
                "radius": (14, 28), "margin": 12},
    "heldout": {"grid": 0.1, "w": (150, 320), "h": (120, 260), "delta": (10, 60),
                "kinds": ("grid", "circle", "line", "stagger"), "pitch": (11, 42),
                "line_pitch": (11, 38), "radius": (18, 55), "margin": 9},
}


def _pattern(rng, w, h, prof):
    g, mg = prof["grid"], prof["margin"]

    def sn(v):
        return round(round(v / g) * g, 6)

    kind = rng.choice(prof["kinds"])
    pts = []
    if kind == "grid":
        nx, ny = rng.choice(((2, 2), (3, 2), (2, 3), (4, 2)))
        sx = sn(rng.uniform(*prof["pitch"]))
        sy = sn(rng.uniform(*prof["pitch"]))
        ox = sn(rng.uniform(mg, max(mg + 1, w - mg - sx * (nx - 1))))
        oy = sn(rng.uniform(mg, max(mg + 1, h - mg - sy * (ny - 1))))
        pts = [(ox + i * sx, oy + j * sy) for i in range(nx) for j in range(ny)]
    elif kind == "circle":
        n = rng.choice((3, 4, 5, 6, 8))
        r = rng.uniform(*prof["radius"])
        cx = sn(rng.uniform(mg + r, max(mg + 1 + r, w - mg - r)))
        cy = sn(rng.uniform(mg + r, max(mg + 1 + r, h - mg - r)))
        a0 = rng.uniform(0, 2 * math.pi)
        pts = [(sn(cx + r * math.cos(a0 + 2 * math.pi * k / n)),
                sn(cy + r * math.sin(a0 + 2 * math.pi * k / n))) for k in range(n)]
    elif kind == "line":
        n = rng.randint(2, 6)
        s = sn(rng.uniform(*prof["line_pitch"]))
        ox = sn(rng.uniform(mg, max(mg + 1, w - mg - s * (n - 1))))
        oy = sn(rng.uniform(mg, h - mg))
        pts = [(ox + i * s, oy) for i in range(n)]
    else:  # stagger: two rows, the second shifted by half a pitch
        n = rng.randint(2, 5)
        s = sn(rng.uniform(*prof["pitch"]))
        row = sn(rng.uniform(*prof["pitch"]))
        ox = sn(rng.uniform(mg, max(mg + 1, w - mg - s * n)))
        oy = sn(rng.uniform(mg, max(mg + 1, h - mg - row)))
        pts = [(ox + i * s, oy) for i in range(n)] + \
              [(sn(ox + s / 2.0 + i * s), oy + row) for i in range(n)]
    return kind, [(round(x, 6), round(y, 6)) for (x, y) in pts]


def make_task(seed, skill=None, profile="default"):
    """Build one task. Rejection-samples until the pattern is legal on both parts."""
    prof = PROFILES[profile]
    g = prof["grid"]

    def sn(v):
        return round(round(v / g) * g, 6)

    rng = random.Random(seed)
    skill = skill or rng.choice(SKILLS)
    while True:
        sw, sh = sn(rng.uniform(*prof["w"])), sn(rng.uniform(*prof["h"]))
        tw = round(sw + rng.choice((-1, 1)) * sn(rng.uniform(*prof["delta"])), 6)
        th = round(sh + rng.choice((-1, 1)) * sn(rng.uniform(*prof["delta"])), 6)
        anchor = rng.choice(ANCHORS)
        kind, pts = _pattern(rng, sw, sh, prof)
        thread = rng.choice(sorted(CLEARANCE))
        if skill == "clearance":
            src_d, dst_d = thread, CLEARANCE[thread]
        else:
            src_d = dst_d = CLEARANCE[thread]
        source = [(x, y, src_d) for (x, y) in pts]
        expected = [to_target((x, y, dst_d), anchor, (sw, sh), (tw, th)) for (x, y) in pts]
        if rule_violations(source, sw, sh) == 0 and rule_violations(expected, tw, th) == 0:
            break
    if skill == "clearance":
        text = ("Add clearance holes on the target plate for the M%g tapped holes of the source "
                "part, with identical spacing and identical offsets from the %s corner."
                % (thread, anchor))
    else:
        text = ("Add mounting holes on the target plate matching the bolt pattern of the source "
                "part, with identical diameters, spacing, and offsets from the %s corner." % anchor)
    return {
        "seed": seed, "skill": skill, "anchor": anchor, "pattern": kind,
        "instruction": text,
        "source": {"width": sw, "height": sh, "holes": source,
                   "tapped": skill == "clearance"},
        "target": {"width": tw, "height": th},
        "expected": expected,
    }


class BoltEnv:
    """reset(seed) and step(action), in the usual gym shape.

    Actions are tuples, the same edits a CAD agent would issue as tool calls:
        ("add", x, y, d)      ("move", i, dx, dy)     ("dia", i, d)
        ("del", i)            ("submit",)
    reward="dense":  change in verifier score on every edit, plus 1.0 for a correct submit.
    reward="sparse": 0.0 on every edit, 1.0 for a correct submit.
    An invalid action changes nothing and returns info["invalid"] = True.
    """

    def __init__(self, skill=None, reward="dense", max_steps=2000):
        if reward not in ("dense", "sparse"):
            raise ValueError("reward must be 'dense' or 'sparse'")
        self.skill, self.reward_mode, self.max_steps = skill, reward, max_steps
        self.task = None

    def reset(self, seed=0):
        self.task = make_task(seed, self.skill)
        self.holes = []
        self.steps = 0
        self.done = False
        self._score = self._check()["score"]
        return self.observation()

    def _check(self):
        t = self.task
        return verify(self.holes, t["expected"], t["target"]["width"], t["target"]["height"])

    def observation(self):
        t = self.task
        return {
            "instruction": t["instruction"], "skill": t["skill"], "anchor": t["anchor"],
            "source": t["source"], "target": t["target"],
            "holes": list(self.holes), "steps": self.steps,
        }

    def step(self, action):
        if self.task is None or self.done:
            raise RuntimeError("call reset() first")
        self.steps += 1
        info = {"invalid": False}
        kind = action[0]
        try:
            if kind == "add":
                self.holes.append((float(action[1]), float(action[2]), float(action[3])))
            elif kind == "move":
                x, y, d = self.holes[action[1]]
                self.holes[action[1]] = (x + float(action[2]), y + float(action[3]), d)
            elif kind == "dia":
                x, y, _ = self.holes[action[1]]
                if float(action[2]) <= 0:
                    raise ValueError
                self.holes[action[1]] = (x, y, float(action[2]))
            elif kind == "del":
                del self.holes[action[1]]
            elif kind != "submit":
                raise ValueError
        except (IndexError, ValueError, TypeError):
            info["invalid"] = True
        result = self._check()
        info.update(result)
        reward = 0.0
        if self.reward_mode == "dense":
            reward = result["score"] - self._score
        self._score = result["score"]
        if kind == "submit" and not info["invalid"]:
            self.done = True
            if result["success"]:
                reward += 1.0
        elif self.steps >= self.max_steps:
            self.done = True
            info["truncated"] = True
            info["success"] = False
        if not self.done:
            info["success"] = False
        return self.observation(), reward, self.done, info


def to_scad(width, height, holes, thickness=6.0):
    """OpenSCAD source for a plate with through holes."""
    lines = ["$fn = 48;", "difference() {",
             "  cube([%g, %g, %g]);" % (width, height, thickness)]
    for (x, y, d) in holes:
        lines.append("  translate([%g, %g, -1]) cylinder(d = %g, h = %g);"
                     % (x, y, d, thickness + 2))
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv):
    if len(argv) >= 2 and argv[0] == "task":
        print(json.dumps(make_task(int(argv[1]), argv[2] if len(argv) > 2 else None), indent=1))
    elif len(argv) >= 2 and argv[0] == "scad":
        t = make_task(int(argv[1]), argv[2] if len(argv) > 2 else None)
        sys.stdout.write(to_scad(t["target"]["width"], t["target"]["height"], t["expected"]))
    else:
        print("usage: python boltgym.py task <seed> [transfer|clearance]\n"
              "       python boltgym.py scad <seed> [transfer|clearance]")


if __name__ == "__main__":
    main(sys.argv[1:])
