"""scadtask: the bolt pattern task with OpenSCAD source as the interface.

The agent gets the source part and the target plate as OpenSCAD files with
Customizer-style parameters, and answers with an edited target file. The verifier
runs the answer through the evaluator in scad.py, compares the holes with the
expected ones, then moves the plate's sliders (width, height, thickness) and
checks the holes again. An answer that is only right at the default sizes fails
the second check.

This file also holds reference writers and a set of deliberately defective
writers used to measure what each check catches. All parts are synthetic.
"""
import random

from boltgym import (CLEARANCE, PROFILES, RULE_PENALTY, make_task, rule_violations,
                     to_target, verify)
from params import extract, set_params
from scad import ScadError, read_plate, tokenize

THICKNESS = 6.0
REQUIRED = ("plate_width", "plate_height", "plate_thickness")
SWEEP_THICKNESS = (10.0, 4.0, 14.0, 8.0)
STYLES = ("list", "explicit", "module")

# Defects that are wrong at the default sizes, and defects that only show when a
# slider moves. They are plausible mistakes that I chose, not samples from a real model.
DEFAULT_DEFECTS = ("paste_source", "wrong_corner", "radius_as_diameter", "blind_hole",
                   "missing_hole", "duplicate_hole", "tapped_size", "syntax_error",
                   "undefined_module", "removed_parameter")
PARAMETRIC_DEFECTS = ("literal_coords", "one_axis_literal", "proportional", "literal_depth")
DEFECTS = DEFAULT_DEFECTS + PARAMETRIC_DEFECTS


def num(v):
    """Shortest text for a number that round-trips to within 1e-6."""
    s = ("%.6f" % v).rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _header(title, w, h):
    return ("// %s (synthetic)\n\n"
            "/* [Plate] */\n"
            "// Plate size along X\n"
            "plate_width = %s;       // [20:0.1:600]\n"
            "// Plate size along Y\n"
            "plate_height = %s;      // [20:0.1:600]\n"
            "// Plate thickness\n"
            "plate_thickness = %s;   // [2:0.5:20]\n\n" % (title, num(w), num(h), num(THICKNESS)))


def source_scad(task):
    """The source part as an OpenSCAD file. Grids and lines are written as loops
    over pitch and count, other patterns as a list of positions."""
    src = task["source"]
    holes = src["holes"]
    d = holes[0][2]
    out = [_header("Source part", src["width"], src["height"]), "/* [Holes] */\n"]
    if src["tapped"]:
        out.append("// Tap drill callout: M%s tapped holes, nominal thread diameter\n" % num(d))
    else:
        out.append("// Clearance hole diameter\n")
    out.append("hole_dia = %s;          // [1:0.1:20]\n" % num(d))
    xs = sorted(set(x for (x, _, _) in holes))
    ys = sorted(set(y for (_, y, _) in holes))
    if task["pattern"] in ("grid", "line") and len(xs) * len(ys) == len(holes):
        px = xs[1] - xs[0] if len(xs) > 1 else 0.0
        py = ys[1] - ys[0] if len(ys) > 1 else 0.0
        out.append("// Position of the first hole from the bottom-left corner\n"
                   "hole_origin = [%s, %s];\n"
                   "// Spacing between holes along X and Y\n"
                   "hole_pitch = [%s, %s];\n"
                   "// Number of holes along X\n"
                   "hole_count_x = %d;       // [1:1:8]\n"
                   "// Number of holes along Y\n"
                   "hole_count_y = %d;       // [1:1:8]\n\n$fn = 48;\n\n"
                   "module hole_pattern() {\n"
                   "    for (i = [0 : hole_count_x - 1])\n"
                   "        for (j = [0 : hole_count_y - 1])\n"
                   "            translate([hole_origin[0] + i * hole_pitch[0],\n"
                   "                       hole_origin[1] + j * hole_pitch[1], -1])\n"
                   "                cylinder(d = hole_dia, h = plate_thickness + 2);\n"
                   "}\n\n" % (num(xs[0]), num(ys[0]), num(px), num(py), len(xs), len(ys)))
    else:
        out.append("// Hole centres from the bottom-left corner, [x, y]\nhole_positions = [\n")
        out.append(",\n".join("    [%s, %s]" % (num(x), num(y)) for (x, y, _) in holes))
        out.append("\n];\n\n$fn = 48;\n\n"
                   "module hole_pattern() {\n"
                   "    for (p = hole_positions)\n"
                   "        translate([p.x, p.y, -1])\n"
                   "            cylinder(d = hole_dia, h = plate_thickness + 2);\n"
                   "}\n\n")
    out.append("difference() {\n    cube([plate_width, plate_height, plate_thickness]);\n"
               "    hole_pattern();\n}\n")
    return "".join(out)


def target_scad(task):
    """The target plate before the edit: a blank plate with its three sliders."""
    t = task["target"]
    return (_header("Target plate", t["width"], t["height"]) + "$fn = 48;\n\n"
            "difference() {\n    cube([plate_width, plate_height, plate_thickness]);\n}\n")


# ---- verifier ----------------------------------------------------------------

def expected_for(task, width, height):
    """Expected holes when the target plate is resized: offsets from the named
    corner stay what they are on the source part."""
    src = task["source"]
    d = task["expected"][0][2]
    return [to_target((x, y, d), task["anchor"], (src["width"], src["height"]), (width, height))
            for (x, y, _) in src["holes"]]


def sweep_sizes(task, count):
    """Deterministic resized plates for a task, each one legal for the expected holes."""
    rng = random.Random(task["seed"] * 7919 + 17)
    tw, th = task["target"]["width"], task["target"]["height"]
    out = []
    for k in range(count):
        for attempt in range(60):
            grow = attempt >= 40
            dw = round(rng.uniform(5 if grow else -30, 45) * 2) / 2.0
            dh = round(rng.uniform(5 if grow else -30, 45) * 2) / 2.0
            if abs(dw) < 5 or abs(dh) < 5:
                continue
            w, h = tw + dw, th + dh
            if rule_violations(expected_for(task, w, h), w, h) == 0:
                break
        else:
            w, h = tw + 20.0 + k, th + 20.0 + k
        out.append((w, h, SWEEP_THICKNESS[k % len(SWEEP_THICKNESS)]))
    return out


def _check(code, expected, width, height, thickness):
    """One evaluation of code against one plate size. Returns (ok, score, problem)."""
    plate = read_plate(code)
    if abs(plate["width"] - width) > 1e-6 or abs(plate["height"] - height) > 1e-6 \
            or abs(plate["thickness"] - thickness) > 1e-6:
        return False, 0.0, ("plate is %s x %s x %s but the parameters say %s x %s x %s"
                            % (num(plate["width"]), num(plate["height"]), num(plate["thickness"]),
                               num(width), num(height), num(thickness)))
    if max(abs(c) for c in plate["origin"]) > 1e-6:
        return False, 0.0, "plate was moved away from the origin"
    r = verify(plate["holes"], expected, width, height)
    score = r["score"] - RULE_PENALTY * 2 * len(plate["defects"])
    problem = None
    if plate["defects"]:
        problem = plate["defects"][0]
    elif not r["success"]:
        if len(plate["holes"]) != len(expected):
            problem = "found %d through holes, expected %d" % (len(plate["holes"]), len(expected))
        elif r["violations"]:
            problem = "%d edge or spacing rule violations" % r["violations"]
        elif r["worst_position_error"] > 0.05:
            problem = "worst hole is %.2f mm from where it should be" % r["worst_position_error"]
        else:
            problem = "a hole diameter is wrong"
    return bool(r["success"] and not plate["defects"]), score, problem


def oracle_sizes(task, count=12):
    """A larger, independently drawn set of plate sizes, used only to judge the
    verifier itself: wider size changes and any thickness from 2 to 20 mm."""
    rng = random.Random(task["seed"] * 104729 + 5)
    tw, th = task["target"]["width"], task["target"]["height"]
    out = []
    while len(out) < count:
        w = tw + round(rng.uniform(-40, 80) * 2) / 2.0
        h = th + round(rng.uniform(-40, 80) * 2) / 2.0
        if w == tw or h == th or (len(out) >= count // 2 and (w < tw or h < th)):
            continue
        if rule_violations(expected_for(task, w, h), w, h) == 0:
            out.append((w, h, round(rng.uniform(2, 20) * 2) / 2.0))
    return out


def verify_scad(code, task, sweeps=3, sizes=None):
    """Score an OpenSCAD answer for a task.

    Returns a dict:
      stage            "parse", "params", "default", "sweep" (where it failed) or "passed"
      success_default  right at the sizes in the file
      success          right at the default sizes and at every resized plate
      score            0.5 * default score + 0.5 * mean score over the resized plates
      error            first problem found, as text an agent could act on
      sweep_ok         one True or False per resized plate
    sizes replaces the built-in resized plates with a list of (width, height, thickness).
    """
    out = {"stage": "parse", "success": False, "success_default": False, "score": 0.0,
           "error": None, "sweeps_passed": 0, "sweeps": sweeps, "sweep_ok": []}
    if sizes is None:
        sizes = sweep_sizes(task, sweeps)
    sweeps = out["sweeps"] = len(sizes)
    tw, th = task["target"]["width"], task["target"]["height"]
    try:
        declared = dict((p["name"], p) for p in extract(code))
        missing = [n for n in REQUIRED if n not in declared or declared[n]["type"] != "number"]
        ok, base, problem = _check(code, task["expected"], tw, th, THICKNESS)
    except ScadError as exc:
        out["error"] = str(exc)
        return out
    if missing:
        out["stage"] = "params"
        out["error"] = "parameter %s must stay a plain number at the top of the file" % missing[0]
        return out
    out["success_default"] = ok
    if not ok:
        out["stage"] = "default"
        out["error"] = problem
    total, first_sweep_problem = 0.0, None
    for (w, h, t) in sizes:
        try:
            moved = set_params(code, {"plate_width": w, "plate_height": h,
                                      "plate_thickness": t})
            s_ok, s_score, s_problem = _check(moved, expected_for(task, w, h), w, h, t)
        except (ScadError, KeyError, ValueError) as exc:
            s_ok, s_score, s_problem = False, 0.0, str(exc)
        total += s_score
        out["sweep_ok"].append(s_ok)
        if s_ok:
            out["sweeps_passed"] += 1
        elif first_sweep_problem is None:
            first_sweep_problem = "with the plate set to %s x %s x %s: %s" % (
                num(w), num(h), num(t), s_problem)
    out["score"] = 0.5 * base + 0.5 * (total / sweeps if sweeps else base)
    if ok and out["sweeps_passed"] == sweeps:
        out["stage"], out["success"] = "passed", True
    elif ok:
        out["stage"], out["error"] = "sweep", first_sweep_problem
    return out


class ScadEnv:
    """reset(seed) then step(code). The action is a whole OpenSCAD file.

    Reward is the gain in verifier score over the best earlier attempt, plus 1.0
    when the answer passes. The episode ends on a pass or after `attempts` tries.
    info["error"] carries the verifier's message so an agent can repair its code.
    """

    def __init__(self, skill=None, profile="default", attempts=3, sweeps=3):
        if profile not in PROFILES:
            raise ValueError("unknown profile %r" % profile)
        self.skill, self.profile, self.attempts, self.sweeps = skill, profile, attempts, sweeps
        self.task = None

    def reset(self, seed=0):
        self.task = make_task(seed, self.skill, self.profile)
        self.used, self.best, self.done, self.feedback = 0, 0.0, False, None
        return self.observation()

    def observation(self):
        t = self.task
        return {"instruction": t["instruction"], "source_scad": source_scad(t),
                "target_scad": target_scad(t), "attempts_left": self.attempts - self.used,
                "feedback": self.feedback}

    def step(self, code):
        if self.task is None or self.done:
            raise RuntimeError("call reset() first")
        self.used += 1
        info = verify_scad(code, self.task, self.sweeps) if isinstance(code, str) else {
            "stage": "parse", "success": False, "success_default": False, "score": 0.0,
            "error": "the action must be OpenSCAD source text", "sweeps_passed": 0,
            "sweeps": self.sweeps, "sweep_ok": []}
        reward = max(0.0, info["score"] - self.best)
        self.best = max(self.best, info["score"])
        if info["success"]:
            reward += 1.0
        self.feedback = info["error"]
        self.done = info["success"] or self.used >= self.attempts
        return self.observation(), reward, self.done, info


# ---- writers -----------------------------------------------------------------

def _coord(mode, offset, source_value, target_value, size_name, size_value, far):
    """Text of one hole coordinate. far is True when the anchor is the right or top edge."""
    if mode == "param":
        return "%s - %s" % (size_name, num(offset)) if far else num(offset)
    if mode == "literal":
        return num(target_value)
    if mode == "source":
        return num(source_value)
    if mode == "prop":
        return "%s * %.9f" % (size_name, target_value / size_value)
    if mode == "wrong":  # anchored to the opposite edge, using the source offset from it
        return num(source_value) if far else "%s - %s" % (size_name, num(offset))
    raise ValueError(mode)


def write_solution(task, style="list", defect=None):
    """Write an answer file for a task. With defect=None the answer is correct and
    fully parametric. With a defect name it contains exactly that mistake, or the
    function returns None when the mistake cannot be made on this task (for example
    pasting source coordinates is not wrong when the anchor is bottom-left)."""
    if style not in STYLES:
        raise ValueError("unknown style %r" % style)
    if defect is not None and defect not in DEFECTS:
        raise ValueError("unknown defect %r" % defect)
    src, tgt, anchor = task["source"], task["target"], task["anchor"]
    sw, sh, tw, th = src["width"], src["height"], tgt["width"], tgt["height"]
    far_x, far_y = "right" in anchor, "top" in anchor
    xmode = ymode = "param"
    if defect == "paste_source":
        if anchor == "bottom-left":
            return None
        xmode = ymode = "source"
    elif defect == "wrong_corner":
        xmode = "wrong"
    elif defect == "literal_coords":
        if anchor == "bottom-left":
            return None
        xmode = ymode = "literal"
    elif defect == "one_axis_literal":
        if far_y:
            ymode = "literal"
        elif far_x:
            xmode = "literal"
        else:
            return None
    elif defect == "proportional":
        xmode = ymode = "prop"
    if defect == "tapped_size" and not src["tapped"]:
        return None
    dia = src["holes"][0][2] if defect == "tapped_size" else task["expected"][0][2]

    rows = []
    for (x, y, _), (ex, ey, _) in zip(src["holes"], task["expected"]):
        ox = sw - x if far_x else x
        oy = sh - y if far_y else y
        if xmode == "wrong":
            ox = x if far_x else sw - x
        rows.append((_coord(xmode, ox, x, ex, "plate_width", tw, far_x),
                     _coord(ymode, oy, y, ey, "plate_height", th, far_y)))
    if defect == "missing_hole":
        rows = rows[:-1]
    elif defect == "duplicate_hole":
        rows.append(rows[0])

    size = "r = hole_dia" if defect == "radius_as_diameter" else "d = hole_dia"
    if defect == "blind_hole":
        z, depth = "plate_thickness / 2", "h = plate_thickness"
    elif defect == "literal_depth":
        z, depth = "-1", "h = %s" % num(THICKNESS + 2)
    else:
        z, depth = "-1", "h = plate_thickness + 2"
    cut = "cylinder(%s, %s);" % (size, depth)
    call = "drill" if defect == "undefined_module" else "hole"

    out = [_header("Target plate", tw, th), "/* [Holes] */\n",
           "// Hole diameter\nhole_dia = %s;          // [1:0.1:20]\n" % num(dia)]
    plain = xmode == ymode == "param"
    if style == "list" and plain:
        out.append("// Hole offsets from the %s corner, [x, y]\nhole_offsets = [\n" % anchor)
        out.append(",\n".join("    [%s, %s]" % (num(sw - x if far_x else x), num(sh - y if far_y else y))
                              for (x, y, _) in src["holes"][:len(rows)]))
        if defect == "duplicate_hole":
            x, y, _ = src["holes"][0]
            out.append(",\n    [%s, %s]" % (num(sw - x if far_x else x), num(sh - y if far_y else y)))
        out.append("\n];\n\n$fn = 48;\n\ndifference() {\n"
                   "    cube([plate_width, plate_height, plate_thickness]);\n"
                   "    for (o = hole_offsets)\n"
                   "        translate([%s, %s, %s])\n            %s\n"
                   % ("plate_width - o[0]" if far_x else "o[0]",
                      "plate_height - o[1]" if far_y else "o[1]", z,
                      cut if call == "hole" else "drill(hole_dia);"))
    elif style == "module":
        out.append("\n$fn = 48;\n\nmodule hole(x, y) {\n    translate([x, y, %s])\n        %s\n}\n\n"
                   "difference() {\n    cube([plate_width, plate_height, plate_thickness]);\n"
                   % (z, cut))
        out.extend("    %s(%s, %s);\n" % (call, a, b) for (a, b) in rows)
    else:
        out.append("\n$fn = 48;\n\ndifference() {\n"
                   "    cube([plate_width, plate_height, plate_thickness]);\n")
        for (a, b) in rows:
            out.append("    translate([%s, %s, %s]) %s\n"
                       % (a, b, z, cut if call == "hole" else "drill(hole_dia);"))
    if defect != "syntax_error":
        out.append("}\n")
    code = "".join(out)
    if defect == "removed_parameter":
        code = code.replace("plate_width = ", "w = ", 1).replace("plate_width", "w")
    return code


def write_copy(task):
    """What a careless edit does: paste the source coordinates and diameter unchanged."""
    holes = task["source"]["holes"]
    out = [_header("Target plate", task["target"]["width"], task["target"]["height"]),
           "$fn = 48;\n\ndifference() {\n    cube([plate_width, plate_height, plate_thickness]);\n"]
    for (x, y, d) in holes:
        out.append("    translate([%s, %s, -1]) cylinder(d = %s, h = plate_thickness + 2);\n"
                   % (num(x), num(y), num(d)))
    return "".join(out) + "}\n"


# ---- random single-token mutants ----------------------------------------------

def mutate(code, rng):
    """Change one token of a correct answer at random. Returns (new code, kind).

    Kinds: "number" (a literal is nudged), "operator" (+ and - or * and / swapped),
    "hardcode" (a plate parameter in an expression becomes its current number),
    "swap" (one plate parameter replaced by another), "radius" (d= becomes r=).
    Many mutants are harmless, for example a change to $fn. Whether a mutant is
    really wrong is decided by evaluating it, not by its kind.
    """
    toks = tokenize(code)
    values = dict((p["name"], p["value"]) for p in extract(code))
    sites = []
    for i, t in enumerate(toks):
        declared = i + 1 < len(toks) and toks[i + 1].text == "=" and toks[i + 1].kind == "op"
        if t.kind == "num":
            sites.append((i, "number"))
        elif t.kind == "op" and t.text in "+-*/" and i > 0 and (
                toks[i - 1].kind in ("num", "id") or toks[i - 1].text in ")]"):
            sites.append((i, "operator"))
        elif t.kind == "id" and t.text in REQUIRED and not declared:
            sites.append((i, "hardcode"))
            sites.append((i, "swap"))
        elif t.kind == "id" and t.text == "d" and declared:
            sites.append((i, "radius"))
    i, kind = rng.choice(sites)
    t = toks[i]
    if kind == "number":
        v = float(t.text)
        new = num(rng.choice((v + 0.5, v + 1.0, max(0.0, v - 1.0), v * 2.0, v + 3.0)))
        if new == t.text:
            new = num(v + 2.0)
    elif kind == "operator":
        new = {"+": "-", "-": "+", "*": "/", "/": "*"}[t.text]
    elif kind == "hardcode":
        new = num(values[t.text])
    elif kind == "swap":
        new = rng.choice([n for n in REQUIRED if n != t.text])
    else:
        new = "r"
    return code[:t.start] + new + code[t.end:], kind
