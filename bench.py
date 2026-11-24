"""Prints the tables quoted in the README and writes bench_results.json.

    python bench.py          everything (about 7 minutes)
    python bench.py scad     only the OpenSCAD verifier measurements (2 to 3 minutes)
    python bench.py gym      only the tuple-action gym baselines (about 4 minutes)

Set CADAM_DIR to a local clone of github.com/Adam-CAD/CADAM to also run the parser
and the parameter reader on CADAM's 13 published benchmark models.
"""
import collections
import glob
import json
import os
import random
import re
import statistics
import sys
import time

import agents
import params
import scad
import scadtask
from boltgym import BoltEnv, assign, assign_greedy, make_task, verify

N = 500           # evaluation tasks, seeds 0..N-1
TRAIN = 2000      # bandit training episodes, seeds 100000..


def evaluate(fn, env, seeds):
    wins, steps, by_skill = 0, [], {}
    for s in seeds:
        ok, n = fn(env, s)
        wins += ok
        steps.append(n)
        k = by_skill.setdefault(env.task["skill"], [0, 0])
        k[0] += ok
        k[1] += 1
    return {"success": wins / len(seeds), "median_steps": statistics.median(steps),
            "max_steps": max(steps),
            "by_skill": {k: v[0] / v[1] for k, v in sorted(by_skill.items())}}


def matcher_gap(trials=5000):
    """How often does greedy matching score a state lower than optimal matching?"""
    rng = random.Random(0)
    worse, gaps, flips = 0, [], 0
    for i in range(trials):
        t = make_task(i)
        w, h = t["target"]["width"], t["target"]["height"]
        sigma = rng.choice((0.0, 2.0, 8.0, 20.0))
        placed = [(x + rng.gauss(0, sigma), y + rng.gauss(0, sigma), d) for (x, y, d) in t["expected"]]
        rng.shuffle(placed)
        a = verify(placed, t["expected"], w, h, matcher=assign)
        g = verify(placed, t["expected"], w, h, matcher=assign_greedy)
        if g["score"] < a["score"] - 1e-9:
            worse += 1
            gaps.append(a["score"] - g["score"])
        if a["success"] != g["success"]:
            flips += 1
    return {"trials": trials, "greedy_lower": worse / trials,
            "mean_gap_when_lower": statistics.mean(gaps) if gaps else 0.0,
            "max_gap": max(gaps) if gaps else 0.0, "success_flips": flips}


def verify_speed(calls=20000):
    tasks = [make_task(i) for i in range(200)]
    t0 = time.perf_counter()
    for i in range(calls):
        t = tasks[i % 200]
        verify(t["source"]["holes"], t["expected"], t["target"]["width"], t["target"]["height"])
    return (time.perf_counter() - t0) / calls * 1e6


def gym_bench():
    seeds = range(N)
    out = {"tasks": N}
    rows = [
        ("random", lambda e, s: agents.run_random(e, s), "dense"),
        ("copy source coordinates", agents.run_copy, "dense"),
        ("reward-only search, sparse reward", agents.run_search, "sparse"),
        ("reward-only search, dense reward", agents.run_search, "dense"),
        ("scripted geometry", agents.run_scripted, "dense"),
    ]
    print("%d tasks, seeds 0..%d, both skills mixed\n" % (N, N - 1))
    print("| agent | success | transfer | clearance | median steps | max steps |")
    print("|---|---|---|---|---|---|")
    out["agents"] = {}
    for name, fn, mode in rows:
        r = evaluate(fn, BoltEnv(reward=mode, max_steps=2000), seeds)
        out["agents"][name] = r
        print("| %s | %.1f%% | %.1f%% | %.1f%% | %g | %d |" % (
            name, 100 * r["success"], 100 * r["by_skill"]["transfer"],
            100 * r["by_skill"]["clearance"], r["median_steps"], r["max_steps"]))

    env = BoltEnv(reward="sparse")
    bandit, curve = agents.train_bandit(env, TRAIN)
    held = evaluate(lambda e, s: (agents.run_macro(e, s, agents.MACROS[
        bandit.choose(agents.peek_context(e, s), explore=False)])[0], e.steps), env, seeds)
    first95 = next((i + 1 for i, v in enumerate(curve) if v >= 0.95), None)
    out["bandit"] = {"train_episodes": TRAIN, "curve_per_100": curve,
                     "first_block_at_95": first95, "held_out_success": held["success"],
                     "contexts": 8, "actions": len(agents.MACROS)}
    print("\nbandit over 8 macro programs, trained on %d episodes with 0/1 reward" % TRAIN)
    print("success per 100 training episodes (10 percent exploration on):")
    print("  " + " ".join("%.2f" % v for v in curve))
    print("first block at or above 0.95: %s" % first95)
    print("held-out success on seeds 0..%d with exploration off: %.1f%%" % (N - 1, 100 * held["success"]))

    g = matcher_gap()
    out["matcher"] = g
    print("\ngreedy vs optimal matching inside the verifier, %d perturbed states" % g["trials"])
    print("greedy gave a lower score in %.1f%% of states, mean gap %.3f, max gap %.3f, "
          "success verdict differed in %d" % (100 * g["greedy_lower"], g["mean_gap_when_lower"],
                                              g["max_gap"], g["success_flips"]))

    us = verify_speed()
    out["verify_microseconds"] = round(us, 1)
    print("\nverify(): %.0f microseconds per call on this machine" % us)

    return out


# ---- OpenSCAD verifier measurements -------------------------------------------

SCAD_TASKS = 60       # tasks per split for the writer and defect tables
MUTANTS = 400         # random single-token mutants per split
SPLITS = (("dev", "default", 0), ("held-out", "heldout", 900000))


def writers_table(profile, first):
    rows = collections.OrderedDict()
    for name in ("paste source coordinates", "correct numbers, hardcoded", "parametric"):
        rows[name] = [0, 0]
    for seed in range(first, first + SCAD_TASKS):
        task = make_task(seed, profile=profile)
        answers = {
            "paste source coordinates": scadtask.write_copy(task),
            "correct numbers, hardcoded": scadtask.write_solution(task, "explicit", "literal_coords")
            or scadtask.write_solution(task, "explicit"),
            "parametric": scadtask.write_solution(task, scadtask.STYLES[seed % 3]),
        }
        for name, code in answers.items():
            r = scadtask.verify_scad(code, task)
            rows[name][0] += r["success_default"]
            rows[name][1] += r["success"]
    return dict((k, {"default_check": v[0] / SCAD_TASKS, "with_slider_sweep": v[1] / SCAD_TASKS})
                for k, v in rows.items())


def defects_table(profile, first):
    table = collections.OrderedDict((d, {"n": 0, "default_check": 0, "with_slider_sweep": 0,
                                         "stages": {}}) for d in scadtask.DEFECTS)
    false_rejects = 0
    for seed in range(first, first + SCAD_TASKS):
        task = make_task(seed, profile=profile)
        for style in scadtask.STYLES:
            if not scadtask.verify_scad(scadtask.write_solution(task, style), task)["success"]:
                false_rejects += 1
            for d in scadtask.DEFECTS:
                code = scadtask.write_solution(task, style, d)
                if code is None:
                    continue
                r = scadtask.verify_scad(code, task)
                row = table[d]
                row["n"] += 1
                row["default_check"] += not r["success_default"]
                row["with_slider_sweep"] += not r["success"]
                row["stages"][r["stage"]] = row["stages"].get(r["stage"], 0) + 1
    return {"correct_answers": SCAD_TASKS * len(scadtask.STYLES), "false_rejects": false_rejects,
            "defects": table}


def mutants_table(profile, first, seed):
    """Random one-token mutants of correct answers. A 12-size oracle decides which are
    really wrong. Counts how many of those the cheaper checks catch."""
    rng = random.Random(seed)
    c = collections.Counter()
    for k in range(MUTANTS):
        task = make_task(first + k, profile=profile)
        code, kind = scadtask.mutate(scadtask.write_solution(task, scadtask.STYLES[k % 3]), rng)
        oracle = scadtask.verify_scad(code, task, sizes=scadtask.oracle_sizes(task))
        mine = scadtask.verify_scad(code, task, sweeps=3)
        flat = scadtask.verify_scad(code, task, sizes=[
            (w, h, scadtask.THICKNESS) for (w, h, _) in scadtask.sweep_sizes(task, 3)])
        c["mutants"] += 1
        if oracle["success"]:
            c["harmless"] += 1
            c["harmless_rejected"] += not mine["success"]
            continue
        c["wrong"] += 1
        if not oracle["success_default"]:
            c["wrong_at_default"] += 1
            continue
        c["wrong_only_when_resized"] += 1
        for n in (1, 2, 3):
            c["caught_by_%d_sizes" % n] += not all(mine["sweep_ok"][:n])
        c["caught_without_thickness"] += not flat["success"]
    return dict(c)


def cadam_files(root):
    table = open(os.path.join(root, "benchmarks", "README.md"), encoding="utf-8").read()
    want = dict((m.group(1), (int(m.group(2)), int(m.group(3)))) for m in re.finditer(
        r"\]\((\d\d-[a-z0-9-]+)\.md\).*?\|\s*(\d+) dims? · (\d+) colors?", table))
    files = sorted(glob.glob(os.path.join(root, "benchmarks", "*.scad")))
    out = {"files": len(files), "parsed": 0, "evaluated": 0, "control_counts_match": 0,
           "parameters": 0, "lines": 0}
    for path in files:
        text = open(path, encoding="utf-8").read()
        out["lines"] += text.count("\n")
        try:
            scad.parse(text)
            out["parsed"] += 1
            scad.evaluate(text)
            out["evaluated"] += 1
        except scad.ScadError:
            pass
        s = params.summary(text)
        out["parameters"] += s["dims"] + s["colors"] + s["other"]
        if (s["dims"] + s["other"], s["colors"]) == want.get(os.path.basename(path)[:-5]):
            out["control_counts_match"] += 1
    return out


def scad_bench():
    out = {"tasks_per_split": SCAD_TASKS, "mutants_per_split": MUTANTS, "splits": {}}
    for label, profile, first in SPLITS:
        w = writers_table(profile, first)
        d = defects_table(profile, first)
        m = mutants_table(profile, first + 5000, 11 if label == "dev" else 23)
        out["splits"][label] = {"profile": profile, "first_seed": first, "writers": w,
                                "defect_table": d, "mutants": m}
        print("\n== %s split: profile %s, seeds %d..%d ==" % (label, profile, first,
                                                            first + SCAD_TASKS - 1))
        print("\n| answer writer | passes default-size check | passes with slider sweep |")
        print("|---|---|---|")
        for name, r in w.items():
            print("| %s | %.1f%% | %.1f%% |" % (name, 100 * r["default_check"],
                                                100 * r["with_slider_sweep"]))
        print("\ncorrect answers rejected: %d of %d" % (d["false_rejects"], d["correct_answers"]))
        print("\n| injected defect | answers | caught by default-size check | caught with slider "
              "sweep | stage |")
        print("|---|---|---|---|---|")
        for name, r in d["defects"].items():
            print("| %s | %d | %d | %d | %s |" % (name, r["n"], r["default_check"],
                                                  r["with_slider_sweep"],
                                                  ", ".join(sorted(r["stages"]))))
        print("\nrandom one-token mutants: %d, harmless per 12-size oracle: %d (of those "
              "rejected: %d), wrong: %d" % (m["mutants"], m.get("harmless", 0),
                                            m.get("harmless_rejected", 0), m.get("wrong", 0)))
        only = m.get("wrong_only_when_resized", 0)
        print("wrong at default sizes: %d, wrong only when resized: %d"
              % (m.get("wrong_at_default", 0), only))
        print("of those %d: caught by 1 size %d, 2 sizes %d, 3 sizes %d, 3 sizes with thickness "
              "held fixed %d" % (only, m.get("caught_by_1_sizes", 0), m.get("caught_by_2_sizes", 0),
                                 m.get("caught_by_3_sizes", 0),
                                 m.get("caught_without_thickness", 0)))
    task = make_task(3)
    code = scadtask.write_solution(task, "list")
    t0 = time.perf_counter()
    for _ in range(100):
        scadtask.verify_scad(code, task)
    out["verify_scad_ms"] = round((time.perf_counter() - t0) / 100 * 1000, 2)
    print("\nverify_scad(): %.1f ms per call (1 default evaluation and 3 resized ones)"
          % out["verify_scad_ms"])
    root = os.environ.get("CADAM_DIR")
    if root:
        out["cadam_benchmarks"] = c = cadam_files(root)
        print("\nCADAM benchmark models read from CADAM_DIR: %d files, %d lines" %
              (c["files"], c["lines"]))
        print("parsed %d of %d, evaluated by the plate subset %d of %d, control counts equal to "
              "the published table %d of %d (%d parameters)" % (
                  c["parsed"], c["files"], c["evaluated"], c["files"],
                  c["control_counts_match"], c["files"], c["parameters"]))
    else:
        print("\nCADAM_DIR is not set, skipping the check against CADAM's benchmark models")
    return out


def main(argv):
    which = argv[0] if argv else "all"
    if which not in ("all", "scad", "gym"):
        print(__doc__)
        return 2
    try:
        results = json.load(open("bench_results.json"))
    except (IOError, ValueError):
        results = {}
    if which in ("all", "scad"):
        results["scad"] = scad_bench()
    if which in ("all", "gym"):
        print()
        results.update(gym_bench())
    json.dump(results, open("bench_results.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
