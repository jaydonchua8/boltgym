"""A one-minute tour: python demo.py

Shows one task, the sliders the parameter reader finds in the target file, four
answers going through the verifier, and one repair loop in the environment.
Everything printed is computed here. The parts are synthetic.
"""
import params
import scadtask
from boltgym import make_task

SEED = 36


def rule(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    task = make_task(SEED)
    rule("1. Task %d (%s, %s pattern, %d holes)" % (SEED, task["skill"], task["pattern"],
                                                   len(task["expected"])))
    print(task["instruction"])
    src, tgt = task["source"], task["target"]
    print("source plate %s x %s mm, target plate %s x %s mm" % (
        scadtask.num(src["width"]), scadtask.num(src["height"]),
        scadtask.num(tgt["width"]), scadtask.num(tgt["height"])))

    rule("2. Source part as the agent sees it (first 14 lines of OpenSCAD)")
    print("\n".join(scadtask.source_scad(task).split("\n")[:14]))

    rule("3. Sliders read from the target file")
    for p in params.extract(scadtask.target_scad(task)):
        print("  %-16s = %-6s group=%-6s range=%s  (%s)" % (
            p["name"], scadtask.num(p["value"]), p["group"] or "-",
            "%g..%g step %g" % (p["range"]["min"], p["range"]["max"], p["range"]["step"])
            if p["range"] else "-", p["description"] or "no description"))

    rule("4. Four answers through the verifier")
    answers = [
        ("paste the source coordinates", scadtask.write_copy(task)),
        ("right numbers, hardcoded", scadtask.write_solution(task, "explicit", "literal_coords")),
        ("right, but cylinder(r = hole_dia)", scadtask.write_solution(task, "list",
                                                                      "radius_as_diameter")),
        ("parametric", scadtask.write_solution(task, "list")),
    ]
    for name, code in answers:
        r = scadtask.verify_scad(code, task)
        print("  %-34s default sizes: %-4s resized: %d/%d  score %.3f  stage %s" % (
            name, "pass" if r["success_default"] else "FAIL", r["sweeps_passed"], r["sweeps"],
            r["score"], r["stage"]))
        if r["error"]:
            print("      -> %s" % r["error"])

    rule("5. The hole lines of the hardcoded and the parametric answer")
    for name, code in (answers[1], answers[3]):
        lines = [ln for ln in code.split("\n") if "translate" in ln or "cylinder" in ln]
        print("  %s:" % name)
        for ln in lines[:2]:
            print("      " + ln.strip())

    rule("6. One episode with a repair")
    env = scadtask.ScadEnv(attempts=3)
    env.reset(SEED)
    good = scadtask.write_solution(env.task, "module")
    head, tail = good.rsplit("    hole(", 1)
    broken = head + "    drill(" + tail
    for label, code in (("first try calls an undefined module", broken),
                        ("second try after reading the feedback", good)):
        obs, reward, done, info = env.step(code)
        print("  %-40s reward %.3f  done %s  feedback: %s" % (label, reward, done,
                                                               obs["feedback"]))

    rule("7. Editing a parameter the way a slider would")
    code = answers[3][1]
    wider = params.set_params(code, {"plate_width": tgt["width"] + 25, "hole_offsets[0][1]": 20})
    changed = [(a, b) for a, b in zip(code.split("\n"), wider.split("\n")) if a != b]
    for a, b in changed:
        print("  - %s\n  + %s" % (a.strip(), b.strip()))
    print("  lines changed: %d of %d" % (len(changed), code.count("\n")))


if __name__ == "__main__":
    main()
