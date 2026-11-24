import random
import time
import unittest

import scadtask
from boltgym import make_task
from scad import ScadError, evaluate, parse, read_plate, tokenize

PLATE = """
plate_width = 100; plate_height = 80; t = 6;
pts = [[10, 10], [30, 10]];
function twice(x) = x * 2;
module hole(p, d = 4.5) { translate([p.x, p[1], -1]) cylinder(d = d, h = t + 2, $fn = 32); }
difference() {
    cube([plate_width, plate_height, t]);
    for (p = pts) hole(p);
    for (i = [0 : 2]) translate([plate_width - 10 - i * twice(5), 60, -1]) cylinder(r = 2, h = t + 2);
}
"""


def holes_of(src):
    return sorted((round(x, 6), round(y, 6), round(d, 6)) for (x, y, d) in read_plate(src)["holes"])


class TestTokenizer(unittest.TestCase):
    def test_comments_and_lines(self):
        toks = tokenize("a = 1; // note\n/* two\nlines */ b = .5e1;", comments=True)
        self.assertEqual([t.kind for t in toks], ["id", "op", "num", "op", "lc", "bc", "id", "op",
                                                  "num", "op"])
        self.assertEqual(toks[6].line, 3)
        self.assertEqual(len(tokenize("a = 1; // note\n/* x */")), 4)

    def test_string_with_comment_markers(self):
        toks = tokenize('s = "a // b /* c"; x = 1;')
        self.assertEqual(toks[2].text, '"a // b /* c"')
        self.assertEqual(len(toks), 8)

    def test_unclosed(self):
        for bad in ("/* never", 'x = "abc;', "x = 1 @ 2;"):
            self.assertRaises(ScadError, tokenize, bad)


class TestParser(unittest.TestCase):
    def test_language_features_parse(self):
        parse("""
        include <BOSL2/std.scad>
        use <MCAD/gears.scad>
        function f(x, y = 2) = x > 0 ? x * f(x - 1) : y;
        v = [for (i = [0 : 2 : 10]) if (i % 4 == 0) i else -i, each [1, 2]];
        w = let (a = 1, b = a + 1) [a, b, a ^ -b];
        g = function (x) x + 1;
        module m(n = 3) { if (n > 0) { m(n - 1); } else cube(1); }
        % translate([1, 2, 3]) # rotate([0, 0, 30]) ! m();
        * cube(5);
        intersection_for (i = [0 : 2]) rotate(i * 10) cube(1);
        echo(str("a", 1));
        """)

    def test_errors_have_lines(self):
        for bad, line in (("a = ;", 1), ("a = 1;\nb = (2;", 2), ("cube(1)\n", 1),
                          ("difference() {\n cube(1);\n", 2), ("x = [1, 2;", 1)):
            with self.assertRaises(ScadError) as ctx:
                parse(bad)
            self.assertEqual(ctx.exception.line, line, bad)

    def test_deep_nesting_is_an_error_not_a_crash(self):
        self.assertRaises(ScadError, parse, "x = " + "(" * 20000 + "1" + ")" * 20000 + ";")


class TestEvaluator(unittest.TestCase):
    def test_plate(self):
        p = read_plate(PLATE)
        self.assertEqual((p["width"], p["height"], p["thickness"]), (100.0, 80.0, 6.0))
        self.assertEqual(holes_of(PLATE), [(10, 10, 4.5), (30, 10, 4.5), (70, 60, 4), (80, 60, 4),
                                            (90, 60, 4)])
        self.assertEqual(p["defects"], [])

    def test_expressions(self):
        def val(text):
            return read_plate("x = %s; difference() { cube([50, 50, 5]); translate([x, 20, -1]) "
                              "cylinder(d = 3, h = 7); }" % text)["holes"][0][0]
        self.assertAlmostEqual(val("2 + 3 * 4"), 14)
        self.assertAlmostEqual(val("-2 ^ 2 + 20"), 16)
        self.assertAlmostEqual(val("10 * sin(30) * 2"), 10)
        self.assertAlmostEqual(val("len([for (i = [0 : 4]) if (i != 2) i]) * 5"), 20)
        self.assertAlmostEqual(val("[1, 2, 3] * [4, 5, 6]"), 32)
        self.assertAlmostEqual(val("let (a = 4) a > 3 && !false ? max(a, 7, 2) : 0"), 7)
        self.assertAlmostEqual(val("([10, 20] + [1, 1]).y"), 21)
        self.assertAlmostEqual(val("round(2.5) + floor(-0.5) + 7 % 4"), 5)

    def test_rotation_about_z(self):
        src = ("difference() { cube([60, 60, 4]); rotate([0, 0, 90]) translate([20, -10, -1]) "
               "cylinder(d = 3, h = 6); }")
        self.assertEqual(holes_of(src), [(10, 20, 3)])
        centered = "difference() { translate([30, 20, 2]) cube([60, 40, 4], center = true); " \
                   "translate([5, 5, 0]) cylinder(d = 3, h = 4); }"
        p = read_plate(centered)
        self.assertEqual((p["width"], p["height"], p["flush_cuts"]), (60.0, 40.0, 1))

    def test_defects(self):
        base = "difference() { cube([60, 60, 6]); %s }"
        blind = read_plate(base % "translate([20, 20, 3]) cylinder(d = 3, h = 10);")
        self.assertEqual(blind["holes"], [])
        self.assertIn("blind hole", blind["defects"][0])
        miss = read_plate(base % "translate([20, 20, 9]) cylinder(d = 3, h = 10);")
        self.assertIn("misses the plate", miss["defects"][0])
        taper = read_plate(base % "translate([20, 20, -1]) cylinder(d1 = 3, d2 = 6, h = 8);")
        self.assertIn("tapered", taper["defects"][0])
        slot = read_plate(base % "translate([20, 20, -1]) cube([5, 5, 8]);")
        self.assertIn("box cut", slot["defects"][0])

    def test_unsupported_is_reported_with_a_line(self):
        cases = {
            "cube(5);\nsphere(3);": "sphere",
            "include <BOSL2/std.scad>\ncube(5);": "BOSL2",
            "rotate([90, 0, 0]) cube(5);": "rotate about x or y",
            "rotate(30) cube(5);": "rotated by 30",
            "module m() { children(); }\nm() cube(1);": "children",
            "cube([10, 10, 2]); cylinder(d = 3, h = 9);": "solid cylinder",
            "cube(5); cube(6);": "exactly one solid plate",
            "difference() { cube(9); difference() { cube(3); cube(1); } }": "subtracted branch",
            "cube([10, undef_name, 2]);": "cube size",
            "difference() { cube(9); cylinder(5, 2); }": "by name",
            "difference() { cube(9); translate([1, 1]) cylinder(d = 0, h = 2); }": "no volume",
            "x = 1; assert(x > 2, \"too small\"); cube(5);": "assert failed",
            "cube(5); y = nosuch(3);": "nosuch",
        }
        for src, needle in cases.items():
            with self.assertRaises(ScadError) as ctx:
                read_plate(src)
            self.assertIn(needle, str(ctx.exception), src)

    def test_runaway_code_stops(self):
        bombs = [
            "function f(x) = f(x + 1); a = f(0); cube(1);",
            "module m() { m(); } m();",
            "for (i = [0 : 1e12]) cube(1);",
            "a = [for (i = [0 : 1e5]) [for (j = [0 : 1e5]) j]]; cube(1);",
            "function g(v, n) = n == 0 ? v : g(concat(v, v), n - 1); a = g([1], 40); cube(1);",
            "function s(v, n) = n == 0 ? v : s(str(v, v), n - 1); a = s(\"ab\", 40); cube(1);",
            "a = 10 ^ 400 * 0; b = 1 / 0; c = 5 % 0; cube(a);",
            "for (i = [0 : 0 : 5]) cube(1); cube([1/0, 1, 1]);",
        ]
        for src in bombs:
            t0 = time.time()
            self.assertRaises(ScadError, read_plate, src)
            self.assertLess(time.time() - t0, 5.0, src)

    def test_fuzz_only_raises_scad_error(self):
        rng = random.Random(7)
        soup = ["cube", "cylinder", "translate", "difference", "for", "if", "else", "module",
                "function", "let", "each", "x", "plate_width", "(", ")", "[", "]", "{", "}", ";",
                ",", "=", "+", "-", "*", "/", "%", "^", "?", ":", "<", "==", "!", "&&", ".", "1",
                "2.5", "1e3", '"s"', "true", "undef", "d", "h", "r", "center", "//c\n", "/*c*/"]
        good = scadtask.write_solution(make_task(1), "module")
        ok = 0
        for n in range(3000):
            if n % 2:
                src = " ".join(rng.choice(soup) for _ in range(rng.randint(1, 40)))
            else:
                a, b = sorted(rng.randrange(len(good)) for _ in range(2))
                src = good[:a] + rng.choice(soup) + good[b if n % 4 else a:]
            try:
                read_plate(src, max_ops=20000)
                ok += 1
            except ScadError:
                pass
        self.assertGreater(ok, 0)


class TestScadTask(unittest.TestCase):
    def test_source_file_matches_task_geometry(self):
        for profile, first in (("default", 0), ("heldout", 900000)):
            for seed in range(first, first + 80):
                task = make_task(seed, profile=profile)
                got = holes_of(scadtask.source_scad(task))
                want = sorted((round(x, 6), round(y, 6), round(d, 6))
                              for (x, y, d) in task["source"]["holes"])
                self.assertEqual(got, want, seed)
                blank = read_plate(scadtask.target_scad(task))
                self.assertEqual(blank["holes"], [])
                self.assertAlmostEqual(blank["width"], task["target"]["width"])

    def test_correct_answers_pass_in_every_style(self):
        for profile, first in (("default", 0), ("heldout", 900000)):
            for seed in range(first, first + 25):
                task = make_task(seed, profile=profile)
                for style in scadtask.STYLES:
                    r = scadtask.verify_scad(scadtask.write_solution(task, style), task)
                    self.assertTrue(r["success"], (seed, style, r))
                    self.assertAlmostEqual(r["score"], 1.0)
                    self.assertEqual(r["stage"], "passed")

    def test_each_defect_is_caught_at_the_expected_stage(self):
        seen = set()
        for seed in range(12):
            task = make_task(seed)
            for style in scadtask.STYLES:
                for defect in scadtask.DEFECTS:
                    code = scadtask.write_solution(task, style, defect)
                    if code is None:
                        continue
                    seen.add(defect)
                    r = scadtask.verify_scad(code, task)
                    self.assertFalse(r["success"], (seed, style, defect))
                    self.assertTrue(r["error"], (seed, style, defect))
                    self.assertLess(r["score"], 1.0)
                    if defect in scadtask.PARAMETRIC_DEFECTS:
                        self.assertTrue(r["success_default"], (seed, style, defect))
                        self.assertEqual(r["stage"], "sweep")
                    else:
                        self.assertFalse(r["success_default"], (seed, style, defect))
                    if defect in ("syntax_error", "undefined_module"):
                        self.assertEqual(r["stage"], "parse")
                    if defect == "removed_parameter":
                        self.assertEqual(r["stage"], "params")
        self.assertEqual(seen, set(scadtask.DEFECTS))

    def test_without_sweep_parametric_defects_pass(self):
        task = make_task(2)
        for defect in scadtask.PARAMETRIC_DEFECTS:
            code = scadtask.write_solution(task, "explicit", defect)
            if code is not None:
                self.assertTrue(scadtask.verify_scad(code, task, sweeps=0)["success"], defect)

    def test_copy_writer(self):
        for seed in range(40):
            task = make_task(seed)
            r = scadtask.verify_scad(scadtask.write_copy(task), task)
            self.assertEqual(r["success_default"],
                             task["anchor"] == "bottom-left" and task["skill"] == "transfer")

    def test_sweep_sizes_are_legal_and_different(self):
        for seed in range(100):
            task = make_task(seed)
            for (w, h, t) in scadtask.sweep_sizes(task, 3) + scadtask.oracle_sizes(task, 6):
                self.assertNotEqual(w, task["target"]["width"])
                self.assertNotEqual(h, task["target"]["height"])
                exp = scadtask.expected_for(task, w, h)
                from boltgym import rule_violations
                self.assertEqual(rule_violations(exp, w, h), 0)
        self.assertEqual(scadtask.expected_for(task, task["target"]["width"],
                                               task["target"]["height"]), task["expected"])

    def test_moved_or_resized_plate_fails(self):
        task = make_task(4)
        good = scadtask.write_solution(task, "explicit")
        moved = good.replace("cube([plate_width, plate_height, plate_thickness]);",
                             "translate([5, 0, 0]) cube([plate_width, plate_height, plate_thickness]);")
        self.assertIn("moved", scadtask.verify_scad(moved, task)["error"])
        fixed = good.replace("cube([plate_width, plate_height, plate_thickness]);",
                             "cube([%s, plate_height, plate_thickness]);"
                             % scadtask.num(task["target"]["width"]))
        r = scadtask.verify_scad(fixed, task)
        self.assertTrue(r["success_default"])
        self.assertFalse(r["success"])
        self.assertIn("parameters say", r["error"])

    def test_env(self):
        env = scadtask.ScadEnv(attempts=3)
        obs = env.reset(5)
        self.assertIn("plate_width", obs["target_scad"])
        self.assertEqual(obs["attempts_left"], 3)
        obs, r, done, info = env.step("difference() { cube(1);")
        self.assertEqual((r, done, info["stage"]), (0.0, False, "parse"))
        self.assertIn("line", obs["feedback"])
        obs, r1, done, info = env.step(scadtask.write_solution(env.task, "list", "literal_coords")
                                       or scadtask.write_solution(env.task, "list", "proportional"))
        self.assertFalse(done)
        self.assertGreater(r1, 0.0)
        self.assertLess(r1, 1.0)
        obs, r2, done, info = env.step(scadtask.write_solution(env.task, "list"))
        self.assertTrue(done and info["success"])
        self.assertAlmostEqual(r1 + r2, 2.0)
        self.assertRaises(RuntimeError, env.step, "cube(1);")
        env.reset(6)
        for _ in range(3):
            _, r, done, info = env.step(12345)
        self.assertTrue(done)
        self.assertFalse(info["success"])
        self.assertRaises(ValueError, scadtask.ScadEnv, profile="nope")

    def test_mutants_never_crash_the_verifier(self):
        rng = random.Random(3)
        kinds = set()
        for seed in range(60):
            task = make_task(seed)
            code, kind = scadtask.mutate(scadtask.write_solution(task, scadtask.STYLES[seed % 3]), rng)
            kinds.add(kind)
            r = scadtask.verify_scad(code, task)
            self.assertIn(r["stage"], ("parse", "params", "default", "sweep", "passed"))
        self.assertGreaterEqual(len(kinds), 4)


if __name__ == "__main__":
    unittest.main()
