import glob
import os
import re
import unittest

from params import extract, is_color, set_param, set_params, summary
from scad import ScadError, read_plate

SAMPLE = """// Bracket (synthetic)

/* [Dimensions] */
// Length of each flange
flange_length = 80;        // [40:1:150]
// Thickness
thickness = 5.5;           // [2:15]
offset = -3; // 0.5

/* [Options] */
// Edge style
edge = "round"; // [round, chamfer:Chamfered, square]
teeth = 12; // [8, 12:Twelve, 16]
label = "A1"; // 8
show_holes = true;
part_color = "#4682B4"; // color
size = [10, 20, 30];

$fn = 64;
derived = flange_length * 2;

module body() { inner = 5; cube(inner); }
after = 7;
"""


def by_name(src):
    return dict((p["name"], p) for p in extract(src))


class TestExtract(unittest.TestCase):
    def test_names_and_order(self):
        self.assertEqual([p["name"] for p in extract(SAMPLE)],
                         ["flange_length", "thickness", "offset", "edge", "teeth", "label",
                          "show_holes", "part_color", "size[0]", "size[1]", "size[2]", "$fn"])

    def test_types_ranges_options(self):
        p = by_name(SAMPLE)
        self.assertEqual(p["flange_length"]["range"], {"min": 40.0, "step": 1.0, "max": 150.0})
        self.assertEqual(p["thickness"]["range"], {"min": 2.0, "max": 15.0})
        self.assertEqual(p["thickness"]["value"], 5.5)
        self.assertEqual(p["offset"]["value"], -3.0)
        self.assertEqual(p["offset"]["range"], {"step": 0.5})
        self.assertEqual(p["edge"]["type"], "string")
        self.assertEqual(p["edge"]["options"], [{"value": "round"},
                                                {"value": "chamfer", "label": "Chamfered"},
                                                {"value": "square"}])
        self.assertEqual(p["teeth"]["options"][1], {"value": 12.0, "label": "Twelve"})
        self.assertEqual(p["label"]["range"], {"max": 8.0})
        self.assertEqual(p["show_holes"]["type"], "boolean")
        self.assertEqual(p["part_color"]["range"], {})
        self.assertEqual(p["size[1]"]["value"], 20.0)
        self.assertEqual(p["size[1]"]["displayName"], "Size Y")

    def test_groups_descriptions_display_names(self):
        p = by_name(SAMPLE)
        self.assertEqual(p["flange_length"]["group"], "Dimensions")
        self.assertEqual(p["edge"]["group"], "Options")
        self.assertEqual(p["flange_length"]["description"], "Length of each flange")
        self.assertIsNone(p["offset"]["description"])
        self.assertIsNone(p["teeth"]["description"])
        self.assertEqual(p["flange_length"]["displayName"], "Flange Length")
        self.assertEqual(p["$fn"]["displayName"], "Resolution")
        self.assertEqual(by_name("__w_ = 1;")["__w_"]["displayName"], "W")

    def test_expressions_and_module_internals_are_not_parameters(self):
        p = by_name(SAMPLE)
        for hidden in ("derived", "inner", "after"):
            self.assertNotIn(hidden, p)

    def test_cases_a_line_regex_gets_wrong(self):
        src = ('/*\nold_width = 5;\n*/\n'
               '// module notes: not a definition\n'
               'label = "a;b"; // 12\n'
               'size = [10,\n        20];\n'
               'pts = [[0, 0], [10, 5]];\n'
               'tol = 1e-3;\nhalf = .5;\n'
               'a = 1; b = 2;\n'
               'c = 3; // [0:5] trailing\n'
               'translate([w = 4, 0, 0]) cube(1);\n'
               'real_width = 9;\n')
        p = by_name(src)
        self.assertNotIn("old_width", p)
        self.assertEqual(p["label"]["value"], "a;b")
        self.assertEqual(p["label"]["range"], {"max": 12.0})
        self.assertEqual((p["size[0]"]["value"], p["size[1]"]["value"]), (10.0, 20.0))
        self.assertEqual(p["pts[1][0]"]["value"], 10.0)
        self.assertEqual(p["tol"]["value"], 0.001)
        self.assertEqual(p["half"]["value"], 0.5)
        self.assertEqual((p["a"]["value"], p["b"]["value"]), (1.0, 2.0))
        self.assertEqual(p["c"]["range"], {})
        self.assertNotIn("w", p)
        self.assertEqual(p["real_width"]["value"], 9.0)

    def test_last_declaration_wins(self):
        src = "w = 1; // [0:5]\nw = 2; // [0:9]\ncube(w);"
        self.assertEqual(by_name(src)["w"]["value"], 2.0)
        self.assertEqual(by_name(src)["w"]["range"], {"min": 0.0, "max": 9.0})
        self.assertIn("w = 1; // [0:5]\nw = 7;", set_param(src, "w", 7))

    def test_bad_input(self):
        self.assertEqual(extract(""), [])
        self.assertEqual(extract("x = ;"), [])
        self.assertEqual(extract("x = 5"), [])
        self.assertEqual(extract("mixed = [1, \"a\"];"), [])
        self.assertRaises(ScadError, extract, 'x = "never closed;')
        self.assertRaises(ScadError, extract, "/* open\nx = 1;")

    def test_colors(self):
        for good in ("red", "LightSlateGray", "#4682B4", "#fff", " Peru "):
            self.assertTrue(is_color(good), good)
        for bad in ("M12x1.75", "#12", "#gggggg", "", "round", 5, None):
            self.assertFalse(is_color(bad), bad)
        self.assertEqual(summary(SAMPLE), {"dims": 8, "colors": 1, "other": 3})


class TestSetParam(unittest.TestCase):
    def test_changes_only_the_value(self):
        out = set_param(SAMPLE, "flange_length", 95.5)
        self.assertIn("flange_length = 95.5;        // [40:1:150]", out)
        self.assertEqual(out.replace("95.5", "80", 1), SAMPLE)
        self.assertEqual(by_name(out)["flange_length"]["range"]["max"], 150.0)

    def test_types(self):
        out = set_params(SAMPLE, {"edge": 'sq"uare', "show_holes": False, "size[2]": 31,
                                  "offset": -0.25, "thickness": 6})
        p = by_name(out)
        self.assertEqual(p["edge"]["value"], 'sq"uare')
        self.assertIs(p["show_holes"]["value"], False)
        self.assertEqual(p["size[2]"]["value"], 31.0)
        self.assertEqual(p["offset"]["value"], -0.25)
        self.assertIn("thickness = 6; ", out)
        self.assertIn("size = [10, 20, 31];", out)

    def test_round_trip_every_parameter(self):
        src = SAMPLE
        for p in extract(SAMPLE):
            new = {"number": 7.25, "string": "x y", "boolean": not p["value"]}[p["type"]]
            src = set_param(src, p["name"], new)
            self.assertEqual(by_name(src)[p["name"]]["value"], new)
        self.assertEqual(len(extract(src)), len(extract(SAMPLE)))
        self.assertEqual(src.count("\n"), SAMPLE.count("\n"))

    def test_rejects_bad_edits(self):
        self.assertRaises(KeyError, set_param, SAMPLE, "derived", 5)
        self.assertRaises(KeyError, set_param, SAMPLE, "inner", 5)
        self.assertRaises(KeyError, set_param, SAMPLE, "nosuch", 5)
        self.assertRaises(KeyError, set_param, SAMPLE, "size[3]", 5)
        self.assertRaises(KeyError, set_param, SAMPLE, "size", 5)
        self.assertRaises(KeyError, set_param, SAMPLE, "bad name", 5)
        self.assertRaises(ValueError, set_param, SAMPLE, "thickness", "thick")
        self.assertRaises(ValueError, set_param, SAMPLE, "thickness", True)
        self.assertRaises(ValueError, set_param, SAMPLE, "thickness", float("nan"))
        self.assertRaises(ValueError, set_param, SAMPLE, "edge", 3)
        self.assertRaises(ValueError, set_params, SAMPLE, {"thickness": 6, "edge": 3})

    def test_edit_ignores_commented_out_declaration(self):
        src = "/*\nw = 5;\n*/\n// w = 6;\nw = 40;\ndifference() { cube([w, 30, 4]); }\n"
        out = set_param(src, "w", 55)
        self.assertTrue(out.startswith("/*\nw = 5;\n*/\n// w = 6;\nw = 55;"))
        self.assertEqual(read_plate(out)["width"], 55.0)

    def test_edit_drives_geometry(self):
        src = "size = [40,\n  30];\nt = 4;\ndifference() { cube([size.x, size.y, t]); }\n"
        p = read_plate(set_params(src, {"size[1]": 35, "t": 5}))
        self.assertEqual((p["width"], p["height"], p["thickness"]), (40.0, 35.0, 5.0))


@unittest.skipUnless(os.environ.get("CADAM_DIR"), "set CADAM_DIR to a clone of Adam-CAD/CADAM")
class TestAgainstCadamBenchmarks(unittest.TestCase):
    """Reads CADAM's published benchmark models. They are not copied into this repo."""

    def test_control_counts_match_published_table(self):
        root = os.path.join(os.environ["CADAM_DIR"], "benchmarks")
        table = open(os.path.join(root, "README.md"), encoding="utf-8").read()
        want = dict((m.group(1), (int(m.group(2)), int(m.group(3)))) for m in re.finditer(
            r"\]\((\d\d-[a-z0-9-]+)\.md\).*?\|\s*(\d+) dims? · (\d+) colors?", table))
        files = sorted(glob.glob(os.path.join(root, "*.scad")))
        self.assertGreaterEqual(len(files), 13)
        for path in files:
            name = os.path.basename(path)[:-5]
            s = summary(open(path, encoding="utf-8").read())
            self.assertEqual((s["dims"] + s["other"], s["colors"]), want[name], name)


if __name__ == "__main__":
    unittest.main()
