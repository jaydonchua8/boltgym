"""params: read and edit the slider parameters at the top of an OpenSCAD file.

The annotation format is the OpenSCAD Customizer one, the same vocabulary the
open-source CADAM app tells its model to emit:

    /* [Group Name] */
    // Description of the parameter
    name = 10;        // [1:50]          min:max
    name = 10;        // [1:0.5:50]      min:step:max
    name = "red";     // [red, green]    options, optionally value:Label
    name = 10;        // 0.5             step (max length for strings)

Only literal values before the first module or function definition count. This
version works on tokens instead of a regular expression, so a declaration that
spans lines, a nested array, a semicolon inside a string, or commented-out code
is handled the way OpenSCAD itself would read it.

Standard library only.
"""
import re

from scad import ScadError, tokenize

_GROUP = re.compile(r"/\*\s*\[([^\]]+)\]\s*\*/\Z")
_NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\Z")

CSS_COLORS = frozenset("""aliceblue antiquewhite aqua aquamarine azure beige bisque black
blanchedalmond blue blueviolet brown burlywood cadetblue chartreuse chocolate coral
cornflowerblue cornsilk crimson cyan darkblue darkcyan darkgoldenrod darkgray darkgreen
darkgrey darkkhaki darkmagenta darkolivegreen darkorange darkorchid darkred darksalmon
darkseagreen darkslateblue darkslategray darkslategrey darkturquoise darkviolet deeppink
deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite forestgreen fuchsia gainsboro
ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink indianred indigo ivory
khaki lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan
lightgoldenrodyellow lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen
lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime limegreen linen
magenta maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen
mediumslateblue mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream
mistyrose moccasin navajowhite navy oldlace olive olivedrab orange orangered orchid
palegoldenrod palegreen paleturquoise palevioletred papayawhip peachpuff peru pink plum
powderblue purple rebeccapurple red rosybrown royalblue saddlebrown salmon sandybrown seagreen
seashell sienna silver skyblue slateblue slategray slategrey snow springgreen steelblue tan
teal thistle tomato turquoise violet wheat white whitesmoke yellow yellowgreen""".split())


def is_color(value):
    """True for an opaque CSS colour: a named colour, #rgb or #rrggbb."""
    if not isinstance(value, str):
        return False
    v = value.strip().lower()
    return v in CSS_COLORS or bool(re.match(r"#(?:[0-9a-f]{3}|[0-9a-f]{6})\Z", v))


def _literal(toks):
    """Turn value tokens into (python value, leaf spans) or None when it is an expression.

    Leaf spans is a list of (index path, start offset, end offset) for every scalar.
    """
    pos = [0]

    def scalar():
        if pos[0] >= len(toks):
            return None
        t = toks[pos[0]]
        start = t.start
        sign = 1.0
        if t.kind == "op" and t.text in "+-" and pos[0] + 1 < len(toks) \
                and toks[pos[0] + 1].kind == "num":
            sign = -1.0 if t.text == "-" else 1.0
            pos[0] += 1
            t = toks[pos[0]]
        if t.kind == "num":
            pos[0] += 1
            return (sign * float(t.text), start, t.end)
        if t.kind == "str" and sign == 1.0 and start == t.start:
            pos[0] += 1
            return (re.sub(r"\\(.)", r"\1", t.text[1:-1]), start, t.end)
        if t.kind == "id" and t.text in ("true", "false"):
            pos[0] += 1
            return (t.text == "true", start, t.end)
        return None

    def value(path, leaves):
        if pos[0] < len(toks) and toks[pos[0]].kind == "op" and toks[pos[0]].text == "[":
            pos[0] += 1
            items = []
            while True:
                if pos[0] >= len(toks):
                    return None
                if toks[pos[0]].text == "]" and toks[pos[0]].kind == "op":
                    pos[0] += 1
                    return (items,)
                got = value(path + (len(items),), leaves)
                if got is None:
                    return None
                items.append(got[0])
                if pos[0] < len(toks) and toks[pos[0]].kind == "op" and toks[pos[0]].text == ",":
                    pos[0] += 1
                elif not (pos[0] < len(toks) and toks[pos[0]].text == "]"):
                    return None
        s = scalar()
        if s is None:
            return None
        leaves.append((path, s[1], s[2]))
        return (s[0],)

    leaves = []
    got = value((), leaves)
    if got is None or pos[0] != len(toks):
        return None
    return got[0], leaves


def _kind(v):
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    return None


def _annotation(text, kind):
    """Parse the trailing comment of a declaration into (range, options)."""
    raw = text[2:].strip()
    rng, options = {}, []
    if not raw:
        return rng, options
    if _NUMBER.match(raw):
        return ({"max": float(raw)} if kind == "string" else {"step": float(raw)}), options
    if not (raw.startswith("[") and raw.endswith("]")):
        return rng, options
    inner = raw[1:-1].strip()
    if "," in inner:
        for part in inner.split(","):
            bits = part.strip().split(":", 1)
            val = bits[0].strip()
            if kind == "number":
                if not _NUMBER.match(val):
                    return {}, []
                val = float(val)
            opt = {"value": val}
            if len(bits) == 2:
                opt["label"] = bits[1].strip()
            options.append(opt)
        return rng, options
    bits = [b.strip() for b in inner.split(":")]
    if not all(_NUMBER.match(b) for b in bits):
        return rng, options
    nums = [float(b) for b in bits]
    if len(nums) == 1:
        rng = {"max": nums[0]}
    elif len(nums) == 2:
        rng = {"min": nums[0], "max": nums[1]}
    elif len(nums) == 3:
        rng = {"min": nums[0], "step": nums[1], "max": nums[2]}
    return rng, options


def _display(name):
    if name == "$fn":
        return "Resolution"
    return " ".join(w[0].upper() + w[1:] for w in name.replace("_", " ").split(" ") if w)


def _declarations(src):
    """Yield one record per top-level literal declaration, in file order."""
    toks = tokenize(src, comments=True)
    out, group, depth, i, n = [], "", 0, 0, len(toks)
    while i < n:
        t = toks[i]
        if t.kind == "bc":
            m = _GROUP.match(t.text)
            if m and depth == 0:
                group = m.group(1).strip()
            i += 1
            continue
        if t.kind == "lc":
            i += 1
            continue
        if t.kind == "op" and t.text in "([{":
            depth += 1
        elif t.kind == "op" and t.text in ")]}":
            depth -= 1
        elif t.kind == "id" and depth == 0 and t.text in ("module", "function"):
            break
        elif t.kind == "id" and depth == 0:
            j = i + 1
            while j < n and toks[j].kind in ("lc", "bc"):
                j += 1
            starts_statement = True
            k = i - 1
            while k >= 0 and toks[k].kind in ("lc", "bc"):
                k -= 1
            if k >= 0 and not (toks[k].kind == "op" and toks[k].text in ";}"):
                starts_statement = toks[k].kind == "path"
            if starts_statement and j < n and toks[j].kind == "op" and toks[j].text == "=":
                vals, k = [], j + 1
                while k < n and not (toks[k].kind == "op" and toks[k].text == ";"):
                    if toks[k].kind not in ("lc", "bc"):
                        vals.append(toks[k])
                    k += 1
                if k < n:
                    lit = _literal(vals) if vals else None
                    if lit is not None:
                        trailing = None
                        if k + 1 < n and toks[k + 1].kind == "lc" and toks[k + 1].line == toks[k].line \
                                and "\n" not in src[toks[k].end:toks[k + 1].start]:
                            trailing = toks[k + 1].text
                        desc = None
                        if i > 0 and toks[i - 1].kind == "lc":
                            c = toks[i - 1]
                            before = src[src.rfind("\n", 0, c.start) + 1:c.start]
                            if before.strip() == "" and src.count("\n", c.end, t.start) == 1:
                                desc = c.text.lstrip("/").strip() or None
                        out.append({"name": t.text, "value": lit[0], "leaves": lit[1],
                                    "group": group, "description": desc, "trailing": trailing,
                                    "line": t.line})
                    i = k + 1
                    continue
        i += 1
    return out


def _flatten(name, value, path=()):
    if isinstance(value, list):
        for idx, item in enumerate(value):
            for row in _flatten("%s[%d]" % (name, idx), item, path + (idx,)):
                yield row
    else:
        yield name, value, path


def extract(src):
    """Return the parameters a UI should expose, as a list of dicts.

    Keys follow CADAM's Parameter type: name, displayName, value, defaultValue,
    type, description, group, range, options. A numeric array becomes one entry
    per element named name[0], name[1], and a nested array name[0][1]. When a name
    is declared twice the last declaration wins, as it does in OpenSCAD.
    Raises ScadError when the file cannot be tokenized.
    """
    latest = {}
    for d in _declarations(src):
        latest[d["name"]] = d
    out = []
    for d in sorted(latest.values(), key=lambda r: r["line"]):
        leaves = list(_flatten(d["name"], d["value"]))
        kinds = set(_kind(v) for (_, v, _) in leaves)
        if not leaves or None in kinds:
            continue
        is_array = isinstance(d["value"], list)
        if is_array and kinds != {"number"}:
            if len(kinds) != 1 or any(isinstance(x, list) for x in d["value"]):
                continue
            kind = kinds.pop()
            rng, options = _annotation(d["trailing"], kind) if d["trailing"] else ({}, [])
            out.append({"name": d["name"], "displayName": _display(d["name"]),
                        "value": d["value"], "defaultValue": d["value"], "type": kind + "[]",
                        "description": d["description"], "group": d["group"],
                        "range": rng, "options": options})
            continue
        kind = kinds.pop() if len(kinds) == 1 else None
        if kind is None:
            continue
        rng, options = _annotation(d["trailing"], kind) if d["trailing"] else ({}, [])
        labels = ["X", "Y", "Z"] if is_array and len(d["value"]) <= 3 else None
        for (name, v, path) in leaves:
            display = _display(d["name"])
            if is_array:
                tag = labels[path[0]] if labels and len(path) == 1 else \
                    " ".join(str(p + 1) for p in path)
                display = "%s %s" % (display, tag)
            out.append({"name": name, "displayName": display, "value": v, "defaultValue": v,
                        "type": kind, "description": d["description"], "group": d["group"],
                        "range": dict(rng), "options": list(options)})
    return out


def summary(src):
    """Count controls the way CADAM's benchmark table does: dimensions and colours."""
    ps = extract(src)
    colors = sum(1 for p in ps if p["type"] == "string" and is_color(p["value"]))
    dims = sum(1 for p in ps if p["type"] == "number")
    return {"dims": dims, "colors": colors, "other": len(ps) - dims - colors}


def _format(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError("parameter value must be finite")
        return str(int(f)) if f == int(f) and abs(f) < 1e15 else repr(f)
    if isinstance(value, str):
        return '"%s"' % value.replace("\\", "\\\\").replace('"', '\\"')
    raise ValueError("unsupported parameter value %r" % (value,))


_INDEXED = re.compile(r"([A-Za-z_$][A-Za-z0-9_]*)((?:\[\d+\])*)\Z")


def set_params(src, changes):
    """Return src with several parameters changed and everything else byte-identical.

    changes maps a name to its new value. A name may address an array element as
    name[1] or name[0][1]. The declaration edited is the last top-level one, the
    one OpenSCAD actually uses. Raises KeyError when a name is not a literal
    top-level declaration and ValueError when a new value has a different type
    from the old one. Nothing is changed unless every edit is valid.
    """
    latest = {}
    for d in _declarations(src):
        latest[d["name"]] = d
    edits = []
    for name, value in changes.items():
        m = _INDEXED.match(name)
        if not m or m.group(1) not in latest:
            raise KeyError(name)
        decl = latest[m.group(1)]
        path = tuple(int(x) for x in re.findall(r"\[(\d+)\]", m.group(2)))
        span = [(a, b) for (leaf_path, a, b) in decl["leaves"] if leaf_path == path]
        if not span:
            raise KeyError(name)
        old = decl["value"]
        for idx in path:
            old = old[idx]
        if isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if _kind(value) is None or _kind(value) != _kind(old):
            raise ValueError("%s is a %s, cannot set it to %r" % (name, _kind(old), value))
        edits.append((span[0][0], span[0][1], _format(value)))
    for (a, b, text) in sorted(edits, reverse=True):
        src = src[:a] + text + src[b:]
    return src


def set_param(src, name, value):
    """Change one parameter. See set_params."""
    return set_params(src, {name: value})


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) != 2:
        print("usage: python params.py <file.scad>")
        sys.exit(2)
    try:
        print(json.dumps(extract(open(sys.argv[1]).read()), indent=1))
    except ScadError as exc:
        print("ScadError: %s" % exc)
        sys.exit(1)
