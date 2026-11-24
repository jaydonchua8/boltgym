"""scad: tokenizer, parser and evaluator for a subset of OpenSCAD.

The parser accepts most of the language (modules, functions, list comprehensions,
let, ranges, modifiers). The evaluator is deliberately narrower: it understands
plates and holes (cube, cylinder, translate, rotate about z, difference, union,
for, if, let, user modules and functions) and raises ScadError with a line number
for everything else. It never executes anything outside this file and it stops at
a fixed operation budget, so it is safe to run on code written by an agent.

Standard library only.
"""
import math
import re

MAX_OPS = 200000     # evaluation budget per program
MAX_DEPTH = 48       # nested user module / function calls


class ScadError(Exception):
    def __init__(self, message, line=None):
        Exception.__init__(self, message if line is None else "line %d: %s" % (line, message))
        self.line = line


# ---- tokenizer ---------------------------------------------------------------

_TOKEN = re.compile(r"""
  (?P<ws>\s+)
 |(?P<lc>//[^\n]*)
 |(?P<bc>/\*.*?\*/)
 |(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
 |(?P<id>[A-Za-z_$][A-Za-z0-9_]*)
 |(?P<str>"(?:[^"\\]|\\.)*")
 |(?P<op><=|>=|==|!=|&&|\|\||[-+*/%^!<>?:=;,.()\[\]{}\#])
""", re.S | re.X)
_PATH = re.compile(r"[ \t]*<([^>\n]*)>")


class Token(object):
    __slots__ = ("kind", "text", "line", "start", "end")

    def __init__(self, kind, text, line, start, end):
        self.kind, self.text, self.line, self.start, self.end = kind, text, line, start, end

    def __repr__(self):
        return "Token(%s, %r, line %d)" % (self.kind, self.text, self.line)


def tokenize(src, comments=False):
    """Return a list of Tokens. Comment tokens (kind 'lc' and 'bc') are kept only
    when comments=True. Offsets are character positions in src."""
    if not isinstance(src, str):
        raise ScadError("source must be text")
    out, pos, line, n = [], 0, 1, len(src)
    match, add = _TOKEN.match, out.append
    while pos < n:
        m = match(src, pos)
        if not m:
            if src.startswith("/*", pos):
                raise ScadError("block comment is never closed", line)
            if src[pos] == '"':
                raise ScadError("string is never closed", line)
            raise ScadError("unexpected character %r" % src[pos], line)
        kind, end = m.lastgroup, m.end()
        if kind == "ws":
            line += src.count("\n", pos, end)
        elif kind == "lc":
            if comments:
                add(Token(kind, m.group(), line, pos, end))
        elif kind == "bc" or kind == "str":
            if comments or kind == "str":
                add(Token(kind, m.group(), line, pos, end))
            line += src.count("\n", pos, end)
        else:
            text = m.group()
            if text == "/" and src.startswith("/*", pos):
                raise ScadError("block comment is never closed", line)
            add(Token(kind, text, line, pos, end))
            if kind == "id" and (text == "include" or text == "use"):
                p = _PATH.match(src, end)
                if p:
                    add(Token("path", p.group(1), line, end, p.end()))
                    end = p.end()
        pos = end
    return out


# ---- parser ------------------------------------------------------------------

_BINARY = [("||",), ("&&",), ("==", "!="), ("<", "<=", ">", ">="), ("+", "-"), ("*", "/", "%")]
_END = Token("eof", "", 0, 0, 0)


class Parser(object):
    def __init__(self, src):
        self.toks = tokenize(src)
        self.i = 0
        self.last_line = self.toks[-1].line if self.toks else 1

    def peek(self, k=0):
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else _END

    def at(self, text, k=0):
        t = self.peek(k)
        return t.text == text and t.kind in ("op", "id")

    def take(self, text=None):
        t = self.peek()
        if t is _END:
            raise ScadError("file ends early, expected %s" % (repr(text) if text else "more code"),
                            self.last_line)
        if text is not None and not (t.text == text and t.kind in ("op", "id")):
            raise ScadError("expected %r but found %r" % (text, t.text), t.line)
        self.i += 1
        return t

    # statements
    def program(self):
        body = []
        while self.peek() is not _END:
            body.append(self.statement())
        return body

    def statement(self):
        t = self.peek()
        if t.kind == "op":
            if t.text == ";":
                self.take()
                return ("block", [], t.line)
            if t.text == "{":
                self.take()
                body = []
                while not self.at("}"):
                    body.append(self.statement())
                self.take("}")
                return ("block", body, t.line)
            if t.text in "*!#%":
                self.take()
                inner = self.statement()
                return ("block", [], t.line) if t.text == "*" else inner
            raise ScadError("unexpected %r" % t.text, t.line)
        if t.kind != "id":
            if t is _END:
                raise ScadError("file ends early, a statement is missing", self.last_line)
            raise ScadError("unexpected %r" % t.text, t.line)
        if t.text in ("include", "use"):
            self.take()
            p = self.take()
            if p.kind != "path":
                raise ScadError("expected <file> after %s" % t.text, t.line)
            return ("use", t.text, p.text, t.line)
        if t.text == "module":
            self.take()
            name = self.ident()
            params = self.params()
            return ("module", name, params, self.statement(), t.line)
        if t.text == "function":
            self.take()
            name = self.ident()
            params = self.params()
            self.take("=")
            e = self.expr()
            self.take(";")
            return ("function", name, params, e, t.line)
        if t.text == "if":
            self.take()
            self.take("(")
            cond = self.expr()
            self.take(")")
            then = self.statement()
            other = None
            if self.at("else"):
                self.take()
                other = self.statement()
            return ("if", cond, then, other, t.line)
        if self.at("=", 1):
            self.take()
            self.take()
            e = self.expr()
            self.take(";")
            return ("assign", t.text, e, t.line)
        name = self.ident()
        args = self.args()
        if self.at(";"):
            self.take()
            child = None
        else:
            child = self.statement()
        return ("inst", name, args, child, t.line)

    def ident(self):
        t = self.take()
        if t.kind != "id":
            raise ScadError("expected a name but found %r" % t.text, t.line)
        return t.text

    def params(self):
        self.take("(")
        out = []
        while not self.at(")"):
            name = self.ident()
            default = None
            if self.at("="):
                self.take()
                default = self.expr()
            out.append((name, default))
            if not self.at(")"):
                self.take(",")
        self.take(")")
        return out

    def args(self):
        self.take("(")
        out = []
        while not self.at(")"):
            if self.peek().kind == "id" and self.at("=", 1):
                name = self.take().text
                self.take()
                out.append((name, self.expr()))
            else:
                out.append((None, self.expr()))
            if not self.at(")"):
                self.take(",")
        self.take(")")
        return out

    # expressions
    def expr(self):
        t = self.peek()
        if t.kind == "id":
            if t.text == "let" and self.at("(", 1):
                self.take()
                return ("let", self.args(), self.expr(), t.line)
            if t.text in ("assert", "echo") and self.at("(", 1):
                self.take()
                args = self.args()
                starts = self.peek()
                if starts is _END or (starts.kind == "op" and starts.text in ";,)]:"):
                    return ("undef",)
                return ("check" if t.text == "assert" else "let", args if t.text == "assert" else [],
                        self.expr(), t.line)
            if t.text == "function" and self.at("(", 1):
                self.take()
                return ("fnlit", self.params(), self.expr(), t.line)
        cond = self.binary(0)
        if self.at("?"):
            self.take()
            a = self.expr()
            self.take(":")
            return ("tern", cond, a, self.expr())
        return cond

    def binary(self, level):
        if level == len(_BINARY):
            return self.unary()
        left = self.binary(level + 1)
        while self.peek().kind == "op" and self.peek().text in _BINARY[level]:
            op = self.take()
            left = ("bin", op.text, left, self.binary(level + 1), op.line)
        return left

    def unary(self):
        t = self.peek()
        if t.kind == "op" and t.text in "-+!":
            self.take()
            return ("un", t.text, self.unary(), t.line)
        base = self.postfix()
        if self.at("^"):
            op = self.take()
            return ("bin", "^", base, self.unary(), op.line)
        return base

    def postfix(self):
        e = self.primary()
        while True:
            t = self.peek()
            if t.kind != "op":
                return e
            if t.text == "(":
                e = ("call", e, self.args(), t.line)
            elif t.text == "[":
                self.take()
                idx = self.expr()
                self.take("]")
                e = ("index", e, idx, t.line)
            elif t.text == ".":
                self.take()
                e = ("index", e, ("num", {"x": 0.0, "y": 1.0, "z": 2.0}.get(self.ident(), -1.0)),
                     t.line)
            else:
                return e

    def primary(self):
        t = self.take()
        if t.kind == "num":
            return ("num", float(t.text))
        if t.kind == "str":
            return ("str", re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)),
                                  t.text[1:-1]))
        if t.kind == "id":
            if t.text in ("true", "false"):
                return ("bool", t.text == "true")
            if t.text == "undef":
                return ("undef",)
            return ("var", t.text, t.line)
        if t.text == "(":
            e = self.expr()
            self.take(")")
            return e
        if t.text == "[":
            if self.at("]"):
                self.take()
                return ("vec", [], t.line)
            first = self.element()
            if self.at(":"):
                self.take()
                second = self.expr()
                third = None
                if self.at(":"):
                    self.take()
                    third = self.expr()
                self.take("]")
                if third is None:
                    return ("range", first, None, second, t.line)
                return ("range", first, second, third, t.line)
            items = [first]
            while self.at(","):
                self.take()
                if self.at("]"):
                    break
                items.append(self.element())
            self.take("]")
            return ("vec", items, t.line)
        raise ScadError("unexpected %r in an expression" % t.text, t.line)

    def element(self):
        """One vector element, which may be a list comprehension generator."""
        t = self.peek()
        if t.kind == "id" and self.at("(", 1):
            if t.text == "for":
                self.take()
                self.take("(")
                binds = []
                while not self.at(")"):
                    name = self.ident()
                    self.take("=")
                    binds.append((name, self.expr()))
                    if self.at(";"):
                        raise ScadError("C-style for is not supported", t.line)
                    if not self.at(")"):
                        self.take(",")
                self.take(")")
                return ("lc_for", binds, self.element(), t.line)
            if t.text == "if":
                self.take()
                self.take("(")
                cond = self.expr()
                self.take(")")
                a = self.element()
                b = None
                if self.at("else"):
                    self.take()
                    b = self.element()
                return ("lc_if", cond, a, b, t.line)
            if t.text == "let":
                self.take()
                return ("lc_let", self.args(), self.element(), t.line)
        if t.kind == "id" and t.text == "each":
            self.take()
            return ("each", self.element(), t.line)
        if t.kind == "op" and t.text == "(" and self.peek(1).kind == "id" \
                and self.peek(1).text in ("for", "if", "each") and self.peek(2).text in ("(", "["):
            self.take()
            e = self.element()
            self.take(")")
            return e
        return self.expr()


def parse(src):
    """Parse OpenSCAD source into a list of statement tuples. Raises ScadError."""
    try:
        return Parser(src).program()
    except RecursionError:
        raise ScadError("code is nested too deeply")


# ---- evaluator ---------------------------------------------------------------

class Range(object):
    __slots__ = ("a", "step", "b")

    def __init__(self, a, step, b):
        self.a, self.step, self.b = a, step, b

    def values(self, budget):
        a, s, b = self.a, self.step, self.b
        if not all(_num(v) for v in (a, s, b)) or s == 0 or any(math.isnan(v) or math.isinf(v)
                                                                 for v in (a, s, b)):
            return []
        n = int(math.floor((b - a) / s + 1e-9)) + 1
        if n <= 0:
            return []
        if n > budget:
            raise ScadError("range has %d steps, over the evaluation budget" % n)
        return [a + i * s for i in range(n)]


class Function(object):
    __slots__ = ("params", "body", "scope")

    def __init__(self, params, body, scope):
        self.params, self.body, self.scope = params, body, scope


class Scope(object):
    __slots__ = ("vars", "parent")

    def __init__(self, parent=None):
        self.vars, self.parent = {}, parent

    def get(self, name):
        s = self
        while s is not None:
            if name in s.vars:
                return s.vars[name]
            s = s.parent
        return None


def _num(v):
    return isinstance(v, float)


def _deg(f):
    return lambda x: f(math.radians(x))


def _inv(f):
    def g(x):
        return math.degrees(f(x)) if -1.0 <= x <= 1.0 else float("nan")
    return g


def _round(x):
    return math.floor(abs(x) + 0.5) * (1.0 if x >= 0 else -1.0)


_MATH1 = {"sin": _deg(math.sin), "cos": _deg(math.cos), "tan": _deg(math.tan),
          "asin": _inv(math.asin), "acos": _inv(math.acos),
          "atan": lambda x: math.degrees(math.atan(x)), "abs": abs, "floor": math.floor,
          "ceil": math.ceil, "round": _round, "exp": math.exp,
          "sqrt": lambda x: math.sqrt(x) if x >= 0 else float("nan"),
          "ln": lambda x: math.log(x) if x > 0 else float("nan"),
          "log": lambda x: math.log10(x) if x > 0 else float("nan"),
          "sign": lambda x: float((x > 0) - (x < 0))}

PASS_THROUGH = ("union", "color", "render", "group")
IGNORED = ("echo",)


class Evaluator(object):
    """Runs a parsed program and collects primitives.

    Each primitive is a dict: kind 'box' with lo and hi corners, or kind 'cyl' with
    x, y, z0, z1, d0 (diameter at z0) and d1. 'neg' is True for geometry that a
    difference() subtracts.
    """

    def __init__(self, max_ops=MAX_OPS):
        self.ops, self.max_ops, self.depth = 0, max_ops, 0
        self.prims = []
        self.modules, self.functions = {}, {}
        self.globals = Scope()

    def tick(self, line=None):
        self.ops += 1
        if self.ops > self.max_ops:
            raise ScadError("evaluation budget of %d operations exceeded" % self.max_ops, line)

    def run(self, program):
        self.block(program, self.globals, (0.0, 0.0, 0.0, 0.0), False)
        return self.prims

    # statements
    def block(self, stmts, scope, xf, neg):
        for s in stmts:
            if s[0] == "module":
                self.modules[s[1]] = s
            elif s[0] == "function":
                self.functions[s[1]] = Function(s[2], s[3], self.globals)
        for s in stmts:
            self.stmt(s, scope, xf, neg)

    def stmt(self, s, scope, xf, neg):
        kind = s[0]
        self.tick(s[-1])
        if kind == "assign":
            scope.vars[s[1]] = self.ev(s[2], scope)
        elif kind == "block":
            self.block(s[1], Scope(scope) if s[1] else scope, xf, neg)
        elif kind == "if":
            branch = s[2] if _truthy(self.ev(s[1], scope)) else s[3]
            if branch is not None:
                self.stmt(branch, Scope(scope), xf, neg)
        elif kind == "inst":
            self.inst(s, scope, xf, neg)
        elif kind == "use":
            raise ScadError("%s <%s> is outside the supported subset (no libraries)" % (s[1], s[2]),
                            s[3])

    def child(self, s, scope, xf, neg):
        if s[3] is not None:
            self.stmt(s[3], Scope(scope), xf, neg)

    def inst(self, s, scope, xf, neg):
        name, args, line = s[1], s[2], s[4]
        if name == "for":
            self.loop(args, 0, s, scope, xf, neg)
        elif name == "let":
            inner = Scope(scope)
            for (k, e) in args:
                if k is not None:
                    inner.vars[k] = self.ev(e, inner)
            self.child(s, inner, xf, neg)
        elif name == "translate":
            v = self.bind(args, ("v",), scope, line)["v"]
            if not (isinstance(v, list) and 2 <= len(v) <= 3 and all(_fin(c) for c in v)):
                raise ScadError("translate needs a vector of 2 or 3 numbers, got %s" % show(v), line)
            a, tx, ty, tz = xf
            c, si = math.cos(math.radians(a)), math.sin(math.radians(a))
            nxt = (a, tx + c * v[0] - si * v[1], ty + si * v[0] + c * v[1],
                   tz + (v[2] if len(v) == 3 else 0.0))
            self.child(s, scope, nxt, neg)
        elif name == "rotate":
            b = self.bind(args, ("a", "v"), scope, line)
            ang, axis = b["a"], b["v"]
            if isinstance(ang, list):
                if len(ang) != 3 or not all(_fin(c) for c in ang):
                    raise ScadError("rotate needs a number or [x, y, z]", line)
                if abs(ang[0]) > 1e-9 or abs(ang[1]) > 1e-9:
                    raise ScadError("rotate about x or y is outside the supported subset", line)
                ang = ang[2]
            elif axis is not None and axis != [0.0, 0.0, 1.0]:
                raise ScadError("rotate about an axis other than z is outside the supported subset",
                                line)
            if not _fin(ang):
                raise ScadError("rotate angle is %s" % show(ang), line)
            self.child(s, scope, (xf[0] + ang, xf[1], xf[2], xf[3]), neg)
        elif name == "difference":
            if neg:
                raise ScadError("difference() inside a subtracted branch is outside the supported "
                                "subset", line)
            kids = []
            if s[3] is not None:
                kids = s[3][1] if s[3][0] == "block" else [s[3]]
            inner = Scope(scope)
            for k in kids:
                if k[0] == "module":
                    self.modules[k[1]] = k
            first = True
            for k in kids:
                if k[0] in ("assign", "module", "function"):
                    self.stmt(k, inner, xf, neg)
                    continue
                self.stmt(k, inner, xf, not first)
                first = False
        elif name in PASS_THROUGH:
            self.child(s, scope, xf, neg)
        elif name in IGNORED:
            pass
        elif name == "assert":
            vals = [self.ev(e, scope) for (_, e) in args]
            if vals and not _truthy(vals[0]):
                raise ScadError("assert failed" + (": %s" % show(vals[1]) if len(vals) > 1 else ""),
                                line)
            self.child(s, scope, xf, neg)
        elif name == "cube":
            self.cube(self.bind(args, ("size", "center"), scope, line), xf, neg, line)
        elif name == "cylinder":
            self.cylinder(args, scope, xf, neg, line)
        elif name in self.modules:
            self.call_module(self.modules[name], args, scope, xf, neg, line)
        elif name == "children":
            raise ScadError("children() is outside the supported subset", line)
        else:
            raise ScadError("module %s is not defined or is outside the supported subset" % name,
                            line)

    def loop(self, binds, k, s, scope, xf, neg):
        if k == len(binds):
            self.child(s, scope, xf, neg)
            return
        name, e = binds[k]
        if name is None:
            raise ScadError("for needs name = values", s[4])
        for v in self.iterate(self.ev(e, scope), s[4]):
            self.tick(s[4])
            inner = Scope(scope)
            inner.vars[name] = v
            self.loop(binds, k + 1, s, inner, xf, neg)

    def iterate(self, v, line):
        if isinstance(v, Range):
            return v.values(self.max_ops - self.ops)
        if isinstance(v, list):
            return v
        if v is None:
            raise ScadError("for loops over undef", line)
        return [v]

    def bind(self, args, names, scope, line, extra=()):
        out = dict((n, None) for n in names)
        pos = 0
        for (k, e) in args:
            v = self.ev(e, scope)
            if k is None:
                if pos >= len(names):
                    raise ScadError("too many arguments", line)
                out[names[pos]] = v
                pos += 1
            elif k in out or k in extra:
                out[k] = v
            elif not k.startswith("$"):
                raise ScadError("unknown argument %s" % k, line)
        return out

    def call_module(self, mod, args, scope, xf, neg, line):
        if self.depth >= MAX_DEPTH:
            raise ScadError("module calls nested more than %d deep" % MAX_DEPTH, line)
        inner = Scope(self.globals)
        self.bind_params(mod[2], args, scope, inner, line)
        self.depth += 1
        try:
            self.stmt(mod[3], inner, xf, neg)
        finally:
            self.depth -= 1

    def bind_params(self, params, args, scope, inner, line):
        names = [p[0] for p in params]
        pos = 0
        given = {}
        for (k, e) in args:
            v = self.ev(e, scope)
            if k is None:
                if pos < len(names):
                    given[names[pos]] = v
                pos += 1
            else:
                given[k] = v
        for (name, default) in params:
            if name in given:
                inner.vars[name] = given[name]
            else:
                inner.vars[name] = self.ev(default, inner) if default is not None else None
        for k, v in given.items():
            if k.startswith("$"):
                inner.vars[k] = v

    # primitives
    def cube(self, b, xf, neg, line):
        size, center = b["size"], b["center"]
        if size is None:
            size = 1.0
        if _num(size):
            size = [size, size, size]
        if not (isinstance(size, list) and len(size) == 3 and all(_fin(c) and c > 0 for c in size)):
            raise ScadError("cube size must be a positive number or 3 positive numbers, got %s"
                            % show(b["size"]), line)
        a, tx, ty, tz = xf
        quarter = a / 90.0
        if abs(quarter - round(quarter)) > 1e-9:
            raise ScadError("a cube rotated by %g degrees is outside the supported subset" % a, line)
        lo = [-size[0] / 2.0, -size[1] / 2.0, -size[2] / 2.0] if _truthy(center) else [0.0, 0.0, 0.0]
        c, si = round(math.cos(math.radians(a))), round(math.sin(math.radians(a)))
        xs, ys = [], []
        for dx in (lo[0], lo[0] + size[0]):
            for dy in (lo[1], lo[1] + size[1]):
                xs.append(tx + c * dx - si * dy)
                ys.append(ty + si * dx + c * dy)
        self.prims.append({"kind": "box", "neg": neg, "line": line,
                           "lo": (min(xs), min(ys), tz + lo[2]),
                           "hi": (max(xs), max(ys), tz + lo[2] + size[2])})

    def cylinder(self, args, scope, xf, neg, line):
        if sum(1 for (k, _) in args if k is None) > 1:
            raise ScadError("pass cylinder radius or diameter by name (r=, d=)", line)
        b = self.bind(args, ("h",), scope, line, extra=("r", "r1", "r2", "d", "d1", "d2", "center"))
        for k in ("h", "r", "r1", "r2", "d", "d1", "d2"):
            if b.get(k) is not None and not _fin(b[k]):
                raise ScadError("cylinder %s is %s" % (k, show(b[k])), line)
        h = 1.0 if b["h"] is None else b["h"]
        d0 = d1 = 2.0
        if b.get("r") is not None:
            d0 = d1 = 2.0 * b["r"]
        if b.get("r1") is not None:
            d0 = 2.0 * b["r1"]
        if b.get("r2") is not None:
            d1 = 2.0 * b["r2"]
        if b.get("d") is not None:
            d0 = d1 = b["d"]
        if b.get("d1") is not None:
            d0 = b["d1"]
        if b.get("d2") is not None:
            d1 = b["d2"]
        if h <= 0 or d0 < 0 or d1 < 0 or (d0 == 0 and d1 == 0):
            raise ScadError("cylinder has no volume (h=%g, d=%g)" % (h, max(d0, d1)), line)
        a, tx, ty, tz = xf
        z0 = tz - h / 2.0 if _truthy(b.get("center")) else tz
        self.prims.append({"kind": "cyl", "neg": neg, "line": line, "x": tx, "y": ty,
                           "z0": z0, "z1": z0 + h, "d0": d0, "d1": d1})

    # expressions
    def ev(self, e, scope):
        self.tick()
        k = e[0]
        if k == "num" or k == "str" or k == "bool":
            return e[1]
        if k == "undef":
            return None
        if k == "var":
            return scope.get(e[1])
        if k == "vec":
            out = []
            for item in e[1]:
                self.element(item, scope, out)
            return out
        if k == "range":
            a = self.ev(e[1], scope)
            step = 1.0 if e[2] is None else self.ev(e[2], scope)
            return Range(a, step, self.ev(e[3], scope))
        if k == "un":
            v = self.ev(e[2], scope)
            if e[1] == "!":
                return not _truthy(v)
            if e[1] == "+":
                return v
            return _neg(v)
        if k == "bin":
            op = e[1]
            if op == "&&":
                return _truthy(self.ev(e[2], scope)) and _truthy(self.ev(e[3], scope))
            if op == "||":
                return _truthy(self.ev(e[2], scope)) or _truthy(self.ev(e[3], scope))
            return _binary(op, self.ev(e[2], scope), self.ev(e[3], scope))
        if k == "tern":
            return self.ev(e[2] if _truthy(self.ev(e[1], scope)) else e[3], scope)
        if k == "index":
            base, idx = self.ev(e[1], scope), self.ev(e[2], scope)
            if isinstance(base, (list, str)) and _fin(idx) and idx == int(idx) \
                    and 0 <= int(idx) < len(base):
                return base[int(idx)]
            return None
        if k == "let":
            inner = Scope(scope)
            for (name, x) in e[1]:
                if name is not None:
                    inner.vars[name] = self.ev(x, inner)
            return self.ev(e[2], inner)
        if k == "check":
            vals = [self.ev(x, scope) for (_, x) in e[1]]
            if vals and not _truthy(vals[0]):
                raise ScadError("assert failed", e[3])
            return self.ev(e[2], scope)
        if k == "fnlit":
            return Function(e[1], e[2], scope)
        if k == "call":
            return self.call(e, scope)
        raise ScadError("a list comprehension keyword is used outside a list")

    def element(self, item, scope, out):
        k = item[0]
        if k == "lc_for":
            self.lc_loop(item[1], 0, item[2], scope, out, item[3])
        elif k == "lc_if":
            self.tick(item[4])
            if _truthy(self.ev(item[1], scope)):
                self.element(item[2], scope, out)
            elif item[3] is not None:
                self.element(item[3], scope, out)
        elif k == "lc_let":
            inner = Scope(scope)
            for (name, x) in item[1]:
                if name is not None:
                    inner.vars[name] = self.ev(x, inner)
            self.element(item[2], inner, out)
        elif k == "each":
            tmp = []
            self.element(item[1], scope, tmp)
            for v in tmp:
                if isinstance(v, Range):
                    out.extend(v.values(self.max_ops - self.ops))
                elif isinstance(v, list):
                    out.extend(v)
                else:
                    out.append(v)
            self.ops += len(out)
        else:
            out.append(self.ev(item, scope))

    def lc_loop(self, binds, k, body, scope, out, line):
        if k == len(binds):
            self.element(body, scope, out)
            return
        name, e = binds[k]
        for v in self.iterate(self.ev(e, scope), line):
            self.tick(line)
            inner = Scope(scope)
            inner.vars[name] = v
            self.lc_loop(binds, k + 1, body, inner, out, line)

    def call(self, e, scope):
        target, args, line = e[1], e[2], e[3]
        fn = None
        if target[0] == "var":
            name = target[1]
            local = scope.get(name)
            if isinstance(local, Function):
                fn = local
            elif name in self.functions:
                fn = self.functions[name]
            else:
                return self.builtin(name, [self.ev(x, scope) for (_, x) in args], line)
        else:
            fn = self.ev(target, scope)
            if not isinstance(fn, Function):
                raise ScadError("called something that is not a function", line)
        if self.depth >= MAX_DEPTH:
            raise ScadError("function calls nested more than %d deep" % MAX_DEPTH, line)
        inner = Scope(fn.scope)
        self.bind_params(fn.params, args, scope, inner, line)
        self.depth += 1
        try:
            return self.ev(fn.body, inner)
        finally:
            self.depth -= 1

    def builtin(self, name, v, line):
        try:
            if name in _MATH1:
                return float(_MATH1[name](v[0])) if len(v) == 1 and _num(v[0]) else None
            if name == "atan2":
                return math.degrees(math.atan2(v[0], v[1])) if len(v) == 2 and all(map(_num, v)) \
                    else None
            if name == "pow":
                return _binary("^", v[0], v[1]) if len(v) == 2 else None
            if name in ("min", "max"):
                items = v[0] if len(v) == 1 and isinstance(v[0], list) else v
                if not items or not all(_num(x) for x in items):
                    return None
                return min(items) if name == "min" else max(items)
            if name == "len":
                return float(len(v[0])) if v and isinstance(v[0], (list, str)) else None
            if name == "norm":
                if v and isinstance(v[0], list) and all(_num(x) for x in v[0]):
                    return math.sqrt(sum(x * x for x in v[0]))
                return None
            if name == "concat":
                out = []
                for x in v:
                    if isinstance(x, list):
                        out.extend(x)
                    else:
                        out.append(x)
                self.ops += len(out)
                return out
            if name == "is_undef":
                return bool(v) and v[0] is None
            if name == "is_num":
                return bool(v) and _num(v[0]) and not math.isnan(v[0])
            if name == "is_list":
                return bool(v) and isinstance(v[0], list)
            if name == "is_string":
                return bool(v) and isinstance(v[0], str)
            if name == "is_bool":
                return bool(v) and isinstance(v[0], bool)
            if name == "str":
                text = "".join(x if isinstance(x, str) else show(x) for x in v)
                self.ops += len(text)
                return text
        except (OverflowError, ValueError, IndexError):
            return float("nan")
        raise ScadError("function %s is not defined or is outside the supported subset" % name, line)


def _fin(v):
    return _num(v) and not math.isnan(v) and not math.isinf(v)


def _truthy(v):
    if v is None or v is False:
        return False
    if _num(v):
        return v != 0 and not math.isnan(v)
    if isinstance(v, (list, str)):
        return len(v) > 0
    return True


def _neg(v):
    if _num(v):
        return -v
    if isinstance(v, list):
        return [_neg(x) for x in v]
    return None


def _binary(op, a, b):
    if op == "==":
        return _same(a, b)
    if op == "!=":
        return not _same(a, b)
    if op in ("<", "<=", ">", ">="):
        if (_num(a) and _num(b)) or (isinstance(a, str) and isinstance(b, str)):
            return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
        return None
    if _num(a) and _num(b):
        try:
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                if b == 0:
                    return float("nan") if a == 0 or math.isnan(a) else math.copysign(float("inf"), a)
                return a / b
            if op == "%":
                return math.fmod(a, b) if b != 0 else float("nan")
            if op == "^":
                r = math.pow(a, b)
                return r if isinstance(r, float) else float("nan")
        except (OverflowError, ValueError):
            return float("nan")
    if isinstance(a, list) and isinstance(b, list):
        if op in ("+", "-"):
            return [_binary(op, x, y) for x, y in zip(a, b)]
        if op == "*" and len(a) == len(b) and all(_num(x) for x in a) and all(_num(x) for x in b):
            return float(sum(x * y for x, y in zip(a, b)))
        return None
    if isinstance(a, list) and _num(b) and op in ("*", "/"):
        return [_binary(op, x, b) for x in a]
    if _num(a) and isinstance(b, list) and op == "*":
        return [_binary(op, a, y) for y in b]
    return None


def _same(a, b):
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (Range, Function)) or isinstance(b, (Range, Function)):
        return a is b
    return type(a) == type(b) and a == b


def show(v):
    if v is None:
        return "undef"
    if isinstance(v, bool):
        return "true" if v else "false"
    if _num(v):
        return "%g" % v
    if isinstance(v, list):
        return "[" + ", ".join(show(x) for x in v) + "]"
    if isinstance(v, str):
        return '"%s"' % v
    return "<%s>" % type(v).__name__.lower()


def evaluate(src, max_ops=MAX_OPS):
    """Parse and run OpenSCAD source. Returns the list of primitives.

    Every failure, including runaway recursion, comes out as ScadError.
    """
    try:
        return Evaluator(max_ops).run(parse(src))
    except RecursionError:
        raise ScadError("code is nested too deeply")
    except (OverflowError, ZeroDivisionError, MemoryError) as exc:
        raise ScadError("arithmetic failed: %s" % type(exc).__name__)


# ---- plate reader ------------------------------------------------------------

EPS = 1e-6


def read_plate(src, max_ops=MAX_OPS):
    """Evaluate src and describe it as one plate with round through holes.

    Returns {"width", "height", "thickness", "holes": [(x, y, d)], "defects": [text],
    "flush_cuts": n}. Hole coordinates are relative to the plate's lower-left corner.
    A cut that does not go through the plate, a tapered cut, or a box cut is listed
    in defects and left out of holes. flush_cuts counts through holes whose end is
    exactly on a plate face, which renders with artifacts in OpenSCAD's preview.
    Raises ScadError when there is not exactly one solid box.
    """
    prims = evaluate(src, max_ops)
    solids = [p for p in prims if not p["neg"]]
    boxes = [p for p in solids if p["kind"] == "box"]
    if len(solids) != len(boxes):
        bad = [p for p in solids if p["kind"] != "box"][0]
        raise ScadError("a solid cylinder is outside the supported subset (plates only)",
                        bad["line"])
    if len(boxes) != 1:
        raise ScadError("expected exactly one solid plate, found %d" % len(boxes),
                        boxes[1]["line"] if len(boxes) > 1 else None)
    lo, hi = boxes[0]["lo"], boxes[0]["hi"]
    holes, defects, flush = [], [], 0
    for p in prims:
        if not p["neg"]:
            continue
        if p["kind"] != "cyl":
            defects.append("line %d: box cut is not a round hole" % p["line"])
        elif abs(p["d0"] - p["d1"]) > EPS:
            defects.append("line %d: tapered cut (d %g to %g)" % (p["line"], p["d0"], p["d1"]))
        elif p["z0"] > lo[2] + EPS or p["z1"] < hi[2] - EPS:
            if p["z1"] <= lo[2] + EPS or p["z0"] >= hi[2] - EPS:
                defects.append("line %d: cut misses the plate in z" % p["line"])
            else:
                defects.append("line %d: blind hole, cut spans z %g to %g but the plate spans "
                               "%g to %g" % (p["line"], p["z0"], p["z1"], lo[2], hi[2]))
        else:
            if abs(p["z0"] - lo[2]) <= EPS or abs(p["z1"] - hi[2]) <= EPS:
                flush += 1
            holes.append((p["x"] - lo[0], p["y"] - lo[1], p["d0"]))
    return {"width": hi[0] - lo[0], "height": hi[1] - lo[1], "thickness": hi[2] - lo[2],
            "origin": lo, "holes": holes, "defects": defects, "flush_cuts": flush}


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) != 2:
        print("usage: python scad.py <file.scad>")
        sys.exit(2)
    try:
        print(json.dumps(read_plate(open(sys.argv[1]).read()), indent=1))
    except ScadError as exc:
        print("ScadError: %s" % exc)
        sys.exit(1)
