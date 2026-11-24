"""Baseline agents for boltgym. Each run_* function plays one episode and returns
(success, steps_used)."""
import random

from boltgym import CLEARANCE, STANDARD_DIAMETERS, to_target


def run_random(env, seed, rng=None):
    """Adds as many holes as the source has, at random places with random standard sizes."""
    rng = rng or random.Random(seed)
    obs = env.reset(seed)
    w, h = obs["target"]["width"], obs["target"]["height"]
    for _ in obs["source"]["holes"]:
        env.step(("add", rng.uniform(0, w), rng.uniform(0, h), rng.choice(STANDARD_DIAMETERS)))
    _, _, _, info = env.step(("submit",))
    return info["success"], env.steps


def run_copy(env, seed):
    """Pastes the source coordinates and diameters unchanged. This is the mistake the
    task is designed to catch: it ignores the anchor corner and the thread callout."""
    obs = env.reset(seed)
    for (x, y, d) in obs["source"]["holes"]:
        env.step(("add", x, y, d))
    _, _, _, info = env.step(("submit",))
    return info["success"], env.steps


def run_scripted(env, seed):
    """Reads the anchor and the thread callout and does the geometry directly."""
    obs = env.reset(seed)
    src = (obs["source"]["width"], obs["source"]["height"])
    dst = (obs["target"]["width"], obs["target"]["height"])
    for hole in obs["source"]["holes"]:
        x, y, d = to_target(hole, obs["anchor"], src, dst)
        if obs["source"]["tapped"]:
            d = CLEARANCE[d]
        env.step(("add", x, y, d))
    _, _, _, info = env.step(("submit",))
    return info["success"], env.steps


def run_search(env, seed, first_step=32.0, last_step=0.5):
    """Reward-only search. It never reads the anchor or the thread callout.

    It pastes the source holes, then for each hole tries standard diameters and
    moves along x and y, keeping an edit only when the reward for it was positive
    and undoing it otherwise. With the sparse reward every edit returns 0, so
    nothing is ever kept and it ends where the copy agent ends.
    """
    obs = env.reset(seed)
    n = len(obs["source"]["holes"])
    for (x, y, d) in obs["source"]["holes"]:
        env.step(("add", x, y, d))

    state = {"done": False, "info": None}

    def attempt(action, undo):
        if state["done"] or env.steps + 3 > env.max_steps:
            return False  # keep one step back for the undo and one for submit
        _, r, done, info = env.step(action)
        state["done"], state["info"] = done, info
        if r > 1e-9:
            return True
        if not done:
            _, _, done, info = env.step(undo)
            state["done"], state["info"] = done, info
        return False

    for _sweep in range(2):
        for i in range(n):
            current = env.holes[i][2]
            for d in STANDARD_DIAMETERS:
                if d != current and attempt(("dia", i, d), ("dia", i, current)):
                    current = d
            for axis in (0, 1):
                step = first_step
                while step >= last_step and env.steps + 3 <= env.max_steps:
                    moved = False
                    for sign in (1.0, -1.0):
                        dx = sign * step if axis == 0 else 0.0
                        dy = sign * step if axis == 1 else 0.0
                        while attempt(("move", i, dx, dy), ("move", i, -dx, -dy)):
                            moved = True
                        if moved:
                            break
                    if not moved:
                        step /= 2.0
    if not state["done"]:
        _, _, _, info = env.step(("submit",))
        state["info"] = info
    return state["info"]["success"], env.steps


# ---- a small learned policy -------------------------------------------------

MACROS = [(fx, fy, rule) for fx in (0, 1) for fy in (0, 1) for rule in ("same", "clearance")]


def run_macro(env, seed, macro):
    """Apply one of 8 whole-pattern programs: shift x from the right edge or not,
    shift y from the top edge or not, keep diameters or look up clearance sizes."""
    fx, fy, rule = macro
    obs = env.reset(seed)
    sw, sh = obs["source"]["width"], obs["source"]["height"]
    tw, th = obs["target"]["width"], obs["target"]["height"]
    for (x, y, d) in obs["source"]["holes"]:
        if fx:
            x = tw - (sw - x)
        if fy:
            y = th - (sh - y)
        if rule == "clearance":
            d = CLEARANCE.get(d, d)
        env.step(("add", x, y, d))
    _, _, _, info = env.step(("submit",))
    return info["success"], (obs["skill"], obs["anchor"])


def peek_context(env, seed):
    obs = env.reset(seed)
    return (obs["skill"], obs["anchor"])


class Bandit:
    """Epsilon-greedy contextual bandit over MACROS. Context is (skill, anchor).
    It starts with no idea what any anchor word means and learns from 0/1 success."""

    def __init__(self, epsilon=0.1, seed=0):
        self.eps, self.rng = epsilon, random.Random(seed)
        self.n, self.q = {}, {}

    def choose(self, ctx, explore=True):
        if explore and self.rng.random() < self.eps:
            return self.rng.randrange(len(MACROS))
        vals = [self.q.get((ctx, a), 0.0) for a in range(len(MACROS))]
        best = max(vals)
        return self.rng.choice([a for a, v in enumerate(vals) if v == best])

    def update(self, ctx, a, reward):
        k = (ctx, a)
        self.n[k] = self.n.get(k, 0) + 1
        self.q[k] = self.q.get(k, 0.0) + (reward - self.q.get(k, 0.0)) / self.n[k]


def train_bandit(env, episodes, seed=0, first_seed=100000, block=100):
    """Returns (bandit, success rate per block of episodes)."""
    b = Bandit(seed=seed)
    curve, wins = [], 0
    for ep in range(episodes):
        s = first_seed + ep
        ctx = peek_context(env, s)
        a = b.choose(ctx)
        ok, _ = run_macro(env, s, MACROS[a])
        b.update(ctx, a, 1.0 if ok else 0.0)
        wins += ok
        if (ep + 1) % block == 0:
            curve.append(wins / block)
            wins = 0
    return b, curve
