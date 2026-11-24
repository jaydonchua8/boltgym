import itertools
import random
import unittest

import agents
from boltgym import (BoltEnv, CLEARANCE, assign, assign_greedy, make_task, rule_violations,
                     to_scad, to_target, verify)


class TestAssign(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = random.Random(1)
        for _ in range(200):
            n = rng.randint(1, 6)
            c = [[rng.random() for _ in range(n)] for _ in range(n)]
            got = sum(c[i][j] for i, j in enumerate(assign(c)))
            best = min(sum(c[i][p[i]] for i in range(n))
                       for p in itertools.permutations(range(n)))
            self.assertAlmostEqual(got, best, places=9)

    def test_is_permutation(self):
        c = [[3, 1, 2], [1, 3, 2], [2, 2, 1]]
        self.assertEqual(sorted(assign(c)), [0, 1, 2])
        self.assertEqual(sorted(assign_greedy(c)), [0, 1, 2])


class TestVerify(unittest.TestCase):
    exp = [(20.0, 20.0, 4.5), (60.0, 20.0, 4.5)]

    def test_perfect_any_order(self):
        r = verify(list(reversed(self.exp)), self.exp, 100, 80)
        self.assertTrue(r["success"])
        self.assertAlmostEqual(r["score"], 1.0)

    def test_empty(self):
        r = verify([], self.exp, 100, 80)
        self.assertFalse(r["success"])
        self.assertEqual(r["score"], 0.0)

    def test_small_error_fails_but_scores_high(self):
        r = verify([(20.2, 20.0, 4.5), (60.0, 20.0, 4.5)], self.exp, 100, 80)
        self.assertFalse(r["success"])
        self.assertGreater(r["score"], 0.9)

    def test_wrong_diameter_fails(self):
        r = verify([(20.0, 20.0, 4.0), (60.0, 20.0, 4.5)], self.exp, 100, 80)
        self.assertFalse(r["success"])

    def test_extra_hole_fails_and_costs(self):
        r = verify(self.exp + [(40.0, 50.0, 4.5)], self.exp, 100, 80)
        self.assertFalse(r["success"])
        self.assertEqual(r["extras"], 1)
        self.assertAlmostEqual(r["score"], 0.9)

    def test_closer_is_better(self):
        scores = [verify([(20.0 + off, 20.0, 4.5), (60.0, 20.0, 4.5)], self.exp, 200, 80)["score"]
                  for off in (64, 32, 16, 8, 4, 2, 1, 0.5, 0)]
        self.assertEqual(scores, sorted(scores))

    def test_rules(self):
        self.assertEqual(rule_violations([(2.0, 20.0, 4.5)], 100, 80), 1)
        self.assertEqual(rule_violations([(20.0, 20.0, 4.5), (24.0, 20.0, 4.5)], 100, 80), 1)
        self.assertEqual(rule_violations(self.exp, 100, 80), 0)


class TestTasks(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(make_task(7), make_task(7))

    def test_anchor_math(self):
        h = (10.0, 15.0, 4.5)
        self.assertEqual(to_target(h, "bottom-left", (100, 80), (120, 60)), (10.0, 15.0, 4.5))
        self.assertEqual(to_target(h, "top-right", (100, 80), (120, 60)), (30.0, -5.0, 4.5))

    def test_generated_tasks_are_legal(self):
        for seed in range(300):
            t = make_task(seed)
            self.assertEqual(rule_violations(t["expected"], t["target"]["width"],
                                             t["target"]["height"]), 0)
            self.assertEqual(len(t["expected"]), len(t["source"]["holes"]))
            self.assertNotEqual(t["source"]["width"], t["target"]["width"])
            if t["skill"] == "clearance":
                self.assertEqual(t["expected"][0][2], CLEARANCE[t["source"]["holes"][0][2]])


class TestEnv(unittest.TestCase):
    def test_scripted_solves_everything(self):
        env = BoltEnv()
        for seed in range(300):
            ok, _ = agents.run_scripted(env, seed)
            self.assertTrue(ok, seed)

    def test_dense_rewards_sum_to_score(self):
        env = BoltEnv(reward="dense")
        env.reset(3)
        total = 0.0
        for a in [("add", 30, 30, 4.5), ("move", 0, 5, 0), ("dia", 0, 5.5), ("add", 50, 50, 4.5)]:
            _, r, _, info = env.step(a)
            total += r
        self.assertAlmostEqual(total, info["score"])

    def test_sparse_is_silent_until_submit(self):
        env = BoltEnv(reward="sparse")
        obs = env.reset(4)
        for h in obs["source"]["holes"]:
            _, r, done, _ = env.step(("add",) + tuple(h))
            self.assertEqual(r, 0.0)
            self.assertFalse(done)

    def test_invalid_action_changes_nothing(self):
        env = BoltEnv()
        env.reset(5)
        _, r, done, info = env.step(("move", 3, 1, 1))
        self.assertTrue(info["invalid"])
        self.assertEqual(r, 0.0)
        self.assertEqual(env.holes, [])
        _, _, _, info = env.step(("explode",))
        self.assertTrue(info["invalid"])

    def test_truncation(self):
        env = BoltEnv(max_steps=3)
        env.reset(6)
        for _ in range(3):
            _, _, done, info = env.step(("add", 30, 30, 4.5))
        self.assertTrue(done)
        self.assertTrue(info["truncated"])
        self.assertRaises(RuntimeError, env.step, ("submit",))

    def test_copy_agent_fails_when_anchor_is_not_bottom_left(self):
        env = BoltEnv(skill="transfer")
        for seed in range(100):
            ok, _ = agents.run_copy(env, seed)
            self.assertEqual(ok, env.task["anchor"] == "bottom-left", seed)

    def test_scad(self):
        s = to_scad(100, 80, [(20, 20, 4.5)])
        self.assertIn("cube([100, 80, 6]);", s)
        self.assertIn("translate([20, 20, -1]) cylinder(d = 4.5, h = 8);", s)
        self.assertEqual(s.count("{"), s.count("}"))


if __name__ == "__main__":
    unittest.main()
