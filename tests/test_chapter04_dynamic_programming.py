import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTER_DIR = ROOT / "Chapter 04 - Dynamic Programming"
SRC_DIR = CHAPTER_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CHAPTER_DIR))

from grid_world import GridWorld
from iterative_policy_evaluation import (
    equiprobable_random_policy,
    iterative_policy_evaluation,
)


class GridWorldExample41Tests(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld()

    def test_nonterminal_state_space_and_actions_match_book_example(self):
        self.assertEqual(self.env.get_state_space(), list(range(1, 15)))
        self.assertEqual(self.env.get_actions(), ("up", "down", "right", "left"))

    def test_deterministic_transitions_match_book_examples(self):
        self.assertEqual(self.env.step(5, "right"), (6, -1.0, False))
        self.assertEqual(self.env.step(7, "right"), (7, -1.0, False))
        possible_next_states = {transition.next_state for transition in self.env.transitions(5, "right")}
        self.assertNotIn(10, possible_next_states)

    def test_terminal_cells_are_one_formal_terminal_state(self):
        self.assertEqual(self.env.step(1, "left"), (0, -1.0, True))
        self.assertEqual(self.env.step(4, "up"), (0, -1.0, True))
        self.assertEqual(self.env.step(14, "right"), (0, -1.0, True))
        self.assertEqual(self.env.step(11, "down"), (0, -1.0, True))


class IterativePolicyEvaluationTests(unittest.TestCase):
    def test_first_synchronous_sweep_sets_each_nonterminal_value_to_step_reward(self):
        env = GridWorld()
        policy = equiprobable_random_policy(env)

        values = iterative_policy_evaluation(env, policy, gamma=1.0, iterations=1)

        self.assertEqual(values[0], 0.0)
        for state in env.get_state_space():
            self.assertEqual(values[state], -1.0)

    def test_equiprobable_random_policy_converges_to_figure_4_1_values(self):
        env = GridWorld()
        policy = equiprobable_random_policy(env)

        values = iterative_policy_evaluation(env, policy, gamma=1.0, theta=1e-10, max_iterations=10_000)

        expected_grid = [
            [0.0, -14.0, -20.0, -22.0],
            [-14.0, -18.0, -20.0, -20.0],
            [-20.0, -20.0, -18.0, -14.0],
            [-22.0, -20.0, -14.0, 0.0],
        ]
        actual_grid = env.values_as_grid(values)
        for actual_row, expected_row in zip(actual_grid, expected_grid):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=6)


if __name__ == "__main__":
    unittest.main()
