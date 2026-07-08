#!/usr/bin/env python
"""Iterative policy evaluation for Sutton & Barto Example 4.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

try:
    from grid_world import GridWorld
except ImportError:  # pragma: no cover - supports running from outside chapter dir
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from grid_world import GridWorld


Policy = Callable[[int, str], float]


def equiprobable_random_policy(env: GridWorld) -> Policy:
    """Return pi(a|s)=1/4 for each action in every nonterminal state."""
    action_probability = 1.0 / len(env.get_actions())

    def policy(state: int, action: str) -> float:
        if state == env.terminal_state:
            return 0.0
        if action not in env.get_actions():
            return 0.0
        return action_probability

    return policy


def iterative_policy_evaluation(
    env: GridWorld,
    policy: Policy,
    *,
    gamma: float = 1.0,
    theta: float = 1e-4,
    max_iterations: int = 1_000,
    iterations: Optional[int] = None,
    return_history: bool = False,
) -> Dict[int, float] | list[Dict[int, float]]:
    """Evaluate a policy with synchronous dynamic-programming sweeps.

    The update is
        v_{k+1}(s) = sum_a pi(a|s) sum_{s',r} p(s',r|s,a) [r + gamma v_k(s')]

    By default the function stops when the maximum state-value change is below
    theta. Pass iterations=N to run exactly N sweeps, which is useful for
    reproducing the sequence v_0, v_1, ... in Figure 4.1.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")
    if iterations is not None and iterations < 0:
        raise ValueError("iterations must be non-negative.")

    states = env.get_state_space()
    values: Dict[int, float] = {env.terminal_state: 0.0, **{state: 0.0 for state in states}}
    history = [values.copy()]
    limit = iterations if iterations is not None else max_iterations

    for _ in range(limit):
        new_values = values.copy()
        delta = 0.0

        for state in states:
            state_value = 0.0
            for action in env.get_actions():
                action_probability = policy(state, action)
                if action_probability == 0.0:
                    continue

                action_value = 0.0
                for transition in env.transitions(state, action):
                    action_value += transition.probability * (
                        transition.reward + gamma * values[transition.next_state]
                    )
                state_value += action_probability * action_value

            new_values[state] = state_value
            delta = max(delta, abs(state_value - values[state]))

        values = new_values
        if return_history:
            history.append(values.copy())

        if iterations is None and delta < theta:
            break

    if return_history:
        return history
    return values


def evaluate_equiprobable_random_policy(
    *,
    theta: float = 1e-4,
    gamma: float = 1.0,
    max_iterations: int = 1_000,
) -> Dict[int, float]:
    """Convenience wrapper for Example 4.1's equiprobable random policy."""
    env = GridWorld()
    return iterative_policy_evaluation(
        env,
        equiprobable_random_policy(env),
        theta=theta,
        gamma=gamma,
        max_iterations=max_iterations,
    )


def format_grid(grid: list[list[float]]) -> str:
    """Format a value grid for terminal output."""
    return "\n".join(" ".join(f"{value:7.2f}" for value in row) for row in grid)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run iterative policy evaluation for Sutton & Barto Example 4.1."
    )
    parser.add_argument("--theta", type=float, default=1e-4, help="Convergence threshold.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--max-iterations", type=int, default=1_000, help="Maximum sweeps.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Run exactly this many synchronous sweeps instead of converging.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional JSON path for the final value function and grid.",
    )
    args = parser.parse_args()

    env = GridWorld()
    values = iterative_policy_evaluation(
        env,
        equiprobable_random_policy(env),
        gamma=args.gamma,
        theta=args.theta,
        max_iterations=args.max_iterations,
        iterations=args.iterations,
    )
    assert isinstance(values, dict)
    grid = env.values_as_grid(values)

    print(format_grid(grid))

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        payload: Mapping[str, object] = {
            "values": {str(state): value for state, value in values.items()},
            "grid": grid,
        }
        args.output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
