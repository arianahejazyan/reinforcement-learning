from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union


Action = Union[str, int]


@dataclass(frozen=True)
class Transition:
    probability: float
    next_state: int
    reward: float
    done: bool


class GridWorld:
    """Sutton & Barto Example 4.1 grid world.

    The 4x4 diagram contains two shaded terminal cells, but they are one formal
    terminal state with value 0. The nonterminal states are labelled 1..14 in
    row-major order:

        T   1   2   3
        4   5   6   7
        8   9  10  11
       12  13  14   T

    Actions are deterministic. Attempts to move off the grid leave the state
    unchanged. Every transition from a nonterminal state has reward -1, matching
    the values in Figure 4.1 (the negative expected number of steps to terminate).
    """

    ACTIONS: Tuple[str, ...] = ("up", "down", "right", "left")
    ACTION_DELTAS: Dict[str, Tuple[int, int]] = {
        "up": (-1, 0),
        "down": (1, 0),
        "right": (0, 1),
        "left": (0, -1),
    }

    def __init__(self, rows: int = 4, cols: int = 4, step_reward: float = -1.0):
        if rows != 4 or cols != 4:
            raise ValueError("Example 4.1 uses a fixed 4x4 grid.")
        self.rows = rows
        self.cols = cols
        self.W = cols
        self.H = rows
        self.step_reward = float(step_reward)
        self.terminal_state = 0
        self.terminal_cells = ((0, 0), (rows - 1, cols - 1))

        # Compatibility with simple Gym-like code in this repository.
        self.observation_space = SimpleNamespace(n=rows * cols)
        self.action_space = SimpleNamespace(n=len(self.ACTIONS))
        self.P = self._build_transition_table()

    def get_state_space(self) -> List[int]:
        """Return the nonterminal states S = {1, 2, ..., 14}."""
        return list(range(1, self.rows * self.cols - 1))

    def get_actions(self) -> Tuple[str, ...]:
        """Return the four actions in the order used by the book."""
        return self.ACTIONS

    def _normalize_action(self, action: Action) -> str:
        if isinstance(action, int):
            try:
                return self.ACTIONS[action]
            except IndexError as exc:
                raise ValueError(f"Unknown action index: {action}") from exc
        if action not in self.ACTION_DELTAS:
            raise ValueError(f"Unknown action: {action!r}")
        return action

    def state_to_position(self, state: int) -> Tuple[int, int]:
        """Convert a book state label to its (row, col) grid position."""
        if state == self.terminal_state:
            return self.terminal_cells[0]
        if state not in self.get_state_space():
            raise ValueError(f"State must be terminal 0 or nonterminal 1..14, got {state}.")
        return divmod(state, self.cols)

    def position_to_state(self, row: int, col: int) -> int:
        """Convert a grid position to a book state label.

        Both shaded terminal cells map to the same formal terminal state 0.
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position {(row, col)} is outside the grid.")
        if (row, col) in self.terminal_cells:
            return self.terminal_state
        return row * self.cols + col

    def step(self, state: int, action: Action) -> Tuple[int, float, bool]:
        """Take a deterministic transition from state using action.

        Returns (next_state, reward, done). Calling step from the formal terminal
        state is allowed and keeps the agent in the terminal state with zero
        reward.
        """
        if state == self.terminal_state:
            return self.terminal_state, 0.0, True
        if state not in self.get_state_space():
            raise ValueError(f"State must be terminal 0 or nonterminal 1..14, got {state}.")

        action_name = self._normalize_action(action)
        row, col = self.state_to_position(state)
        d_row, d_col = self.ACTION_DELTAS[action_name]
        next_row, next_col = row + d_row, col + d_col

        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            next_state = state
        else:
            next_state = self.position_to_state(next_row, next_col)

        return next_state, self.step_reward, next_state == self.terminal_state

    def transitions(self, state: int, action: Action) -> Tuple[Transition, ...]:
        """Return p(s', r | s, a) as a one-element deterministic transition list."""
        next_state, reward, done = self.step(state, action)
        return (Transition(1.0, next_state, reward, done),)

    def values_as_grid(self, values: Dict[int, float]) -> List[List[float]]:
        """Render a value mapping as a 4x4 grid with both terminal cells as 0."""
        grid: List[List[float]] = []
        for row in range(self.rows):
            grid_row = []
            for col in range(self.cols):
                state = self.position_to_state(row, col)
                grid_row.append(float(values.get(state, 0.0)))
            grid.append(grid_row)
        return grid

    def _build_transition_table(self) -> Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]:
        table: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = {}
        for state in range(self.rows * self.cols):
            book_state = 0 if state in (0, self.rows * self.cols - 1) else state
            table[state] = {}
            for action_index, action in enumerate(self.ACTIONS):
                next_state, reward, done = self.step(book_state, action)
                table[state][action_index] = [(1.0, next_state, reward, done)]
        return table
