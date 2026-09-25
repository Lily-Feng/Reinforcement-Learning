import numpy as np

from interfaces.policy import Policy
from lib.envs.grid_world import GridWorld, Actions

class GridWorld2x2(GridWorld):
    """
    A simple 2x2 grid world environment.

    The agent starts in the upper left corner and must navigate to the lower right corner.

    The agent receives a reward of -1 on all steps, until it reaches the goal state in the lower right corner.
    There's no reward for reaching the goal state, the episode just ends.
    """
    def __init__(self):
        super().__init__(
            rows = 2,
            columns = 2,
            terminal_state = [3] # the lower right corner
        )

        self.P = self._build_transitions()

    def _build_transitions(self):
        P = np.ndarray((self.observation_space.n, self.action_space.n), dtype=object)

        ## Setup the translation matrix, if you go off the grid, you stay in the same state
        ## You always get a -1 reward.
        ## You can't leave the terminal state, and you get reward of 0

        for s in range(self.observation_space.n):
            for a in range(self.action_space.n):
                if s in self.terminal_states:
                    P[s, a] = [(1.0, s, 0, True)]
                    continue

                # Determinstically determine the next state
                n_row, n_col = self._deterministic_transition(s, a)

                if self._is_valid(n_row, n_col):
                    next_state = self._get_state(n_row, n_col)
                    P[s, a] = [(1.0, next_state, -1, next_state in self.terminal_states)]
                
                else:
                    # ran into the wall so loop us back
                    P[s, a] = [(1.0, s, -1, False)]

        return P
        
# Off-policy evaluation test with optimal policy
class GridWorld2x2OptimalPolicy(Policy):
    def action_prob(self, state, action):
        if state == 0:
            return 0.5 if action in [Actions.RIGHT, Actions.DOWN] else 0.0
        return 1.0 if self.action(state) == action else 0.0

    def action(self, state):
        if state == 0:
            return np.random.choice([Actions.RIGHT, Actions.DOWN])

        return [Actions.RIGHT, Actions.DOWN, Actions.RIGHT, Actions.LEFT][state]
    
    def __str__(self) -> str:
        return "GridWorld2x2OptimalPolicy"