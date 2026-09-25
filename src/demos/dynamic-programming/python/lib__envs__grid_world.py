import gymnasium as gym
import numpy as np

from lib.envs.wrapped_gridworld import Actions

class GridWorld(gym.Env):
    """
    Helper class for modeling a simple grid world environment.

    This can be used as a base class for custom grid world environments,
    or you can use the WrappedGridWorld class to wrap an existing grid-based
    environment from gymnasium.

    Also contains utilities for visualizing:

    """

    def __init__(self, rows: int, columns: int, terminal_state: list[int]):
        """
        Create a new grid world environment.

        Parameters:
            rows(int): The rows of the grid world
            columns(int): The columns of the grid world
            terminal_state(list[int]): A list of terminal states in the grid world
        """

        self.rows: int = rows
        self.columns: int = columns

        self.observation_space = gym.spaces.Discrete(rows * columns)
        self.action_space = gym.spaces.Discrete(4)
        self.terminal_states = terminal_state

        self.state = None

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to a random initial state.

        Returns:
            int: The initial state
        """
        super().reset(seed=seed)
        # get a list of valid non terminal states
        valid_states = [s for s in range(self.observation_space.n) if s not in self.terminal_states]
        self.state = self.np_random.choice(valid_states)
        return self.state, {}
    
    def step(self, action):
        """
        Take a step in the environment.

        Parameters:
            action(int): The action to take

        Returns:
            tuple(int, float, bool, dict): The next state, reward, done, info
        """
        assert action in range(self.action_space.n), "Invalid Action"

        # grab the transition probabilities [(prob, next_state, reward, done)]
        transitions: list = self.P[self.state, action]
        
        # sample an index from transitions based on the prob
        probs, _, _, _ = zip(*transitions)
        i = self.np_random.choice(np.arange(len(transitions)), p=probs)
        prob, next_state, reward, done = transitions[i]
        
        self.state = next_state
        return self.state, reward, done, False, {}

    def _deterministic_transition(self, state, action):
        """
        Determine the next state given a state and action.

        Parameters:
            state(int): The current state
            action(int): The action to take

        Returns:
            int: The next state. Note it could be invalid! Use _is_valid() to check.
        """
        row, column = self._state_to_position(state)

        if action == Actions.LEFT:
            column -= 1
        elif action == Actions.RIGHT:
            column += 1
        elif action == Actions.UP:
            row -= 1
        elif action == Actions.DOWN:
            row += 1
        
        # convert back to state index
        return row, column

    def _is_valid(self, row, column):
        """
        Check if the state is within the grid world.
        """
        return 0 <= row and row < self.rows and 0 <= column and column < self.columns
    
    def _get_state(self, row, column):
        """
        Convert a row and column position to a state index.
        """
        return row * self.columns + column

    def _state_to_position(self, state):
        """
        Convert a state index to a row and column position.

        Parameters:
            state(int): The state index

        Returns:
            tuple(int, int): The row and column position
        """
        return divmod(state, self.columns)

    def render(self):
        pass
