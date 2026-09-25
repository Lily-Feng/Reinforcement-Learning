import gymnasium as gym
import numpy as np
from tabulate import tabulate
from typing import override

from interfaces.policy import Policy

class Actions:
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

## Special class to handle non-grid MDP's
class WrappedMDP(gym.Wrapper):
    TERMINAL = -1

    def __init__(self, env: gym.Env, shape: np.ndarray, gamma: float):
        super().__init__(env)
        self.gamma = gamma
        self.shape = shape
        self._active_env = env

        # calculate terminal states by looking at the transition probabilities
        self.terminal_states = set()
        for s in range(self.observation_space.n):
            for a in range(self.action_space.n):
                for _, next_state, _, done in self.P[s][a]:
                    if done:
                        self.terminal_states.add(next_state)

    @property
    def observation_space(self) -> gym.spaces.Discrete:
        """
        Returns the observation space of the environment.
        """
        return self._active_env.observation_space
    
    @property
    def action_space(self) -> gym.spaces.Discrete:
        """
        Returns the action space of the environment.
        """
        return self._active_env.action_space
    
    @property
    def P(self):
        """
        Returns the transition probabilities of the environment.
        """
        return self._active_env.unwrapped.P

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to a random initial state.

        Returns:
            int: The initial state
        """
        return self._active_env.reset(seed=seed, options=options)
    
    def step(self, action):
        """
        Take a step in the environment.

        Parameters:
            action(int): The action to take

        Returns:
            tuple(int, float, bool, dict): The next state, reward, done, info
        """
        return self._active_env.step(action)

    def visualize(self, title: str = "States"):
        """
        Visualize the environment as a table of states.

        This displays the index of each state in the grid world.

        Parameters:
            title(str): Optional, Title for the visualization

        Returns:
            str: A string representation of the grid world
        """
        return self._create_grid(np.arange(self.observation_space.n), title)

    def visualize_policy(self, pi: Policy, title="Policy"):
        """
        Visualizes a policy within the gridworld.

        This works best for deterministic policies.

        Parameters:
            pi(Policy): The policy to visualize
            title(str): Optional, Title for the visualization

        Returns:
            str: A string representation of the policy
        """
        data = np.empty(self.observation_space.n, dtype=object)
        
        for s in range(self.observation_space.n):
            data[s] = self._action_char(pi.action(s)) if s not in self.terminal_states else self._action_char(WrappedGridWorld.TERMINAL)

        return self._create_grid(data, title)
    
    def visualize_v(self, V: np.ndarray, title: str = "V(s)"):
        """
        Visualize a value function within the grid world.

        Parameters:
            V(np.ndarray): The value function to visualize. Shape: [nS]
            title(str): Optional, Title for the visualization

        Returns:
            str: A string representation of the value function
        """
        return self._create_grid(V, title)

    def visualize_q(self, Q, title="Q(s, a)"):
        """
        Visualize a state-action value function within the grid world.

        Returns a grid with the Q values for each state-action pair.

        Example:
                5.0 
                 ↑
            3.0 ← → 4.0 
                 ↓
                2.0

        Parameters:
            Q(np.ndarray): The state-action value function to visualize. Shape: [nS, nA]
            title(str): Optional, Title for the visualization

        Returns:
            str: A string representation of the state-action value function
        """        
        visual_q = []
        for s in range(self.observation_space.n):
            if s in self.terminal_states:
                visual_q.append('★         ★\n★   ★\n✪\n★   ★\n★        ★')
            else:
                # Combine the formatted lines into a single string
                visual = '\n'.join([
                    f"{Q[s][0]:.2f} ← → {Q[s][1]:.2f}",
                ])

                visual_q.append(visual)
        
        data = [visual_q[i:i + 2] for i in range(0, len(visual_q), 2)]  # Adjust based on grid size

        return self._create_grid(np.array(data), title)

    def _create_grid(self, data: np.ndarray | list[str], title: str):
        """
        Helper function to add a title to a table.

        Parameters:
            data(np.ndarray | list[str]): The data to visualize. Length: nS
            title(str): The title for the visualization

        Returns:
            str: A string representation of the table
        """
        table = tabulate(data.reshape(self.shape), tablefmt="simple_grid", stralign="center")
        title_line = title.center(len(table.splitlines()[0])) + "\n"
        return title_line + table
    
    def __str__(self):
        return f"{self._active_env.__class__.__name__}\n{self.visualize()}\nTerminal States: {self.terminal_states}"

    def _action_char(self, action: int) -> int:
        if action == WrappedGridWorld.TERMINAL:
            return '✪'

        ACTIONS_MAP = {
            0: '→',
            1: '←'
        }

        return ACTIONS_MAP[action]

    
class WrappedGridWorld(WrappedMDP):
    """
    Wrapper for either custom grid worlds, or existing grid world environments from gymnasium.

    Adds extra visualization capabilities to the environment.
        - The environment
        - A policy
        - Value (V) function
        - State-Action (Q) function
    """
    def __init__(
            self,
            env: gym.Env,
            rows: int,
            columns: int,
            gamma: float,
            action_remap: list[int] = [Actions.LEFT, Actions.UP, Actions.RIGHT, Actions.DOWN]
        ):
        """
        Create a new wrapped grid world environment.

        Parameters:
            env(gym.Env): The grid world environment
            rows(int): The rows of the grid world
            columns(int): The columns of the grid world
            gamma(float): The discount factor that works well with the environment
            action_remap(list[int]): Optional, Remap the actions in the environment to our Actions class
        """
        super().__init__(env, np.array((rows, columns)), gamma)
        self.rows = rows
        self.columns = columns
        self.action_remap = action_remap
        self.reverse_remap = {v: k for k, v in enumerate(action_remap)}
            
    @override
    def _action_char(self, action: int) -> int:
        if action == WrappedGridWorld.TERMINAL:
            return '✪'

        ACTIONS_MAP = {
            Actions.LEFT: '←',
            Actions.RIGHT: '→',
            Actions.UP: '↑',
            Actions.DOWN: '↓',
            4: 'PU',
            5: 'DO',
        }

        # Say the env action is DOWN=1
        # Our DOWN=3
        # The remap is    0=>0, 1=>3, 2=>2, 3=>1
        # The reverse is  0=>0, 3=>1, 2=>2, 1=>3

        # The policy is going to give us a 1. We need to convert it to 3
        # We use the regular remap to get 3
        if action >= len(self.action_remap):
            return ACTIONS_MAP[action]
        return ACTIONS_MAP[self.action_remap[action]]
    
    def _env_action(self, action: Actions) -> int:
        # in this mode, we know what direction we care about, but we want to know the action
        # that the environment expects. If we have DOWN=3, we use the reverse remap to get 1
        if action >= len(self.action_remap):
            return action
        return self.reverse_remap[action]

    @override
    def visualize_q(self, Q, title="Q(s, a)"):
        """
        Visualize a state-action value function within the grid world.

        Returns a grid with the Q values for each state-action pair.

        Example:
                5.0 
                 ↑
            3.0 ← → 4.0 
                 ↓
                2.0

        Parameters:
            Q(np.ndarray): The state-action value function to visualize. Shape: [nS, nA]
            title(str): Optional, Title for the visualization

        Returns:
            str: A string representation of the state-action value function
        """        
        visual_q = []
        for s in range(self.observation_space.n):
            if s in self.terminal_states:
                visual_q.append('★         ★\n★   ★\n✪\n★   ★\n★        ★')
            else:
                # Combine the formatted lines into a single string
                visual = '\n'.join([
                    f"{Q[s][self._env_action(Actions.UP)]:.2f}",
                    f"{'↑':}",
                    f"{Q[s][self._env_action(Actions.LEFT)]:.2f} ← → {Q[s][self._env_action(Actions.RIGHT)]:.2f}",
                    f"{'↓':}",
                    f"{Q[s][self._env_action(Actions.DOWN)]:.2f}"
                ])

                visual_q.append(visual)
        
        data = [visual_q[i:i + 2] for i in range(0, len(visual_q), 2)]  # Adjust based on grid size

        return self._create_grid(np.array(data), title)
    
