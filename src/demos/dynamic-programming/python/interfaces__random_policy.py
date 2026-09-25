import numpy as np
from interfaces.policy import Policy

## This is a random policy that takes a static probability distribution over all actions.
## If none is specified, all actions can be taken randomly.
class RandomPolicy(Policy):
    def __init__(self, nA, p=None):
        self.p = p if p is not None else np.array([1/nA]*nA)

    def action_prob(self, state, action=None):
        return self.p[action]

    def action(self, state):
        return np.random.choice(len(self.p), p=self.p)
    
    def __str__(self) -> str:
        return f"RandomPolicy, p={self.p}"