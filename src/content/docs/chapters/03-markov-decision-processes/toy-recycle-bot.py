import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RecyclingRobotEnv(gym.Env):
    """
    Recycling Robot Environment based on Sutton & Barto Example 3.3
    States: 0 (High), 1 (Low)
    Actions: 0 (Search), 1 (Wait), 2 (Recharge)
    """
    def __init__(self, alpha=0.8, beta=0.8, r_search=5, r_wait=1):
        super().__init__()
        
        # Hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.r_search = r_search
        self.r_wait = r_wait
        
        # Observation Space: 0 for 'high', 1 for 'low'
        self.observation_space = spaces.Discrete(2)
        
        # Action Space: 0 for 'search', 1 for 'wait', 2 for 'recharge'
        self.action_space = spaces.Discrete(3)
        
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Always start with high energy
        self.state = 0 
        return self.state, {}

    def step(self, action):
        reward = 0
        
        if self.state == 0:  # Current state is HIGH
            if action == 0:  # SEARCH
                if self.np_random.random() < self.alpha:
                    self.state = 0 # Stays high
                else:
                    self.state = 1 # Drops to low
                reward = self.r_search
            
            elif action == 1:  # WAIT
                self.state = 0
                reward = self.r_wait
            
            elif action == 2:  # RECHARGE (Foolish when high)
                # Sutton's book ignores this action for HIGH state, 
                # so we just keep state HIGH and give 0 reward
                self.state = 0
                reward = 0

        elif self.state == 1:  # Current state is LOW
            if action == 0:  # SEARCH
                if self.np_random.random() < self.beta:
                    self.state = 1 # Stays low
                    reward = self.r_search
                else:
                    self.state = 0 # Depleted and rescued (back to high)
                    reward = -3    # Penalty for rescue
            
            elif action == 1:  # WAIT
                self.state = 1
                reward = self.r_wait
            
            elif action == 2:  # RECHARGE
                self.state = 0
                reward = 0
                
        # This is a continuing task, so it never naturally terminates
        terminated = False 
        truncated = False
        
        return self.state, reward, terminated, truncated, {}

    def render(self):
        state_name = "HIGH" if self.state == 0 else "LOW"
        print(f"Current Battery State: {state_name}")


# ==========================================
# How to run and test it
# ==========================================
if __name__ == "__main__":
    # 1. Initialize the environment
    env = RecyclingRobotEnv(alpha=0.8, beta=0.8, r_search=5, r_wait=1)

    # 2. Reset to starting state
    state, info = env.reset()
    assert state == 0, "the robot must always start with a HIGH battery"

    # These assertions pin down the deterministic parts of Example 3.3's
    # dynamics table, so a refactor that breaks the transition/reward rules
    # fails here instead of only showing up as an off-looking demo run.
    for s in (0, 1):  # HIGH, LOW
        env.state = s
        next_state, reward, *_ = env.step(1)  # WAIT
        assert next_state == s, "WAIT never changes the battery state"
        assert reward == env.r_wait, "WAIT always pays r_wait"

    env.state = 0
    next_state, reward, *_ = env.step(2)  # RECHARGE while HIGH
    assert next_state == 0 and reward == 0, "RECHARGE while HIGH is a no-op"

    env.state = 1
    next_state, reward, *_ = env.step(2)  # RECHARGE while LOW
    assert next_state == 0 and reward == 0, "RECHARGE while LOW returns to HIGH for free"

    state, info = env.reset()
    print("--- Starting Recycling Robot Episode ---")
    env.render()
    
    action_names = ["SEARCH", "WAIT", "RECHARGE"]
    
    # 3. Run a random policy for 10 steps
    for step in range(1, 11):
        # Sample a random action from the action space
        action = env.action_space.sample() 
        
        # Take the step in the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: Action taken: {action_names[action]} -> Reward: {reward}")
        env.render()
        print("-" * 40)
        
    env.close()