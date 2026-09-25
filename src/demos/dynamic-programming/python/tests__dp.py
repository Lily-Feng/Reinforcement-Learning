import unittest
import numpy as np
import gymnasium as gym

from interfaces.policy import Policy
from interfaces.random_policy import RandomPolicy
from lib.envs.wrapped_gridworld import Actions
from lib.envs.one_state_mdp import OneStateMDP, OneStateMDPOptimalPolicy
from lib.envs.grid_world_2x2 import GridWorld2x2, GridWorld2x2OptimalPolicy

from assignments.dp import value_iteration, value_prediction

class TestValueIteration(unittest.TestCase):
    """value_iteration()"""
    def test_onestatemdp_value_iteration(self):
        """OneStateMDP: V"""
        env = OneStateMDP()

        # Test Value Iteration
        V, _ = value_iteration(env, np.zeros(env.observation_space.n), theta=1e-4, gamma=1.0)
        expected_V = np.array([1., 0.])

        self.assertIsNotNone(V, 'V should be defined')
        self.assertIsInstance(V, np.ndarray, 'V should be a numpy array')
        self.assertEqual(V.shape, (2,), 'V should be a numpy array of shape (nS,)')

        self.assertTrue(np.allclose(
            V,
            expected_V,
            rtol = 1e-5,
            atol = 1e-2
        ), f"Expected V = {expected_V}, but got: {V}")

    def test_onestatemdp_value_iteration_pi(self):
        """OneStateMDP: pi"""
        env = OneStateMDP()

        # Test Value Iteration
        _, pi = value_iteration(env, np.zeros(env.observation_space.n), theta=1e-4, gamma=1.0)

        self.assertIsNotNone(pi, 'pi should be defined')
        self.assertIsInstance(pi, Policy, 'pi should be an instance of Policy')
        self.assertEqual(pi.action(0), 0, f'The optimal action for state 0 should be {0}')

    def test_gridworld_value_iteration(self):
        """GridWorld2x2: V"""
        env = GridWorld2x2()

        # Test Value Iteration
        V, _ = value_iteration(env, np.zeros(env.observation_space.n), theta=1e-4, gamma=1.0)
        expected_V = np.array([-2., -1., -1., 0.])

        self.assertIsNotNone(V, 'V should be defined')
        self.assertIsInstance(V, np.ndarray, 'V should be a numpy array')
        self.assertEqual(V.shape, (env.observation_space.n,), 'V should be a numpy array of shape (nS,)')

        self.assertTrue(np.allclose(
            V,
            expected_V,
            rtol = 1e-5,
            atol = 1e-2
        ), f"Expected V = {expected_V}, but got: {V}")

    def test_gridworld_value_iteration_pi(self):
        """GridWorld2x2: pi"""
        env = GridWorld2x2()

        # Test Value Iteration
        _, pi = value_iteration(env, np.zeros(env.observation_space.n), theta=1e-4, gamma=1.0)

        self.assertIsNotNone(pi, 'pi should be defined')
        self.assertIsInstance(pi, Policy, 'pi should be an instance of Policy')
        self.assertIn(pi.action(0), [Actions.DOWN, Actions.RIGHT], f'The optimal action for state 0 should be DOWN ({Actions.DOWN}) or RIGHT ({Actions.RIGHT})')
        self.assertEqual(pi.action(1), Actions.DOWN, f'The optimal action for state 1 should be DOWN ({Actions.DOWN})')
        self.assertEqual(pi.action(2), Actions.RIGHT, f'The optimal action for state 2 should be RIGHT ({Actions.RIGHT})')


    def run_episode(self, env, pi):
        state, _ = env.reset()
        done = False
        G = 0
        while not done:
            action = pi.action(state)
            state, reward, done, terminated, _ = env.step(action)
            done = done or terminated
            G += reward
        return G
    
    def test_frozenlake_pi(self):
        """WrappedFrozenLake-v0: pi"""
        env = gym.make('WrappedFrozenLake-v0')
        _, pi = value_iteration(env, np.zeros(env.observation_space.n), theta=0.000001, gamma=0.99)

        Gs = []
        for i in range(1000):
            Gs.append(self.run_episode(env, pi))

        self.assertGreaterEqual(np.mean(Gs), 1.0, f'Should score >=1.0 on every episode of deterministic FrozenLake environment, got {np.mean(Gs)}')
    
    def test_frozenlake_stochastic_pi(self):
        """WrappedFrozenLakeSlippery-v0: pi"""
        env = gym.make('WrappedFrozenLakeSlippery-v0')
        _, pi = value_iteration(env, np.zeros(env.observation_space.n), theta=0.000001, gamma=0.99)

        Gs = []
        for i in range(100):
            Gs.append(self.run_episode(env, pi))

        expected_mean = 0.73816
        expected_stddev = 0.4396360021654278
        n = 100

        # Standard error of the mean
        standard_error = expected_stddev / np.sqrt(n)

        # Z-score calculation
        z_score = (np.mean(Gs) - expected_mean) / standard_error

        # Manually set the critical z-value for 99% confidence
        z_critical = 2.576  # 99% confidence level

        self.assertLessEqual(abs(z_score),  z_critical, f'Z-score should be less than 2.576, got {z_score}. Expected score ~{expected_mean}, but got {np.mean(Gs)}')

    def test_taxi(self):
        """WrappedTaxi-v0: pi"""
        env = gym.make('WrappedTaxi-v0')
        _, pi = value_iteration(env, np.zeros(env.observation_space.n), theta=0.000001, gamma=0.99)

        Gs = []
        for i in range(100):
            Gs.append(self.run_episode(env, pi))

        expected_mean = 7.92265
        expected_stddev = 2.581055399928487
        n = 100

        # Standard error of the mean
        standard_error = expected_stddev / np.sqrt(n)

        # Z-score calculation
        z_score = (np.mean(Gs) - expected_mean) / standard_error

        # Manually set the critical z-value for 99% confidence
        z_critical = 2.576  # 99% confidence level

        self.assertLessEqual(abs(z_score),  z_critical, f'Z-score should be less than 2.576, got {z_score}. Expected score ~{expected_mean}, but got {np.mean(Gs)}')


class TestValuePrediction(unittest.TestCase):
    """value_prediction()"""
    def test_onestatemdp_value_prediction_optimal_policy_v(self):
        """OneStateMDP using Optimal Policy: V"""
        env = OneStateMDP()
        policy = OneStateMDPOptimalPolicy()

        # Test Value Prediction
        V, _ = value_prediction(env, policy, np.zeros(env.action_space.n), theta=1e-4, gamma=1.0)

        self.assertIsNotNone(V, 'V should be defined')
        self.assertIsInstance(V, np.ndarray, 'V should be a numpy array')
        self.assertEqual(V.shape, (2,), 'V should be a numpy array of shape (nS,)')

        expected_V = np.array([1., 0.])
        self.assertTrue(np.allclose(
            V,
            expected_V,
            rtol = 1e-5,
            atol = 1e-2
        ), f"Expected V = {expected_V}, but got: {V}")

    def test_onestatemdp_value_prediction_optimal_policy_q(self):
        """OneStateMDP using Optimal Policy: Q"""
        env = OneStateMDP()
        policy = OneStateMDPOptimalPolicy()

        # Test Value Prediction
        _, Q = value_prediction(env, policy, np.zeros(env.action_space.n), theta=1e-4, gamma=1.0)

        self.assertIsNotNone(Q, 'Q should be defined')
        self.assertIsInstance(Q, np.ndarray, 'Q should be a numpy array')
        self.assertEqual(Q.shape, (env.observation_space.n, env.action_space.n), 'Q should be a numpy array of shape (nS, nA)')
        expected_Q = np.array([[1., 0.], [0., 0.]])
        self.assertTrue(np.allclose(
            Q,
            expected_Q,
            rtol = 1e-5,
            atol = 1e-2
        ), f"Expected Q = \n{expected_Q}, but got:\n{Q}")


    def test_onestatemdp_value_prediction_random_policy(self):
        """OneStateMDP using Random Policy: V"""
        env = OneStateMDP()
        behavior_policy = RandomPolicy(env.action_space.n)

        V, _ = value_prediction(env, behavior_policy, np.zeros(env.action_space.n), theta=1e-4, gamma=1.0)
        self.assertIsNotNone(V, 'V should be defined')
        self.assertIsInstance(V, np.ndarray, 'V should be a numpy array')
        self.assertEqual(V.shape, (env.observation_space.n,), 'V should be a numpy array of shape (nS,)')

        expected_V = np.array([0.1, 0.])
        self.assertTrue(np.allclose(
            V,
            expected_V,
            rtol = 1e-5,
            atol = 1e-2
        ), f"Expected V = {expected_V}, but got: {V}")


    def test_onestatemdp_value_prediction_random_policy_q(self):
        """OneStateMDP using Random Policy: Q"""
        env = OneStateMDP()
        behavior_policy = RandomPolicy(env.action_space.n)

        _, Q = value_prediction(env, behavior_policy, np.zeros(env.action_space.n), theta=1e-4, gamma=1.0)

        self.assertIsNotNone(Q, 'Q should be defined')
        self.assertIsInstance(Q, np.ndarray, 'Q should be a numpy array')
        self.assertEqual(Q.shape, (env.observation_space.n, env.action_space.n), 'Q should be a numpy array of shape (nS, nA)')
        expected_Q = np.array([[0.19, 0.], [0., 0.]])
        self.assertTrue(np.allclose(
            Q,
            expected_Q,
            rtol = 1e-5,
            atol = 1e-2
        ), f"Expected Q =\n{expected_Q}, but got:\n{Q}")


    def test_gridworld_value_prediction_optimal_policy_v(self):
        """GridWorld2x2 using Optimal Policy: V"""
        env = GridWorld2x2()
        policy = GridWorld2x2OptimalPolicy()

        # Test Value Prediction
        V, _ = value_prediction(env, policy, np.zeros(env.action_space.n), theta=1e-4, gamma=1.0)

        self.assertIsNotNone(V, 'V should be defined')
        self.assertIsInstance(V, np.ndarray, 'V should be a numpy array')
        self.assertEqual(V.shape, (env.observation_space.n,), 'V should be a numpy array of shape (nS,)')

        expected_V = np.array([-2., -1., -1., 0.])
        self.assertTrue(np.allclose(
            V,
            expected_V,
            rtol = 1e-5,
            atol = 1e-2
        ), f"Expected V = {expected_V}, but got: {V}")

    def test_gridworld_value_prediction_optimal_policy_q(self):
        """GridWorld2x2 using Optimal Policy: Q"""
        env = GridWorld2x2()
        policy = GridWorld2x2OptimalPolicy()

        # Test Value Prediction
        _, Q = value_prediction(env, policy, np.zeros(env.action_space.n), theta=1e-4, gamma=1.0)

        self.assertIsNotNone(Q, 'Q should be defined')
        self.assertIsInstance(Q, np.ndarray, 'Q should be a numpy array')
        self.assertEqual(Q.shape, (env.observation_space.n, env.action_space.n), 'Q should be a numpy array of shape (nS, nA)')
        expected_Q = np.array([[-3., -3., -2, -2,], [-3., -2., -2, -1,], [-2., -3., -1, -2,], [0., 0., 0, 0,]])
        self.assertTrue(np.allclose(
            Q,
            expected_Q,
            rtol = 1e-5,
            atol = 1e-2
        ), f"Expected Q = \n{expected_Q}, but got:\n{Q}")


    def test_gridworld_value_prediction_random_policy(self):
        """GridWorld2x2 using Random Policy: V"""
        env = GridWorld2x2()
        behavior_policy = RandomPolicy(env.action_space.n)

        V, _ = value_prediction(env, behavior_policy, np.zeros(env.action_space.n), theta=1e-4, gamma=1.0)
        self.assertIsNotNone(V, 'V should be defined')
        self.assertIsInstance(V, np.ndarray, 'V should be a numpy array')
        self.assertEqual(V.shape, (env.observation_space.n,), 'V should be a numpy array of shape (nS,)')

        expected_V = np.array([-8, -6., -6., 0.])
        self.assertTrue(np.allclose(
            V,
            expected_V,
            rtol = 1e-5,
            atol = 1e-2
        ), f"Expected V = {expected_V}, but got: {V}")


    def test_gridworld_value_prediction_random_policy_q(self):
        """GridWorld2x2 using Random Policy: Q"""
        env = GridWorld2x2()
        behavior_policy = RandomPolicy(env.action_space.n)

        _, Q = value_prediction(env, behavior_policy, np.zeros(env.action_space.n), theta=1e-4, gamma=1.0)

        self.assertIsNotNone(Q, 'Q should be defined')
        self.assertIsInstance(Q, np.ndarray, 'Q should be a numpy array')
        self.assertEqual(Q.shape, (env.observation_space.n, env.action_space.n), 'Q should be a numpy array of shape (nS, nA)')
        expected_Q = np.array([[-9, -9., -7., -7.], [-9, -7., -7., -1.], [-7, -9., -1., -7.], [0., 0., 0., 0.]])
        self.assertTrue(np.allclose(
            Q,
            expected_Q,
            rtol = 1e-5,
            atol = 1e-2
        ), f"Expected Q =\n{expected_Q}, but got:\n{Q}")
    
