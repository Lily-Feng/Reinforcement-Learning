################################
# Hide warnings about wrappers because they are annoying and not directly
# caused by our code.
import warnings
warnings.showwarning = lambda *args, **kwargs: None
################################

import argparse
from lib.envs.register_custom_grid_worlds import register_custom_grid_worlds

def main():
    #########################################################################
    # Create the top-level parser & setup subparsers for each assignment
    parser = argparse.ArgumentParser(description="Visualize your algorithms in different environments and with different configurations.")
    subparsers = parser.add_subparsers(dest='assignment', required=True)
    #########################################################################


    #########################################################################
    # Bandits
    parser_bandits = subparsers.add_parser('bandit', help="Bandits w/ 10-armed testbed")
    parser_bandits.add_argument('--num_iterations', type=int, required=False, default=10000, help="Number of iterations to run")
    parser_bandits.add_argument('--num_simulations', type=int, required=False, default=300, help="Number of simulations to run")
    parser_bandits.add_argument('--k', type=int, required=False, default=10, help="Number of arms in the bandit")
    parser_bandits.set_defaults(func=action_for_bandits)
    #########################################################################

    #########################################################################
    # Dynamic Programming
    register_custom_grid_worlds()
    environment_names = [
        'OneStateMDP-v0',
        'GridWorld2x2-v0',
        'WrappedFrozenLake-v0',
        'WrappedFrozenLakeSlippery-v0',
        'WrappedTaxi-v0',
        'WrappedCliffWalking-v0'
    ]
    
    parser_dp = subparsers.add_parser('dp', help="Value Iteration and Value Prediction - solving an MDP")
    parser_dp.add_argument('--environment', type=str, required=False, default='GridWorld2x2-v0', choices=environment_names, help="Environment to visualize")
    parser_dp.set_defaults(func=action_for_dp)
    #########################################################################

    #########################################################################
    # Parse the arguments, call the function
    args = parser.parse_args()
    args.func(args)
    #########################################################################


def action_for_bandits(args):
    from lib.visualize.bandit import visualize
    visualize(
        num_simulations = args.num_simulations,
        num_iterations = args.num_iterations,
        k = args.k
    )

def action_for_dp(args):
    from lib.visualize.dp import visualize
    visualize(args.environment)

if __name__ == "__main__":
    main()
