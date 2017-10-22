"""
Solution for value_iteration problem
"""
import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    while True:
        delta = 0
        for s in range(env.nS):
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0

    return policy, V


if __name__ == '__main__':
    env = GridworldEnv()
    policy, v = value_iteration(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")
    expected_v = np.array([0, -1, -2, -3, -1, -2, -3, -
                           2, -2, -3, -2, -1, -3, -2, -1,  0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
