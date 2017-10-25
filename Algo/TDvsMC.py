"""
This is an example of policy evaluation by TD(0) and MC on a random walk at different alpha values.

Scenario:
Our protaganist, Chad is attemting to stubble his way home after a long night at the pub. In his druken state, Chad has forgotten the direction to his house and has resorted to taking random steps. There are a total of five states, (1, 2, 3, 4, 5)  inbetween the pub at state 0 and his home at state 6. There is an equal probability for Chad going right towards his home or left towards the pub.

Reward for making it home: +1
Reward for ending up back at the pub: 0


"""
import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt


def td_zero(states, alpha=0.1):
    state = 3
    path = [state]
    rewards = [0]
    while True:
        reward = 0
        last_state = state
        state += np.random.choice([-1, 1])
        path.append(state)
        states[last_state] += alpha * \
            (reward + states[state] - states[last_state])
        if state == 6 or state == 0:
            break
        rewards.append(reward)


def mc(states, alpha=0.1):
    state = 3
    path = [state]
    reward = 0
    while True:
        state += np.random.choice([-1, 1])
        path.append(state)
        if state == 6:
            reward = 1
            break
        elif state == 0:
            break
    for state_ in path[:-1]:
        states[state_] += alpha * (reward - states[state_])


def state_values():
    episodes = [0, 1, 10, 100]
    currentStates = np.copy(states)
    plt.figure(1)
    axisX = np.arange(0, 7)
    for i in range(101):
        if i in episodes:
            plt.plot(axisX, currentStates, label=str(i) + ' episodes')
        td_zero(currentStates)
    plt.plot(axisX, expected_v, label='expected values')
    plt.xlabel('State')
    plt.ylabel('State Value')
    plt.legend()


def calc_RMS_err():
    Alpha_values = [0.15, 0.1, 0.05]
    plt.figure(2)
    axisX = np.arange(0, 101)
    runs = 100
    for method in ['TD', 'MC']:
        for alpha in Alpha_values:
            total_err = np.zeros(101)
            for run in range(runs):
                errors = []
                currentStates = np.copy(states)
                # Run through 100 episodes
                for i in range(101):
                    errors.append(
                        np.sqrt(np.sum(np.power(expected_v - currentStates, 2)) / 5.0))
                    if method == 'TD':
                        td_zero(currentStates, alpha=alpha)
                    else:
                        mc(currentStates, alpha=alpha)
                total_err += np.asarray(errors)
            total_err /= runs
            plt.plot(axisX, total_err, label=method + ', alpha=' + str(alpha))
    plt.xlabel('Walks/episodes')
    plt.ylabel('Error')
    plt.legend()


if __name__ == '__main__':
    states = np.zeros(7)
    states[1:6] = 0.5
    states[6] = 1
    expected_v = np.zeros(7)
    expected_v[1:6] = np.arange(1, 6) / 6.0
    expected_v[6] = 1

    state_values()
    calc_RMS_err()
    plt.show()
