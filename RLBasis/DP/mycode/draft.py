import numpy as np
import sys

from myPolicyIteration import policy_improvement
if "../" not in sys.path:
  sys.path.append("../") 
from gridworld import GridworldEnv
env = GridworldEnv()
random_policy = np.ones([env.nS, env.nA]) / env.nA

# print(random_policy)

def get_max_index(action_values):
    indexs = []
    policy_arr = np.zeros(len(action_values))

    action_max_value = np.max(action_values)

    for i in range(len(action_values)):
        action_value = action_values[i]

        if action_value == action_max_value:
            indexs.append(i)
            policy_arr[i] = 1
    return indexs,policy_arr

def change_policy(policys):
    action_tuple = []

    for policy in policys:
        indexs, policy_arr = get_max_index(policy)
        action_tuple.append(tuple(indexs))

    return action_tuple

# policy = np.ones(100)
# print(policy)
print(range(100))
policy = []
for state in range(100):
    policy[state] = np.ones(100)/100