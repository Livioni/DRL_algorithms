from typing import ValuesView
import numpy as np
import pprint
import sys
from myPolicyEvaluation import policy_eval
if "../" not in sys.path:
  sys.path.append("../") 
from gridworld import GridworldEnv
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

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

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    policy_stable = True
    policy_old = np.ones([env.nS, env.nA]) / env.nA
    time = 1
    while time<3:
        V = policy_eval_fn(policy,env,discount_factor)
        print(V)
        for state in range(env.nS):
            action_value = np.zeros(env.nA)
            for action in range(env.nA):
                for prob,next_state,reward,done in env.P[state][action]:
                    action_value[action] += prob * (reward + discount_factor * V[next_state])
            index,policy_arr = get_max_index(action_value)
            policy[state][:] = 0
            policy[state][index] = 1
            

        if  (policy_old != policy).all:
            policy_stable = False
            policy_old = policy
            time += 1
            print("第%d次迭代，策略为：\n" % time)
            print(policy)
        else:
            policy_stable = True

        if policy_stable == True:
            break

    return policy, V

policy,V = policy_improvement(env)
print("值函数的网格形式:")
print(V.reshape(env.shape))
print("")

