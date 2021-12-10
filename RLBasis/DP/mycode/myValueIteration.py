from gym.core import RewardWrapper
import numpy as np
import pprint
import sys
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

def change_policy(policys):
    action_tuple = []#这里初始化的列表

    for policy in policys:
        indexs, policy_arr = get_max_index(policy)
        action_tuple.append(tuple(indexs))#这里在列表里填充的tuple

    return action_tuple

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    
    V = np.zeros(env.nS)#initialize value arry
    policy = np.ones([env.nS, env.nA]) / env.nA#initialize random policy
    time = 0
    while True:
        delta = 0
        time += 1 
        for state in range(env.nS):
            v = V[state]
            action_value = np.zeros(env.nA)
            for action in range(env.nA):
                for prob,next_state,reward,done in env.P[state][action]:
                    action_value[action] += prob * (reward + discount_factor * V[next_state])
            V[state] = max(action_value)#bellman optimal eqution使用当前状态的最大行为值函数更新为当前状态值函数
            delta = max(delta,abs(v-V[state]))

        if delta < theta:
            break
    
    for state in range(env.nS):
        action_value = np.zeros(env.nA)
        for action in range(env.nA):
            for prob,next_state,reward,done in env.P[state][action]:
                action_value[action] += prob * (reward + discount_factor * V[next_state])
            index,policy_arr = get_max_index(action_value)
            policy[state][:] = 0
            policy[state][index] = 1

    print(V)
    print(time)
    # Implement!
    return policy, V

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

update_policy =  change_policy(policy)
print("this is updated policy:")
print(update_policy)

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
# print(np.reshape(np.argmax(policy, axis=1), env.shape))
# print("")

update_policy_type = change_policy(policy)
print(np.reshape(update_policy_type, (5,5)))
print("")

# print("Value Function:")
# print(v)
# print("")

# print("Reshaped Grid Value Function:")
# print(v.reshape(env.shape))
# print("")