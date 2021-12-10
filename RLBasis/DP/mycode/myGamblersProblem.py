import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy.core.numeric import tensordot
if "../" not in sys.path:
  sys.path.append("../")


def value_iteration_for_gamblers(p_h, theta=0.0001, discount_factor=1.0):

    
    rewards = np.zeros(101)
    rewards[100] = 1 
    
    def one_step_lookahead(s, V, rewards):
        A = np.zeros(min(s,100-s)+1)#初始化这么多个动作
        for action in range(1,len(A)):
            A[action] = p_h * (rewards[state+action] + discount_factor * V[s+action]) + (1-p_h) * (rewards[state-action] + discount_factor * V[s-action])
        return A
    
    V = np.zeros(101)
   
    while True:
        delta = 0
        for state in range(1,100):
            A = one_step_lookahead(state, V, rewards)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[state]))
            V[state] = best_action_value
        if delta < theta:
            break

    policy = np.zeros(100)
    for state in range(1,100):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(state, V, rewards)
        best_action = np.argmax(A)
        # Always take the best action
        policy[state] = best_action

    return policy, V


policy, v = value_iteration_for_gamblers(0.4)

print("Optimized Policy:")
print(policy)
print("")

print("Optimized Value Function:")
print(v)
print("")


# x axis values
x = range(100)
# corresponding y axis values
y = v[:100]
 
# plotting the points 
plt.plot(x, y)
 
# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Value Estimates')
 
# giving a title to the graph
plt.title('Final Policy (action stake) vs State (Capital)')
 
# function to show the plot
plt.show()

# Plotting Capital vs Final Policy

# x axis values
x = range(100)
# corresponding y axis values
y = policy
# plotting the bars
plt.bar(x, y, align='center', alpha=0.5)
# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Final policy (stake)')
# giving a title to the graph
plt.title('Capital vs Final Policy')
# function to show the plot
plt.show()