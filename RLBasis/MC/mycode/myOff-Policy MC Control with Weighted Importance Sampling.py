from os import stat
import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

from numpy.core.numeric import ones
if "../" not in sys.path:
  sys.path.append("../") 
from blackjack import BlackjackEnv
import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()
def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn

def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.
    
    Args:
        Q: A dictionary that maps from state -> action values
        
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """
    def policy_fn(observation):
        A_max = np.argmax(Q[observation])
        A = np.zeros(len(Q))
        A[A_max] = 1.0
        return A
    return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
        discount_factor: Gamma discount factor.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    """
    
    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)

    for i_episode in range(1,num_episodes+1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        episode = []
        state = env.reset()
        for t in range(100):
            prob = behavior_policy(state)#
            action = np.random.choice(np.arange(len(prob)),p=prob)
            next_state, reward, done, _ = env.step(action)
            episode.append((state,action,reward))
            if done:
                break
            else:
                state = next_state
        W = 1.0
        G = 0.0
        #增量式算法
        #对该幕的每个出现的状态都进行价值更新
        for t in range(len(episode))[::-1]:
            state, action ,reward = episode[t]
            G = discount_factor * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            if action != np.argmax(target_policy(state)):
                break
            else:
                W = W/behavior_policy(state)[action] 


    return Q, target_policy

random_policy = create_random_policy(env.action_space.n)

Q, policy = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)
# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")