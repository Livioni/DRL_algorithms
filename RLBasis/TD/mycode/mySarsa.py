import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt

if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from windy_gridworld import WindyGridworldEnv
import plotting

matplotlib.style.use('ggplot')
env = WindyGridworldEnv()
def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)#A=[prob,prob,prob,prob]based on epsilon-greedy
        return A
    return policy_fn

def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))#Q = {state : {action : value}}
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # state = np.ravel_multi_index((3,0),[7,10]) 改错(好像也行)
        state = env.reset()
        action = np.random.choice([0,1,2,3],p=policy(state))
        for t in itertools.count():#itertools.count用于计数，用于替代while
            # prob, next_state, reward, is_done = env.P[state][action]改错
            next_state, reward,is_done,_ = env.step(action)#用env.step()获取数据
            new_action = np.random.choice([0,1,2,3],p=policy(next_state))
            Q[state][action] += alpha * (reward + discount_factor * Q[next_state][new_action] - Q[state][action])

            #Update statistics#
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            if is_done:
                break
            state = next_state
            action = new_action   



    return Q, stats

Q, stats = sarsa(env, 200)
#运行程序正确，但是画图程序错误
plotting.plot_episode_stats(stats)

