import gym
import numpy as np
import sys

if "../" not in sys.path:
  sys.path.append("../") 

from windy_gridworld import WindyGridworldEnv

env = WindyGridworldEnv()

# print(env.isd)
# print(env.P)

# print(env.reset())
# env.render()

# print(np.unravel_index(10,[7,10]))#从二维矩阵中编号从左到右从上往下
# action = np.random.choice([0,1,2,3],p=[0.25,0.25,0.25,0.25])
# print(action)

print(np.arange(len(policy(state)))
# print(np.array([-1,0]))
# print(np.ravel_multi_index((1,9),[7,10]))#从编号反求二维矩阵的（行，列） 所以索引都是从0开始
# print(env.step(1))
# env.render()

# print(env.step(1))
# env.render()

# print(env.step(1))
# env.render()

# print(env.step(2))
# env.render()

# print(env.step(1))
# env.render()

# print(env.step(1))
# env.render()