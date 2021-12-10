import gym
import numpy as np
import sys

if "../" not in sys.path:
  sys.path.append("../") 

from cliff_walking import CliffWalkingEnv

env = CliffWalkingEnv()

print(env.reset())
env.render()

print(env.step(0))
env.render()

print(env.step(1))
env.render()

print(env.step(1))
env.render()

print(env.step(2))
env.render()