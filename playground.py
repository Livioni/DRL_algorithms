import gym
from torch.distributions import normal
name = "Pendulum-v1"

env = gym.make(name)
env.reset()
action = env.action_space.sample()
print(env.action_space)
print(env.observation_space)
observation, reward, done, info = env.step(action)
print(observation, reward, done, info)
while True:
    # action = normal.Normal(0,1).sample().item()
	action = env.action_space.sample()
	observation, reward, done, info = env.step(action)
	# print(reward)
	print(action)
	env.render()