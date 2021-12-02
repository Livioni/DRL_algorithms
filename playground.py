import gym
import Box2D

env = gym.make('LunarLander-v2')
 
print (env.observation_space)
print (env.action_space)
 
for i_episode in range(100):
	observation = env.reset()
	for t in range(100):
		env.render()
		print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
