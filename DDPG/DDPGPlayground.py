#深度强化学习——原理、算法与PyTorch实战，代码名称：代40-DDPG算法的实验过程.py
import numpy as np
import torch
import gym
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
                )

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)


    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)

actor1=Actor(17,6,1.0)
for ch in actor1.children():
    print(ch)
    print("*********************")

critic1=Critic(17,6)
for ch in critic1.children():
    print(ch)


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
    # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        def save(self, filename):
            torch.save(self.critic.state_dict(), filename + "_critic")
            torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

            torch.save(self.actor.state_dict(), filename + "_actor")
            torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        def load(self, filename):
            self.critic.load_state_dict(torch.load(filename + "_critic"))
            self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
            self.critic_target = copy.deepcopy(self.critic)

            self.actor.load_state_dict(torch.load(filename + "_actor"))
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
            self.actor_target = copy.deepcopy(self.actor)


        # Runs policy for X episodes and returns average reward
        # A fixed seed is used for the eval environment
        def eval_policy(policy, env_name, seed, eval_episodes=10):
            eval_env = gym.make(env_name)
            eval_env.seed(seed + 100)

            avg_reward = 0.
            for _ in range(eval_episodes):
                state, done = eval_env.reset(), False
            while not done:
                action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

            avg_reward /= eval_episodes

            print("---------------------------------------")
            print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            return avg_reward


policy = "DDPG"
env_name = "Walker2d-v2" # OpenAI gym environment name
seed = 0 # Sets Gym, PyTorch and Numpy seeds
start_timesteps = 25e3 # Time steps initial random policy is used
eval_freq = 5e3 # How often (time steps) we evaluate
max_timesteps = 1e6 # Max time steps to run environment
expl_noise = 0.1 # Std of Gaussian exploration noise
batch_size = 256 # Batch size for both actor and critic
discount = 0.99 # Discount factor
tau = 0.005 # Target network update rate
policy_noise = 0.2 # Noise added to target policy during critic update
noise_clip = 0.5 # Range to clip target policy noise
policy_freq = 2 # Frequency of delayed policy updates
save_model = "store_true" # Save model and optimizer parameters
load_model = "" # Model load file name, "" doesn't load, "default" uses file_name

file_name = f"{policy}_{env_name}_{seed}"
print("---------------------------------------")
print(f"Policy: {policy}, Env: {env_name}, Seed: {seed}")
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")

if save_model and not os.path.exists("./models"):
    os.makedirs("./models")

env = gym.make(env_name)

# Set seeds
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

kwargs = {
"state_dim": state_dim,
"action_dim": action_dim,
"max_action": max_action,
"discount": discount,
"tau": tau,
}

policy = DDPG(**kwargs)

if load_model != "":
    policy_file = file_name if load_model == "default" else load_model
    policy.load(f"./models/{policy_file}")

replay_buffer = ReplayBuffer(state_dim, action_dim)

# Evaluate untrained policy
evaluations = [eval_policy(policy, env_name, seed)]

state, done = env.reset(), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0

for t in range(int(max_timesteps)):

    episode_timesteps += 1

# Select action randomly or according to policy
if t < start_timesteps:
    action = env.action_space.sample()
else:
    action = (
    policy.select_action(np.array(state))
    + np.random.normal(0, max_action * expl_noise, size=action_dim)
    ).clip(-max_action, max_action)

# Perform action
next_state, reward, done, _ = env.step(action)
done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

# Store data in replay buffer
replay_buffer.add(state, action, next_state, reward, done_bool)

state = next_state
episode_reward += reward

# Train agent after collecting sufficient data
if t >= start_timesteps:
    policy.train(replay_buffer, batch_size)

if done:
    # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
    print(
    f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
# Reset environment
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num += 1

# Evaluate episode
if (t + 1) % eval_freq == 0:
    evaluations.append(eval_policy(policy, env_name, seed))
    np.save(f"./results/{file_name}", evaluations)

if save_model:
    policy.save(f"./models/{file_name}")