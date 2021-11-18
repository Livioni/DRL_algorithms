import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from itertools import count
import gym
import matplotlib.pyplot as plt

#用策略梯度方法解决mountain_car问题
learning_rate = 0.001
discount_factor = 0.99
episode_number = 500
env = gym.make('myMountainCar-v0')

#定义策略网络
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(2, 24) #2个状态输入，小车的速度和位置
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 3) #输出为3个动作 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return F.softmax(x,dim=0)

policy = PolicyNet()
optimizer = optim.AdamW(policy.parameters(),lr=learning_rate)
state_pool = []
action_pool = []
reward_pool = []
saved_log_probs = []
step = 0
sum_reward = 0

#定义一个动作选择方法:输入一个动作输出动作的概率并采样一个动作
def action_select(state):
    state = torch.from_numpy(state).float()
    probs = policy(state)#从策略网络中输出动作概率
    m = Categorical(probs)#构造动作概率
    action = m.sample()#采样一个动作
    saved_log_probs.append(m.log_prob(action))
    return action.item()

# episode_durations = []
def plot_durations():
    plt.figure(2)
    # durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(i,sum_reward,'or')
    # plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated


for i in range(episode_number):
    state = env.reset()#初始状态：数组形式
    action = action_select(state) 
    env.render(mode='rgb_array')#显示画面
    for t in count():
        next_state,reward,done,_ = env.step(action)
        env.render(mode='rgb_array')
        state_pool.append(state)#记录状态
        action_pool.append(action)#记录动作
        reward_pool.append(reward)#记录收益
        state = next_state
        action = action_select(state)
        step += 1
        sum_reward += reward


        if done:
            # episode_durations.append(t + 1)
            plot_durations()
            break


    running_add = 0
    for t in reversed(range(step)):#反向计算每一次的期望收益，用采样值代替
        running_add = running_add * discount_factor + reward_pool[t]
        reward_pool[t] = running_add

    # Normalize reward 标准化收益
    reward_mean = np.mean(reward_pool)
    reward_std = np.std(reward_pool)
    for t in range(step):
        reward_pool[t] = (reward_pool[t] - reward_mean) / reward_std 
    
    optimizer.zero_grad()
    loss = 0
    for t in range(step):
        loss = loss - saved_log_probs[t] * reward_pool[t]

    loss.backward()
    optimizer.step()
    state_pool = []
    action_pool = []
    reward_pool = []
    saved_log_probs = []
    step = 0
    sum_reward = 0