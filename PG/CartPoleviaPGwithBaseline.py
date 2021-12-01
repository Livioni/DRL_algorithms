from typing import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
import gym
from itertools import count
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment='Cartpole Reward Record')

#用策略梯度方法解决cartpole问题
env = gym.make('CartPole-v0')
learning_rate1 = 0.001          #学习率
learning_rate2 = 0.01
discount_factor = 0.9       #折扣值
episode_number = 1000            #幕数
state_pool = []               #状态列表
action_pool = []              #动作列表
reward_pool = []              #收益列表
step = 0                      #步骤数
sum_reward = 0                #plot收益需要
episode_durations = []        #cartpole里面的持续状态数
value_loss = nn.MSELoss()

#定义期望回报网络
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 36)
        self.dropout = nn.Dropout(p=0.6)
        self.fc3 = nn.Linear(36, 1)  # 当前状态的价值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.dropout(x))
        x = F.relu(self.fc3(x))
        return x

#这个网络很优秀
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        return x

policy = Policy()
value_prediction = PolicyNet()
optimizer = torch.optim.Adam(policy.parameters(),lr=learning_rate1)
value_optimizer = torch.optim.Adam(value_prediction.parameters(),lr=learning_rate2)

def value_estimator(state,network):
    state = torch.from_numpy(state).float()
    predicted_value = network(state)#从策略网络中输出动作概率
    return predicted_value

def update_network(true_value,state,network):
    predicted_value =  value_estimator(state,network)
    true_value = true_value * torch.ones(1,1)
    # loss1 = -value_loss(predicted_value,true_value)
    loss1 = -(true_value-predicted_value).pow(2).mean()#一样的
    value_optimizer.zero_grad()
    loss1.backward()
    value_optimizer.step()

#定义一个动作选择方法:输入一个动作输出动作的概率并采样一个动作
def action_select(state,network):
    state = torch.from_numpy(state).float()
    out = network(state)#从策略网络中输出动作概率
    probs = torch.softmax(out,dim=0)
    m = Categorical(probs)#构造动作概率
    action = m.sample()#采样一个动作
    return action.item()

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # plt.plot(i,sum_reward,'or')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated

def add_to_pool(state,action,reward):
    state_pool.append(state)#记录状态
    action_pool.append(action)#记录动作
    reward_pool.append(reward)#记录收益

def learning():
    global reward_pool,state_pool
    # Step 1: 计算每一步的状态价值
    # 处理一条轨迹中的每一个状态的回报
    running_add = 0
    for t in reversed(range(step)):#反向计算每一次的期望收益，用采样值代替
        running_add = running_add * discount_factor + reward_pool[t]
        update_network(running_add,state_pool[-t],value_prediction)
        reward_pool[t] = running_add - value_estimator(state_pool[-t],value_prediction).item()


    # Normalize reward 标准化收益
    reward_mean = np.mean(reward_pool)
    reward_std = np.std(reward_pool)
    for t in range(step):
        reward_pool[t] = (reward_pool[t] - reward_mean) / reward_std 
    
    optimizer.zero_grad()
    # Step 2: 前向传播
    state_pool = np.array(state_pool)
    softmax_input = policy.forward(torch.FloatTensor(state_pool))
    neg_log_prob = F.cross_entropy(input=softmax_input, target=torch.LongTensor(action_pool), reduction='none')
    reward_pool = torch.FloatTensor(reward_pool)
    # Step 3: 反向传播
    loss = torch.mean(neg_log_prob * reward_pool)
    loss.backward()
    optimizer.step()


for i in range(episode_number):
    state = env.reset()#初始状态：数组形式
    # env.render(mode='rgb_array')#显示画面
    for t in count():
        action = action_select(state,policy) 
        next_state,reward,done,_ = env.step(action)
        add_to_pool(state,action,reward)
        # env.render(mode='rgb_array')
        state = next_state
        sum_reward += reward
        step += 1
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            learning()
            break

    state_pool = []
    action_pool = []
    reward_pool = []
    step = 0
    sum_reward = 0

#绘制曲线
for data in range(episode_number):
    writer.add_scalar('Reward',episode_durations[data],data)   

writer.close()
#保存策略网络训练参数
torch.save(policy, 'Cartpole_net.pth')
#加载网络
trained_network = torch.load('Cartpole_net.pth')
#evaluation 1 episode
state = env.reset()#初始状态：数组形式
env.render(mode='rgb_array')#显示画面
for t in count():
    action = action_select(state,trained_network) 
    next_state,reward,done,_ = env.step(action)
    env.render(mode='rgb_array')
    state = next_state
    sum_reward += reward
    if done:
        print("The final reward is ",sum_reward)
        break

env.close()
