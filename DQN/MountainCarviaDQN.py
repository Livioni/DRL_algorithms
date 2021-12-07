import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

# Hyper-parameters
seed = 1
render = False
num_episodes = 40000
env = gym.make('MountainCar-v0').unwrapped #无限轮次数
num_state = env.observation_space.shape[0] #2个状态
num_action = env.action_space.n #3个动作
torch.manual_seed(seed)
env.seed(seed)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])#有点像结构体定义

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = self.fc2(x)
        return action_prob

class DQN():

    capacity = 8000        #经验池
    learning_rate = 1e-3   #学习率
    memory_count = 0       #经验池里的数据个数
    batch_size = 256       #学习批量数
    gamma = 0.995          #折扣率
    update_count = 0       #更新次数

    def __init__(self):
        super(DQN, self).__init__()
        self.target_net, self.act_net = Net(), Net() #目标网络和真实网络
        self.memory = [None]*self.capacity #[]列表里有capacity个None
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)#Adam优化器
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')


    def select_action(self,state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)#增加一个维度
        value = self.act_net(state)#从Q网络众多动作中获得一个值
        action_max_value, index = torch.max(value, 1)#1表示每行的最大值，因为value只有一行，所以就是最大值
        action = index.item()#取最大的索引值
        if np.random.rand(1) >= 0.9: # epslion greedy
            action = np.random.choice(range(num_action), 1).item()#10%的概率采取非贪心值
        return action

    def store_transition(self,transition):#存储经验回放数据
        index = self.memory_count % self.capacity #计算新的经验该放在哪个位置 总数%容量 余数等于索引
        self.memory[index] = transition #存放一个经验
        self.memory_count += 1 #总数+1 
        return self.memory_count >= self.capacity #返回经验池是否存满，1为满

    def update(self):
        if self.memory_count >= self.capacity: #如果经验池存满了
            state = torch.tensor([t.state for t in self.memory]).float()#将经验池中的state放进列表state里面 
            action = torch.LongTensor([t.action for t in self.memory]).view(-1,1).long()#将经验池中的action放进 action列表里
            reward = torch.tensor([t.reward for t in self.memory]).float()
            next_state = torch.tensor([t.next_state for t in self.memory]).float()

            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            with torch.no_grad():
                target_v = reward + self.gamma * self.target_net(next_state).max(1)[0] #使用目标网络计算Target .max(1)[0]按行取最大 [0]取第一个

            #Update...
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
                v = (self.act_net(state).gather(1, action))[index]#相当于找到act网络中对应（s，a）的q值
                loss = self.loss_func(target_v[index].unsqueeze(1), v)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                self.update_count +=1
                if self.update_count % 100 ==0:
                    self.target_net.load_state_dict(self.act_net.state_dict())#每隔100步复制网络参数
                break #这里需要加上break 不然就是更新整个经验池了
        else:
            print("Memory Buff is too less")
def main():

    agent = DQN()
    for i_ep in range(num_episodes):
        state = env.reset()
        if render: env.render()
        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            if render: env.render()
            transition = Transition(state, action, reward, next_state)
            agent.store_transition(transition)
            state = next_state
            if done or t >=9999:
                agent.writer.add_scalar('live/finish_step', t+1, global_step=i_ep)
                agent.update()
                if i_ep % 10 == 0:
                    print("episodes {}, step is {} ".format(i_ep, t))
                break

if __name__ == '__main__':
    main()
