import argparse
import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNet(nn.Module):#定义策略网络
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 24)#4个状态输入
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 2)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return F.softmax(x,dim=0)

policy = PolicyNet()
saved_log_probs = []

def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    saved_log_probs.append(m.log_prob(action))
    return action.item()

print(select_action(np.array([1,1,1,1])))
print(saved_log_probs)


# def select_action(self, state):
#     probs = policy(Variable(state))       
#     action = probs.multinomial().data
#     prob = probs[:, action[0,0]].view(1, -1)
#     log_prob = prob.log()
#     entropy = - (probs*probs.log()).sum()
#     return action[0], log_prob, entropy

plt.title('Training...')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.plot(1,2,'-r')
plt.plot(2,4,'-r')
plt.show()