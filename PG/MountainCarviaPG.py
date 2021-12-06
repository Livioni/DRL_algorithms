import numpy as np
import torch,os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
from itertools import count
import gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment='Cartpole Reward Record')

#用策略梯度方法解决mountain_car问题
env = gym.make('MountainCar-v0')
env = env.unwrapped
#用env.unwrapped可以得到原始的类，原始类想step多久就多久，不会200步后失败：
env.seed(1)
learning_rate = 0.02         #学习率
discount_factor = 0.9       #折扣值
episode_number = 1000           #幕数
state_pool = []               #状态列表
action_pool = []              #动作列表
saved_log_probs = []          #采取动作的概率
step = 0                      #步骤数
sum_reward = 0                #plot收益需要
episode_durations = []        #cartpole里面的持续状态数
eps = np.finfo(np.float32).eps.item()

#定义策略网络
#这个网络很优秀
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(2, 20)
        self.affine2 = nn.Linear(20, 3)
        self.rewards = []
    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = self.affine2(x)
        return x

policy = Policy()

if os.path.exists('models/MountainCar_PG.pkl'):
    policy = torch.load('models/MountainCar_PG.pkl')
    print("Network loaded.")

optimizer = torch.optim.Adam(policy.parameters(),lr=learning_rate)


#定义一个动作选择方法:输入一个动作输出动作的概率并采样一个动作
def action_select(state,network):
    state = torch.from_numpy(state).float()
    out = network(state)#从策略网络中输出动作概率
    probs = torch.softmax(out,dim=0)
    m = Categorical(probs)#构造动作概率
    action = m.sample()#采样一个动作
    saved_log_probs.append(m.log_prob(action))
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
    policy.rewards.append(reward)

def learning():
    policy_loss = []
    reward_pool = []
    # Step 1: 计算每一步的状态价值
    # 处理一条轨迹中的每一个状态的回报
    running_add = 0
    for t in reversed(range(step)):#反向计算每一次的期望收益，用采样值代替
        running_add = running_add * discount_factor + policy.rewards[t]
        reward_pool.insert(0,running_add)

    reward_pool = torch.tensor(reward_pool)    
    # Normalize reward 标准化收益
    reward_mean = reward_pool.mean()
    reward_std = reward_pool.std()
    for t in range(step):
        reward_pool[t] = (reward_pool[t] - reward_mean) / (reward_std + eps) 

    optimizer.zero_grad()
    # Step 2: 前向传播
    for reward, log_prob in zip(reward_pool, saved_log_probs):
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = Variable(torch.Tensor(policy_loss).sum(),requires_grad = True)
    policy_loss.backward()
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
        if (done or t>10000):
            episode_durations.append(sum_reward)
            print('Iteration: {}, Score: {}'.format(i, sum_reward))
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
torch.save(policy, 'models/MountainCar_PG.pkl')
#加载网络
if os.path.exists('models/MountainCar_PG.pkl'):
    trained_network = torch.load('models/MountainCar_PG.pkl')
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
