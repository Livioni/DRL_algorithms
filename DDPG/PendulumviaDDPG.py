import argparse
from itertools import count
import os, sys, random
import numpy as np
import time
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter, writer


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient 软更新目标网络参数
parser.add_argument('--test_iteration', default=10, type=int)#测试10次
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
parser.add_argument('--max_length_of_trajectory', default=10000, type=int)
# optional parameters
parser.add_argument('--render', default=True, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #每训练50个episode 保存一次网络
parser.add_argument('--load', default=False, type=bool) # 是否load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)#噪声
parser.add_argument('--max_episode', default=10000, type=int) # num of games
parser.add_argument('--update_iteration', default=20, type=int)
parser.add_argument('--sleep_time', default=0.05, type=float)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = gym.make("Pendulum-v1")

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

directory = 'models/' + "Pendulum-v1" + '/'
writer = SummaryWriter(directory, comment='Env Reward Record')
class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):#经验回放池最大容量
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):#压入经验回放池
        if len(self.storage) == self.max_size:#如果经验池满了，把最开始的经验挤掉
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):#采样batch_size个轨迹
        ind = np.random.randint(0, len(self.storage), size=batch_size)#在整个经验池里随机采样
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))#state
            y.append(np.array(Y, copy=False))#next_action
            u.append(np.array(U, copy=False))#action
            r.append(np.array(R, copy=False))#reward
            d.append(np.array(D, copy=False))#done

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):#行动家网络
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):#评论家网络
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):#输入的是状态和动作
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x#返回的是Q值


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)#行动家预测网络
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)#行动家目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())#加载目标网络参数
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)#行动家预测网络优化器

        self.critic = Critic(state_dim, action_dim).to(device)#评论家预测网络
        self.critic_target = Critic(state_dim, action_dim).to(device)#评论家目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())#加载目标参数网络
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)#行评论家预测网络优化器
        self.replay_buffer = Replay_buffer()
        

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()#用预测网络更新的动作

    def update(self):
        for it in range(args.update_iteration):#200次
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)#1表示没done 0表示done
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))#这里的target_Q是用目标网络更新的
            target_Q = reward + (done * args.gamma * target_Q).detach()#最后一幕收益为0 detach分离向量不计算梯度

            # Get current Q estimate
            current_Q = self.critic(state, action)#计算预测评论家网络q值

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)#使用MBGD，根据最小化损失函数来更新价值网络
            if args.mode == 'train':
                writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()#使用行动家预测网络的动作输入给评论家预测网络，得出期望Q更新行动家预测网络
            if args.mode == 'train':
                writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models#软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def main():
    agent = DDPG(state_dim, action_dim, max_action)
    ep_r = 0
    if args.mode == 'test': #如果是测试 
        agent.load()
        for i in range(args.test_iteration):#测10幕
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                time.sleep(args.sleep_time)
                if done or t >= args.max_length_of_trajectory:
                    print("Ep_i \t{}, the sum reward is \t{:0.2f}, the step is \t{}".format(i, ep_r, t+1))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        if args.load: 
            agent.load()#如果load之前网络，就load
            print("successfully loaded")
        total_step = 0
        for i in range(args.max_episode):
            total_reward = 0
            step =0
            sum_reward = 0#记录reward
            state = env.reset()
            for t in count():
                action = agent.select_action(state)#探索动作输入
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)#剪切最大最小动作输入

                next_state, reward, done, info = env.step(action)
                # if args.render and i >= args.render_interval : env.render()
                agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))
                sum_reward += reward
                state = next_state
                if done:
                    break
                step += 1
                total_reward += reward
            total_step += step+1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            agent.update()
            writer.add_scalar('Sum_Reward', sum_reward, i)
           # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            if i % args.log_interval == 0:
                agent.save()
    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
