import gym, os,time, argparse
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, distribution
from torch.utils.tensorboard import SummaryWriter, writer


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument("--env_name", default="CartPole-v0") # envs: CartPole-v0  LunarLander-v2 MountainCar-v0
parser.add_argument('--test_iteration', default=5, type=int)#测试10次
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
# optional parameters
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #每训练50个episode 保存一次网络
parser.add_argument('--load', default=False, type=bool) # 是否load model
parser.add_argument('--max_episode', default=1000, type=int) # num of games
parser.add_argument('--sleep_time', default=0.02, type=float)
args = parser.parse_args()

env = gym.make(args.env_name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

directory = 'models/' + 'AC' + args.env_name + '/'
writer = SummaryWriter(directory, comment='Env Reward Record')

class Actor(nn.Module): #策略网络
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution #输出动作概率分布


class Critic(nn.Module): #状态值函数网络
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value #输出状态值函数

class Actor_Critic(object):
    def __init__(self,state_size,action_size):
        self.actor = Actor(state_size,action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr = args.learning_rate)
        
        self.critic = Critic(state_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr = args.learning_rate)
        self.loss = nn.MSELoss()
        self.num_training = 0

    def select_action(self,state):
        state = torch.FloatTensor(state)
        dist = self.actor(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(0)
        return action.item(),log_prob

    def leaning(self,state,next_state,done,reward,log_prob):
        next_state = torch.FloatTensor(next_state)
        state = torch.FloatTensor(state)
        td_target = reward + (args.gamma * self.critic(next_state) * (1-done)).detach()
        value = self.critic(state)
        td_error = td_target - value.detach()
        
        critic_loss = self.loss(td_target, value)
        writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_training)
        actor_loss = -(log_prob * td_error)
        writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_training)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.num_training += 1

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
    agent = Actor_Critic(state_size,action_size)
    sum_reward = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):#测10幕
            state = env.reset()
            for t in count():
                action, _ = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                sum_reward += reward
                env.render()
                time.sleep(args.sleep_time)
                if done:
                    print("Episode: \t{}, the reward is \t{:0.2f}, the step is \t{}".format(i, sum_reward, t+1))
                    sum_reward = 0
                    break
                state = next_state

    elif args.mode == 'train':
        if args.load: 
            agent.load()
        for i in range(args.max_episode):
            state = env.reset()
            for t in count():
                action, log_prob = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.leaning(state,next_state,done,reward,log_prob)
                sum_reward += reward
                if done:
                    print('Episode: {}, reward: {}, step:{}'.format(i, sum_reward,t+1))
                    writer.add_scalar('Sum_Reward', sum_reward, i)
                    sum_reward = 0
                    break
                state = next_state                
            if i % args.log_interval == 0:
                agent.save()

    else:
        raise NameError("Mode wrong!!!")



if __name__ == '__main__':
    main()