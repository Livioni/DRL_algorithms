## Policiy_Gradient

策略梯度是强化学习的一类方法，大致的原理是使用神经网络构造一个策略网络，输入是状态，输出为动作的概率，在这些动作里采样选择一个动作去与环境交互，这样可以起到**Exploration 和 Exploitation的tradeoff**。与环境交互后获得一个收益，根据设计的损失函数和收益使用梯度上升法更新网络参数。输出的直接是策略$\pi(a|s)$，以概率的形式呈现，且$\sum_{a} \pi(a \mid s)=1$。

### 目标函数：

这里的目标函数是指在该策略下所取得的（状态，动作）价值函数期望。

<img src="/Users/livion/Documents/GitHub/DRL_algorithms/readme.assets/image-20211211101353371.png" alt="image-20211211101353371" style="zoom:20%;" />

为了使这个目标函数最大化，需要做Gradient Ascent，即求出梯度，

<img src="/Users/livion/Documents/GitHub/DRL_algorithms/readme.assets/image-20211211101625686.png" alt="image-20211211101625686" style="zoom:20%;" />



### REINFORCE

根据以上对目标函数的推导，提出REINFORCE，一种Monte-Carlo policy gradient算法，依赖于用Monte-Carlo估计的return来更新参数![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta)。

REINFORCE可以work是因为sample gradient的期望和真实的gradient一样。

<img src="/Users/livion/Documents/GitHub/DRL_algorithms/readme.assets/image-20211211100223043.png" alt="image-20211211100223043" style="zoom:20%;" />

我们可以通过采样的轨迹来获得![[公式]](https://www.zhihu.com/equation?tex=G_t)（即discounted reward的和，可见前面的变量定义表)。由于一次更新需要一个完整的trajectory，REINFORCE被称为Monte-Carlo方法。

REINFORCE里的梯度更新公式：

<img src="/Users/livion/Documents/GitHub/DRL_algorithms/readme.assets/image-20211211100310609.png" alt="image-20211211100310609" style="zoom:20%;" />

伪代码如下：

![截屏2021-11-19 16.15.47](/Users/livion/Library/Application Support/typora-user-images/截屏2021-11-19 16.15.47.png)

#### 实现

首先了解最重要的损失函数是怎么求的，在pytoch库中，描述了核心代码实现：

> In practice we would sample an action from the output of a network, apply this action in an environment, and then use `log_prob` to construct an equivalent loss function. Note that we use a negative because optimizers use gradient descent, whilst the rule above assumes gradient ascent. With a categorical policy, the code for implementing REINFORCE would be as follows:
>
> ```python
> probs = policy_network(state)
> # Note that this is equivalent to what used to be called multinomial
> m = Categorical(probs)
> action = m.sample()
> next_state, reward = env.step(action)
> loss = -m.log_prob(action) * reward
> loss.backward()
> 
> ```
>

这里的$log\_prob(action)$相当于求出来$\frac{\nabla_{\theta} \pi\left(A_{t} \mid S_{t}, \theta_{t}\right)}{\pi\left(A_{t} \mid S_{t}, \theta_{t}\right)}$。在一般的REINFORCE实现中，我们一般是用一幕的数据做一次Gradient Ascent，因此，在一幕中的每一步，我们记录轨迹数据（状态，动作，回报，下一个状态，是否完成）。在一幕完成后，先计算每一步的累计回报$G$。

```python
running_add = 0
for t in reversed(range(step)):#反向计算每一次的期望收益，用采样值代替
	running_add = running_add * discount_factor + reward_pool[t]
	update_network(running_add,state_pool[t],value_prediction)
	reward_pool[t] = running_add - value_estimator(state_pool[t],value_prediction).item()
```

 接下来一般需要计算累计回报的[标准分数](https://baike.baidu.com/item/%E6%A0%87%E5%87%86%E5%88%86%E6%95%B0/1694868?fr=aladdin)，能够表明原数据在其分布中的位置外，还能对未来不能直接比较的各种不同单位的数据进行比较。

```python
# Normalize reward 标准化收益
reward_mean = np.mean(reward_pool)
reward_std = np.std(reward_pool)
for t in range(step):
  reward_pool[t] = (reward_pool[t] - reward_mean) / reward_std 
```

有了计算好的reward_pool和记录的action_pool后，就可以计算梯度了，注意我们目标是收益最大化，因此这里使用梯度上升法，losss计算公式需要加上负号。

```python
optimizer.zero_grad()

for i in range(steps):
  state = state_pool[i]
  action = Variable(torch.FloatTensor([action_pool[i]]))
  reward = reward_pool[i]
  probs = policy(state)
  c = Categorical(probs)
  loss = -c.log_prob(action) * reward
  loss.backward()

optimizer.step()
```

接下来关注一下在连续动作空间和连续动作空间上Policy网络结构的不同点和动作选取方式：

1. 对于离散动作空间，策略网络输入为状态的维数，输出为动作的维数，

```python
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()     
        self.state_space = state_space
        self.action_space = action_space
        self.affine1 = nn.Linear(self.state_space, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, self.action_space)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        return x

policy = Policy()
```

​				根据输出动作的概率评分值（这里还不是概率）进行Softmax操作，使得选择动作的概率之和为1.

<img src="/Users/livion/Documents/GitHub/DRL_algorithms/readme.assets/image-20211211104050387.png" alt="image-20211211104050387" style="zoom:20%;" />

​				之后通过Categorical()函数构造概率分布，通过sample()函数进行动作采样。

```python
def action_select(state,network):
    state = torch.from_numpy(state).float()
    out = network(state)#从策略网络中输出动作概率
    probs = torch.softmax(out,dim=0)
    m = Categorical(probs)#构造动作概率
    action = m.sample()#采样一个动作
    return action.item()
```

2. 对于连续动作空间，我们不直接计算每个动作的概率，而是学习动作的概率分布，例如根据正态分布选择动作,正太分布的概率密度函数可以写为：

   <img src="/Users/livion/Documents/GitHub/DRL_algorithms/readme.assets/image-20211211104610639.png" alt="image-20211211104610639" style="zoom:20%;" />

   $\mu$和$\sigma$是我们需要的两个参数，因此完全可以使用网络来近似这两个参数，然后构造成正态分布的形式，然后再采样一个动作，此时的策略网络这样写：

```python
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = state_space
        self.affine1 = nn.Linear(self.state_space, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = self.affine2(x)
        mu = 2 * F.tanh(x[0])
        sigma = F.softplus(x[1])
        return mu,sigma
```

​				其中softpuls操作保证了标准差$\sigma$为正数，$\mu$的范围根据实际动作空间取值来定。	

```python
#定义一个动作选择方法:输入一个动作输出动作的概率并采样一个动作
def action_select(state,network):
    state = torch.from_numpy(state).float()
    mu,sigma = network(state)#从策略网络中输出动作概率
    m = normal.Normal(mu,sigma)
    action = m.sample()#采样一个动作
    return action,m
```

到此为止，REINFORCE算法完成。

### REINFORCE with Baseline

带有基线的REINFORCE算法可以有效降低方差，也加快了学习速度，基线应该根据状态的变化而变化，在一些状态下，所以动作的价值可能都比较大，因此我们需要一个较大的基线用于区分拥有更大值的动作和相对值不那么高的动作，其他状态下当所有动作的值都比较低时，基线也应该较低。

经过证明得到带有Baseline的更新公式：

<img src="/Users/livion/Documents/GitHub/DRL_algorithms/readme.assets/image-20211211105338700.png" alt="image-20211211105338700" style="zoom:20%;" />

在累计回报中减去这个基线不会使更新值的期望发生变化。在实现中，我们可以使用另外一个神经网络近似的值函数来作为基线，在代码上体现在计算累计收益上：

```python
    running_add = 0
    for t in reversed(range(step)):#反向计算每一次的期望收益，用采样值代替
        running_add = running_add * discount_factor + reward_pool[t]
        update_network(running_add,state_pool[t],value_prediction)
        reward_pool[t] = running_add - value_estimator(state_pool[t],value_prediction).item()
        #value_estimator是另外一个神经网络，用来近似状态价值函数，作为基线。这个网络的学习率需要精心设计。
```

## Actor Critic 

REINFORCE算法是利用一幕中采样到的数据进行更新的，这样虽然是无偏的，但是方差高，使用自举的方法可以在每一步或几步后就做出策略改进，这样做引入了偏差但减小了方差，很多算法都是基于Actor_Critic架构的。

Actor_Critic的核心思想就是将PG和值函数逼近法相结合，同时学习策略和值函数，实现实时在线地学习。

AC算法的流程可以表述为：

1. Agent根据任务的当前状态选择一个动作（基于当前策略）；
2. 评论家根据当前状态-动作对，针对策略的表现打分；
3. 行动家依据评论家的打分，改进策略；
4. 评论家根据环境返回的reward，改进打分策略；
5. 利用更新后的策略在下一状态选择动作，重复以上过程。

### 优势函数（advantage function）

AC算法里的优势函数类似于PG算法里的累计回报，Actor的目标是最大化优势函数，根据基线的思想，将状态$s$的价值$v_\pi(s)$作为基线$b(s)$，定义优势函数$A_{\pi_\theta}$：

<img src="/Users/livion/Documents/GitHub/DRL_algorithms/readme.assets/image-20211211112227142.png" alt="image-20211211112227142" style="zoom:20%;" />

<img src="/Users/livion/Documents/GitHub/DRL_algorithms/readme.assets/image-20211211112313169.png" alt="image-20211211112313169" style="zoom:20%;" />

利用优势函数可以减小策略梯度的方差，一般采样TD误差代替$q_{\pi_\theta}(s,a)$:

<img src="readme.assets/image-20211211112829549.png" alt="image-20211211112829549" style="zoom:20%;" />

于是目标函数变为：<img src="readme.assets/image-20211211113459960.png" alt="image-20211211113459960" style="zoom:20%;" />

伪代码如下：

<img src="readme.assets/截屏2021-12-11 11.29.58.png" alt="截屏2021-12-11 11.29.58" style="zoom:50%;" />

网络更新部分代码：

```python
def leaning(self,state,next_state,done,reward,log_prob):
  next_state = torch.FloatTensor(next_state)
  state = torch.FloatTensor(state)
  td_target = reward + (args.gamma * self.critic(next_state) * (1-done)).detach()
  value = self.critic(state)
  td_error = td_target - value.detach() #优势函数

  critic_loss = self.loss(td_target, value) #评论家损失函数
  writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_training)
  actor_loss = -(log_prob * td_error)   #同REINFORCE
  writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_training)

  self.actor_optimizer.zero_grad()
  actor_loss.backward()
  self.actor_optimizer.step()

  self.critic_optimizer.zero_grad()
  critic_loss.backward()
  self.critic_optimizer.step()

  self.num_training += 1
```

## Deep Deterministic Policy Gradient (DDPG)