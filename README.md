## my_CartPole_Policiy_Gradient

策略梯度是强化学习的一类方法，大致的原理是使用神经网络构造一个策略网络，输入是状态，输出为动作的概率，在这些动作里采样选择一个动作去与环境交互，这样可以起到**Exploration 和 Exploitation的tradeoff**。与环境交互后获得一个收益，根据设计的损失函数和收益使用梯度上升法更新网络参数。输出的直接是策略$\pi(a|s)$，以概率的形式呈现，且$\sum_{a} \pi(a \mid s)=1$。

### 目标函数：

$$
\bar{R}_{\theta}=\sum_{\tau} R(\tau) p_{\theta}(\tau)=E_{\tau \sim p_{\theta}(\tau)}[R(\tau)]
$$

为了使这个目标函数最大化，需要做gradient ascent，即求出梯度，

### 似然技巧：

$$
\begin{aligned} \nabla_{\theta} \pi_{\theta}(s, a) &=\pi_{\theta}(s, a) \frac{\nabla_{\theta} \pi_{\theta}(s, a)}{\pi_{\theta}(s, a)} \\ &=\pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a) \end{aligned}
$$

### REINFORCE

REINFORCE，一种Monte-Carlo policy gradient算法，依赖于用Monte-Carlo估计的return来更新参数![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta)。

REINFORCE可以work是因为sample gradient的期望和真实的gradient一样。
$$
\begin{aligned}
\nabla_{\theta} J(\theta) &=\mathbb{E}_{\pi}\left[Q^{\pi}(s, a) \nabla_{\theta} \ln \pi_{\theta}(a \mid s)\right] \\
&=\mathbb{E}_{\pi}\left[G_{t} \nabla_{\theta} \ln \pi_{\theta}\left(A_{t} \mid S_{t}\right)\right]
\end{aligned} \quad ; \text { Because } Q^{\pi}\left(S_{t}, A_{t}\right)=\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}, A_{t}\right]
$$
我们可以通过采样的trajectory来获得![[公式]](https://www.zhihu.com/equation?tex=G_t)（即discounted reward的和，可见前面的变量定义表)。由于一次更新需要一个完整的trajectory，REINFORCE被称为Monte-Carlo方法。

REINFORCE里的梯度更新公式：
$$
\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_{t}+\alpha G_{t} \frac{\nabla_{\boldsymbol{\theta}} \pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)}{\pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)}
$$
伪代码如下：

![截屏2021-11-19 16.15.47](/Users/livion/Library/Application Support/typora-user-images/截屏2021-11-19 16.15.47.png)

代码实现如下：

```python
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
```

#### 对PyTorch中F.cross_entropy()函数的理解

PyTorch提供了求交叉熵的两个常用函数，一个是`F.cross_entropy()`，另一个是`F.nll_entropy()`

1. 交叉熵的公式：
   $$
   H(p, q)=-\sum_{i} P(i) \log Q(i)
   $$
   其中P为真实值，Q为预测值。

2. 计算交叉熵的步骤：

   1. 将predict_scores进行softmax运算，将运算结果记为pred_scores_soft；
   2. 将pred_scores_soft进行log运算，将运算结果记为pred_scores_soft_log；
   3. 将pred_scores_soft_log与真实值进行计算处理。

3. 举一个例子对计算进行说明：
   $$
   P_{1}=\left[\begin{array}{lllll}
   1 & 0 & 0 & 0 & 0
   \end{array}\right]
   \\
   Q_{1}=\left[\begin{array}{lllll}
   0.4 & 0.3 & 0.05 & 0.05 & 0.2
   \end{array}\right]\\
   
   H(p, q)=-\sum_{i} P(i) \log Q(i)=-(1 * \log 0.4+0 * \log 0.3+0 * \log 0.05+0 * \log 0.05+0 * \log 0.2) 
   =-\log 0.4 \approx 0.916
   $$

   ```python
   torch.nn.functional.cross_entropy(input, target, weight=None, size_average=True)
   ```

   ``F.Cross_entropy(input, target)``函数中包含了$Softmax$和$log$的操作，即网络计算送入的$input$参数不需要进行这两个操作。

   经过测试：

   ```python
   neg_log_prob = F.cross_entropy(input=torch.tensor([[0.4,0.6]]), target=torch.LongTensor([0]), reduction='none')
   n1 = torch.softmax(x,dim=0)
   print(n1)
   print(neg_log_prob)
   ```

   ```python
   tensor([0.4502, 0.5498])
   tensor([0.7981])
   ```

   可知，neg_log_prob = - log( softmax( input ) )[ target ] 

   当在t时刻，策略网络输出概率向量若与采样到的t时刻的动作越相似，那么交叉熵会越小。最小化这个交叉熵误差也就能够使策略网络的决策越接近我们采样的动作。最后用交叉熵乘上对应time-step的reward，就将reward的大小引入损失函数，entropy * reward越大，神经网络调整参数时计算得到的梯度就会越偏向该方向。

   

   进一步理解：

   ```python
   x = np.array([[1, 2,3,4,5],#共三3样本，有5个类别
                  [1, 2,3,4,5],
                  [1, 2,3,4,5]]).astype(np.float32)
    y = np.array([1, 1, 0])#这3个样本的标签分别是1,1,0即两个是第2类，一个是第1类
    x = torch.from_numpy(x)
    y = torch.from_numpy(y).long()
    
   soft_out = F.softmax(x,dim=1)#给每个样本的pred向量做指数归一化---softmax
   
   log_soft_out = torch.log(soft_out)#将上面得到的归一化的向量再point-wise取对数
   
   loss = F.nll_loss(log_soft_out, y)#将归一化且取对数后的张量根据标签求和，实际就是计算loss的过程
   """
   这里的loss计算式根据batch_size归一化后的，即是一个batch的平均单样本的损失，迭代一次模型对一个样本平均损失。
   在多个epoch训练时，还会求每个epoch内的总损失，用于衡量epoch之间模型性能的提升。
   """
   print(soft_out)
   print(log_soft_out)
   print(loss)
     
   loss = F.cross_entropy(x, y)
   print(loss)
   
   #输出：
   softmax：
   tensor([[0.0117, 0.0317, 0.0861, 0.2341, 0.6364],
   [0.0117, 0.0317, 0.0861, 0.2341, 0.6364],
   [0.0117, 0.0317, 0.0861, 0.2341, 0.6364]])
   
   
   tensor([[-4.4519, -3.4519, -2.4519, -1.4519, -0.4519],
   [-4.4519, -3.4519, -2.4519, -1.4519, -0.4519],
   [-4.4519, -3.4519, -2.4519, -1.4519, -0.4519]])
   
   
   tensor(3.7852)
   tensor(3.7852)
   ```

   ``softmax_input = policy.forward(torch.FloatTensor(state_pool))
   neg_log_prob = F.cross_entropy(input=softmax_input, target=torch.LongTensor(action_pool), reduction='none')``

### 带基线的Policy Gradient算法

使用另外一个神经网络近似期望回报值，当回报超过基线值时，该动作的概率提高，反之降低

训练效果

![](https://raw.githubusercontent.com/Livioni/DRL_algorithm/main/figures/CartPoleviaPGwithBaseline.png)

不带基线的效果

![](https://raw.githubusercontent.com/Livioni/DRL_algorithm/main/figures/CartpoleviaPGwithoutBaseline.png)

两种方法对比

![](https://raw.githubusercontent.com/Livioni/DRL_algorithm/main/figures/%E5%AF%B9%E6%AF%94.png)





红色是baseline，橘色是withoutbaseline 好像也差不多嘛
