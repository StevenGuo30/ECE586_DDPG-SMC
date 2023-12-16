import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from parsers import args
from RL_brain import ReplayBuffer, DDPG
from tqdm import tqdm
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -------------------------------------- #
# 环境加载
# -------------------------------------- #
from gym import spaces

class SMCenv(gym.Env):
    metadata = {'render.modes': ['console']}  # 如果需要其他渲染模式，可以修改这里

    def __init__(self):
        super(SMCenv, self).__init__()
        self.x1 = 0
        self.x2 = 0
        self.am = 0
        self.at = 0
        self.R = 10
        self.Va = 306
        self.Vt = 310
        # 定义动作空间和观察空间
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self):
        self.x1 = 0
        self.x2 = 0
        self.am = 0
        self.R = 10
        return np.array([self.x1, self.x2], dtype=np.float32)

    def step(self, action):
        # 解包动作值
        c, k, epsilon = action[0]

        # 计算状态导数
        x1_dot = self.x2
        x2_dot = -(2 * self.R_dot / self.R) * self.x2 - (self.am / self.R) + (self.at / self.R)

        # 更新状态
        self.x1 += x1_dot
        self.x2 += x2_dot

        # 计算滑动面和横向过载
        S = c * self.x2
        S_dot = -k * S - epsilon * np.sign(S)
        U = -self.R / self.gravity * (-k * self.x2 - epsilon * np.sign(c * self.x2) - c * (-2 * self.R_dot / self.R * self.x2 - self.gravity / self.R + self.at / self.R))

        # 计算奖励
        reward = -self.R**2

        # 判断回合是否结束（根据您的环境逻辑自定义）
        done = False  # 例如，根据某些条件判断回合结束

        # 可选的额外信息
        info = {}

        # 返回下一个状态、奖励、结束标志和额外信息
        next_state = np.array([self.x1, self.x2], dtype=np.float32)
        return next_state, reward, done, info


    def render(self, mode='console'):
        if mode == 'console':
            print(f"State: {self.x1}, {self.x2}")
        else:
            raise NotImplementedError("Only console rendering is implemented.")

    @property
    def R_dot(self):
        # Define how R_dot is computed based on the current state
        # This is a placeholder for the actual computation
        return -self.Vt * np.sin(self.x1) + self.Va * np.sin(self.x1)

    @property
    def gravity(self):
        # Define the gravitational constant
        return 9.81


env = SMCenv()
n_states = env.observation_space.shape[0]  # 状态数 2
n_actions = env.action_space.shape[0]  # 动作数 3
action_bound = env.action_space.high[0]  # 动作的最大值 1.0


# -------------------------------------- #
# 模型构建
# -------------------------------------- #

# 经验回放池实例化
replay_buffer = ReplayBuffer(capacity=args.buffer_size)
# 模型实例化
agent = DDPG(n_states = n_states,  # 状态数
             n_hiddens = args.n_hiddens,  # 隐含层数
             n_actions = n_actions,  # 动作数
             action_bound = action_bound,  # 动作最大值
             sigma = args.sigma,  # 高斯噪声
             actor_lr = args.actor_lr,  # 策略网络学习率
             critic_lr = args.critic_lr,  # 价值网络学习率
             tau = args.tau,  # 软更新系数
             gamma = args.gamma,  # 折扣因子
             device = device
            )

# -------------------------------------- #
# 模型训练
# -------------------------------------- #

return_list = []  # 记录每个回合的return
mean_return_list = []  # 记录每个回合的return均值

for i in tqdm(range(10)):  # 迭代10回合
    episode_return = 0  # 累计每条链上的reward
    state = env.reset()  # 初始时的状态
    # print("State shape:", state.shape)  # 打印 state 的形状
    done = False  # 回合结束标记

    while not done:
        # 获取当前状态对应的动作
        action = agent.take_action(state)
        # 环境更新
        next_state, reward, done, _ = env.step(action)
        # 更新经验回放池
        replay_buffer.add(state, action, reward, next_state, done)
        # 状态更新
        state = next_state
        # 累计每一步的reward
        episode_return += reward

        # 如果经验池超过容量，开始训练
        if replay_buffer.size() > args.min_size:
            # 经验池随机采样batch_size组
            s, a, r, ns, d = replay_buffer.sample(args.batch_size)
            # print(a.shape)
            # 构造数据集
            transition_dict = {
                'states': s,
                'actions': a,
                'rewards': r,
                'next_states': ns,
                'dones': d,
            }
            # 模型训练
            agent.update(transition_dict)
    
    # 保存每一个回合的回报
    return_list.append(episode_return)
    mean_return_list.append(np.mean(return_list[-10:]))  # 平滑

    # 打印回合信息
    print(f'iter:{i}, return:{episode_return}, mean_return:{np.mean(return_list[-10:])}')



# -------------------------------------- #
# 绘图
# -------------------------------------- #

x_range = list(range(len(return_list)))

plt.subplot(121)
plt.plot(x_range, return_list)  # 每个回合return
plt.xlabel('episode')
plt.ylabel('return')
plt.subplot(122)
plt.plot(x_range, mean_return_list)  # 每回合return均值
plt.xlabel('episode')
plt.ylabel('mean_return')
