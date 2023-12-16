import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple
from tqdm import tqdm

#dt 用于求微分，类似于单片机上的处理效果，假设是10k
dt = 1e-4



import numpy as np

class SMCenv:
    def __init__(self):
        # Define initial conditions and constants
        # These values will need to be updated according to the specific scenario you are simulating
        self.x1 = 0  # qβ initial condition
        self.x2 = 0  # q˙β initial condition
        self.am = 0  # nz*c*g, initial lateral overload
        self.at = 0  # Target acceleration
        self.R = 10  # Initial relative distance, for example
        self.Va = 306  # Speed of the aircraft
        self.Vt = 310  # Speed of the target

    def reset(self):
        # Reset the state of the environment to the initial conditions
        self.x1 = 0  # Reset qβ
        self.x2 = 0  # Reset q˙β
        self.am = 0  # Reset lateral overload
        self.R = 10  # Reset relative distance
        # Return the initial state as a numpy array
        return np.array([self.x1, self.x2])

    def step(self, action):
        # Update the state based on the action
        # action is expected to be the array [c, k, epsilon]
        c, k, epsilon = action
        
        # Compute the state derivatives
        x1_dot = self.x2
        x2_dot = -(2 * self.R_dot / self.R) * self.x2 - (self.am / self.R) + (self.at / self.R)

        # Update the state
        self.x1 += x1_dot
        self.x2 += x2_dot

        # Compute the sliding surface
        S = c * self.x2
        S_dot = -k * S - epsilon * np.sign(S)

        # Compute the lateral overload
        U = -self.R / self.gravity * (-k * self.x2 - epsilon * np.sign(c * self.x2) - c * (-2 * self.R_dot / self.R * self.x2 - self.gravity / self.R + self.at / self.R))

        # Compute reward, next_state, done, and any additional info
        reward = -self.R**2
        next_state = np.array([self.x1, self.x2])
        done = False  # Define your own condition for 'done'
        info = {}

        return next_state, reward, done, info

    @property
    def R_dot(self):
        # Define how R_dot is computed based on the current state
        # This is a placeholder for the actual computation
        return -self.Vt * np.sin(self.x1) + self.Va * np.sin(self.x1)

    @property
    def gravity(self):
        # Define the gravitational constant
        return 9.81

# Usage
env = SMCenv()
state = env.reset()
for _ in range(1000):  # Example number of steps
    action = np.random.random(3)  # Example action, replace with actual control algorithm
    next_state, reward, done, _ = env.step(action)
    if done:
        break



#Create DDPG network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # 定义Actor网络结构
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        # 定义如何从状态生成动作
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = 5*torch.tanh(self.fc3(x)) #根据SMC参数进行调整
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 定义Critic网络结构
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        # 定义如何评估状态和动作
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

#initialize Actor and critic
state_dim = 2 # 环境状态维度 -e**2
action_dim = 3 # 动作维度（SMC参数数量） c,k,epsi

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)

actor_optimizer = optim.Adam(actor.parameters(), lr=5e-3)
critic_optimizer = optim.Adam(critic.parameters(), lr=5e-3)

#training circle
num_episodes = 100
max_steps = 200  # 每个episode的最大步数
batch_size = 128  # 从经验回放中采样的批量大小
gamma = 0.6  #damping coefficient 0-1之前使时间越晚，奖励越少
total_step = 0

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# 初始化经验回放缓冲区
replay_buffer = ReplayBuffer(capacity=10000)
# 定义Transition具名元组，它有五个字段
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

#创建高斯噪声
class GaussianNoise:
    def __init__(self, action_dimension, mu=0, sigma=0.1, sigma_min=0.01, decay_rate=0.995):
        self.action_dimension = action_dimension
        self.mu = mu
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.decay_rate = decay_rate

    def reset(self):
        self.sigma = self.sigma * self.decay_rate
        self.sigma = max(self.sigma, self.sigma_min)

    def get_noise(self):
        return np.random.normal(self.mu, self.sigma, self.action_dimension)

#初始化噪声
noise = GaussianNoise(action_dim)

#training circle    
for episode in tqdm(range(num_episodes)):
    state = env.reset()
    episode_reward = 0
    total_step = total_step + 1
    

    for step in tqdm(range(max_steps)):
        # 转换state为PyTorch张量
        # state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state_tensor = torch.tensor([state], dtype=torch.float32)
        
        # 生成动作
        action = actor(state_tensor).detach().numpy().squeeze()
        noise_sample = noise.get_noise()
        action = action+noise_sample
        
        # 与环境交互
        next_state, reward, done, _ = env.step(action)
        
        # 存储经验
        replay_buffer.push(state, action, reward, next_state, done)

        # 准备下一个状态
        state = next_state
        episode_reward += reward

        # 如果经验回放缓冲区足够大，开始学习
        if len(replay_buffer) > batch_size:
            # 从缓冲区采样一个批量的经验
            transitions = replay_buffer.sample(batch_size)
            batch = Transition(*zip(*transitions))

            # 提取变量
            states = torch.tensor(np.array(batch.state), dtype=torch.float32)
            actions = torch.tensor(np.array(batch.action), dtype=torch.float32)
            rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32)
            next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
            dones = torch.tensor(np.array(batch.done), dtype=torch.float32)

            # 更新Critic
            target_actions = actor(next_states)
            target_q = critic(next_states, target_actions)
            expected_q = rewards + (gamma * target_q * (1 - dones))
            current_q = critic(states, actions)
            critic_loss = F.mse_loss(current_q, expected_q.detach())

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # 更新Actor
            predicted_actions = actor(states)
            actor_loss = -critic(states, predicted_actions).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

        if done:
            break

    print(f"Episode: {episode}, Total Reward: {episode_reward}")
