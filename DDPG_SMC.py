import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple

#dt 用于求微分，类似于单片机上的处理效果，假设是10k
dt = 1e-4


#Create the environment for SMC
import numpy as np

class SMCenv:
    def __init__(self):
        # Define constants based on the servo valve model
        # These should be defined based on the mathematical model provided in the paper
        self.mass = 0.1  # Total mass of the moving parts
        self.damping_coefficient = 30  # Damping coefficient
        self.spring_stiffness = 1e4  # Spring stiffness
        self.pressure_system = 0  # System pressure
        self.valve_spring_preload = 6.5e-3  # Preload of the spring
        self.initial_spool_position = 0  # Initial position of the spool
        self.force_coefficient = 40  # Coefficient relating current to force for the solenoid
        self.position = 0 #position at present

        # Define the state space as per the model in the paper
        self.state = np.array([self.initial_spool_position, 0])  # Initial state

        # Define bounds for uncertainty and disturbance as per the paper's model
        self.disturbance_lower_bound = -10
        self.disturbance_upper_bound = 10

    def reset(self):
        # Reset the environment state back to initial conditions
        self.state = np.array([self.initial_spool_position, 0])
        self.position = 0
        return self.state

    def step(self, action,desired_state):
        # Apply the action to the environment and update the state
        # The action would be the control input to the valve

        # Implement the valve dynamics here based on the equations from the paper
        # Calculate the new state based on the current state and the action
        # Update the disturbance observer and sliding mode controller as per the equations (10), (11), (12)...

        # Example placeholder for new state calculation, this should be replaced by actual model equations
        x_1 = self.state[0]
        x_2 = self.state[1]
        xr_1 = desired_state
        xr_2 = desired_state / dt
        
        
        new_x_2 = x_1/dt
        u = x_2/dt
        
        new_position = self.state[0] + action - self.damping_coefficient*self.state[1]
        new_velocity = self.state[1] + action/self.mass

        # Update state
        self.state = np.array([new_position, new_velocity])

        # Calculate reward (or use any other metrics as per the research goals)
        reward = -np.square(self.state - self.desired_state).sum()  # Example placeholder

        # Check if episode is done
        done = np.abs(self.state[0] - self.desired_state[0]) < self.tolerance  # Example placeholder

        return self.state, reward, done, {}

    # Additional methods to implement the mathematical model of the valve as per the paper
    # Including the disturbance observer and sliding mode controller
    # ...

# Usage
env = SMCenv()
state = env.reset()
for _ in range(1000):  # Example number of steps
    action = np.random.random(1)  # Example action, replace with actual control algorithm
    next_state, reward, done, _ = env.step(action)
    if done:
        break


env = SMCenv

#Create DDPG network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # 定义Actor网络结构
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        # 定义如何从状态生成动作
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) #根据SMC参数进行调整
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 定义Critic网络结构
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        # 定义如何评估状态和动作
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

#initialize Actor and critic
state_dim = 1 # 环境状态维度 -e**2
action_dim = 3 # 动作维度（SMC参数数量） c,k,epsi

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

#training circle
num_episodes = 100
max_steps = 200  # 每个episode的最大步数
batch_size = 64  # 从经验回放中采样的批量大小
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

#training circle    
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    total_step = total_step + 1
    
    if total_step <= (num_episodes//2) :
        desired_state = 0
    else:
        desired_state = 1e-3

    for step in range(max_steps):
        # 生成动作
        action = actor(state).detach().numpy()
        
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
            target_actions = Actor(next_states)
            target_q = Critic(next_states, target_actions)
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
