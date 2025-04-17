import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class dueling_ddqn(nn.Module):
    """双重网络(Dueling Network)结构的深度Q网络"""
    
    def __init__(self, state_size, action_size, hidden_units=64):
        """初始化参数和构建模型"""
        super(dueling_ddqn, self).__init__()
        
        # 特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU()
        )
        
        # 优势流
        self.advantage = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_size)
        )
        
        # 状态值流
        self.value = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )
        
    def forward(self, x):
        """前向传播"""
        feature = self.feature(x)
        advantage = self.advantage(feature)
        value = self.value(feature)
        return value + advantage - advantage.mean(1, keepdim=True)
    
    def act(self, state, epsilon=0):
        """使用ε-贪婪策略选择动作"""
        # 获取Q值
        q_values = self.forward(state).detach().cpu().numpy()[0]
        
        # ε-贪婪动作选择
        if np.random.random() < epsilon:
            # 随机动作
            probs = np.ones(len(q_values)) / len(q_values)
        else:
            # 贪婪动作 (使用softmax生成概率分布)
            temperature = 0.1  # 低温度会使分布更接近于贪婪选择
            scaled_q_values = q_values / temperature
            exp_q = np.exp(scaled_q_values - np.max(scaled_q_values))  # 减去最大值以避免数值溢出
            probs = exp_q / np.sum(exp_q)
            
        return probs


class policy(nn.Module):
    """策略网络，用于NFSP的监督学习部分"""
    
    def __init__(self, state_size, action_size, hidden_units=64):
        """初始化参数和构建模型"""
        super(policy, self).__init__()
        
        # 网络层
        self.fc1 = nn.Linear(state_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, action_size)
        
    def forward(self, x):
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        """根据策略网络选择动作"""
        # 获取动作概率
        probs = self.forward(state).detach().cpu().numpy()[0]
        return probs 