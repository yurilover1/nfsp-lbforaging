import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class dueling_ddqn(nn.Module):
    """双重网络(Dueling Network)结构的深度Q网络"""
    
    def __init__(self, state_size, action_size, hidden_units=128):
        """初始化参数和构建模型"""
        super(dueling_ddqn, self).__init__()
        
        # 特征提取层 - 加深到5层并使用更大的隐藏单元
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units*2),
            nn.ReLU(),
            nn.Linear(hidden_units*2, hidden_units*2),
            nn.ReLU(),
            nn.Linear(hidden_units*2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU()
        )
        
        # 优势流 - 加深到5层
        self.advantage = nn.Sequential(
            nn.Linear(hidden_units, hidden_units*2),
            nn.ReLU(),
            nn.Linear(hidden_units*2, hidden_units*2),
            nn.ReLU(),
            nn.Linear(hidden_units*2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units//2),
            nn.ReLU(),
            nn.Linear(hidden_units//2, action_size)
        )
        
        # 状态值流 - 加深到5层
        self.value = nn.Sequential(
            nn.Linear(hidden_units, hidden_units*2),
            nn.ReLU(),
            nn.Linear(hidden_units*2, hidden_units*2),
            nn.ReLU(),
            nn.Linear(hidden_units*2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units//2),
            nn.ReLU(),
            nn.Linear(hidden_units//2, 1)
        )
        
        # 初始化权重，使用He初始化提高训练稳定性
        self._init_weights()
        
    def _init_weights(self):
        """使用He初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
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
    
    def __init__(self, state_size, action_size, hidden_units=128):
        """初始化参数和构建模型"""
        super(policy, self).__init__()
        
        # 加深策略网络以匹配Q网络的能力
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units*2),
            nn.ReLU(),
            nn.Linear(hidden_units*2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units//2),
            nn.ReLU(),
            nn.Linear(hidden_units//2, action_size)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """使用He初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """前向传播"""
        logits = self.net(x)
        return F.softmax(logits, dim=1)
    
    def act(self, state):
        """根据策略网络选择动作"""
        # 获取动作概率
        probs = self.forward(state).detach().cpu().numpy()[0]
        return probs 