import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class dueling_ddqn(nn.Module):
    """双重网络(Dueling Network)结构的深度Q网络"""
    
    def __init__(self, state_size, action_size, hidden_units=256):
        """初始化参数和构建模型"""
        super(dueling_ddqn, self).__init__()
        
        # 处理hidden_units参数，确保其为整数
        if isinstance(hidden_units, (list, tuple)):
            hidden_size = hidden_units[0] if hidden_units else 256
        else:
            hidden_size = hidden_units
            
        # 特征提取层 - 加深网络结构并使用更大的隐藏单元
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # 优势流 - 加深网络结构
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size//2, action_size)
        )
        
        # 状态值流 - 加深网络结构
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size//2, 1)
        )
        
        # 使用Kaiming初始化方法
        self._init_weights()
        
    def _init_weights(self):
        """使用He初始化权重，提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
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
            # 随机探索
            probs = np.ones(len(q_values)) / len(q_values)
        else:
            # 贪婪策略，但使用softmax生成更好的概率分布
            temperature = 0.05  # 降低温度使分布更接近贪婪选择
            scaled_q_values = q_values / temperature
            exp_q = np.exp(scaled_q_values - np.max(scaled_q_values))  # 减去最大值以避免数值溢出
            probs = exp_q / np.sum(exp_q)
            
        return probs


class policy(nn.Module):
    """策略网络，用于NFSP的监督学习部分"""
    
    def __init__(self, state_size, action_size, hidden_units=256):
        """初始化参数和构建模型"""
        super(policy, self).__init__()
        
        # 处理hidden_units参数，确保其为整数
        if isinstance(hidden_units, (list, tuple)):
            hidden_size = hidden_units[0] if hidden_units else 256
        else:
            hidden_size = hidden_units
        
        # 加深网络以匹配状态空间的复杂度
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size//2, action_size)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """使用He初始化权重，提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
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