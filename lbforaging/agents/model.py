import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class dueling_ddqn(nn.Module):
    """双重网络(Dueling Network)结构的深度Q网络"""
    
    def __init__(self, state_size, action_size, hidden_units=256, num_layers=3, activation='LeakyReLU'):
        """初始化参数和构建模型"""
        super(dueling_ddqn, self).__init__()
        
        # 处理hidden_units参数，确保其为整数
        if isinstance(hidden_units, (list, tuple)):
            hidden_size = hidden_units[0] if hidden_units else 256
        else:
            hidden_size = hidden_units
            
        # 特征提取层
        self.feature = self._build_feature_layers(state_size, hidden_size, num_layers, activation)
        
        # 优势流 - 简化网络结构
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            self._get_activation(activation, 0.1),
            nn.Dropout(0.05),
            
            nn.Linear(hidden_size//2, action_size)
        )
        
        # 状态值流 - 简化网络结构
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            self._get_activation(activation, 0.1),
              nn.Dropout(0.05),
            
            nn.Linear(hidden_size//2, 1)
        )
        
        # 使用Xavier初始化方法
        self._init_weights()

    def _get_activation(self, activation_name, negative_slope=0.1):
        """获取激活函数实例"""
        if activation_name == 'LeakyReLU':
            return nn.LeakyReLU(negative_slope)
        elif activation_name == 'ReLU':
            return nn.ReLU()
        elif activation_name == 'GELU':
            return nn.GELU()
        elif activation_name == 'ELU':
            return nn.ELU()
        elif activation_name == 'SELU':
            return nn.SELU()
        elif activation_name == 'Tanh':
            return nn.Tanh()
        else:
            # 默认使用LeakyReLU
            return nn.LeakyReLU(negative_slope)

    def _repeat_layer(self, input_size, output_size, activation='LeakyReLU', use_layer_norm=True, dropout_rate=0.1):
        """创建一个可重复使用的网络层，包括线性层、归一化、激活函数和Dropout"""
        layers = [nn.Linear(input_size, output_size)]
        
        if use_layer_norm:
            layers.append(nn.LayerNorm(output_size))
            
        layers.append(self._get_activation(activation))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
            
        return nn.Sequential(*layers)
        
    def _build_feature_layers(self, input_size, hidden_size, num_layers, activation):
        """构建特征提取层，支持动态层数调整"""
        if num_layers < 1:
            num_layers = 1  # 至少有一层
        elif num_layers > 10:
            num_layers = 10  # 最多10层
            
        layers = []
        # 第一层，输入层到隐藏层
        layers.append(self._repeat_layer(input_size, hidden_size, activation))
        
        # 中间层，隐藏层到隐藏层
        for _ in range(num_layers - 1):
            layers.append(self._repeat_layer(hidden_size, hidden_size, activation))
            
        return nn.Sequential(*layers)
        
    def _init_weights(self):
        """使用Xavier初始化权重，提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
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
            temperature = 0.1  # 增加温度使分布更平滑
            scaled_q_values = q_values / temperature
            exp_q = np.exp(scaled_q_values - np.max(scaled_q_values))  # 减去最大值以避免数值溢出
            probs = exp_q / np.sum(exp_q)
            
        return probs


class policy(nn.Module):
    """策略网络，用于NFSP的监督学习部分"""
    
    def __init__(self, state_size, action_size, hidden_units=256, num_layers=5, activation='LeakyReLU'):
        """初始化参数和构建模型"""
        super(policy, self).__init__()
        
        # 处理hidden_units参数，确保其为整数
        if isinstance(hidden_units, (list, tuple)):
            hidden_size = hidden_units[0] if hidden_units else 256
        else:
            hidden_size = hidden_units
        
        # 使用_repeat_layer构建网络
        layers = []
        # 输入层
        layers.append(self._repeat_layer(state_size, hidden_size, activation))
        # 中间层1
        layers.append(self._repeat_layer(hidden_size, hidden_size * 2, activation))
        # 中间层2
        layers.append(self._repeat_layer(hidden_size * 2, hidden_size, activation))
        # 后续层，根据num_layers添加
        for _ in range(max(0, num_layers - 3)):  # 已经有3层了
            layers.append(self._repeat_layer(hidden_size, hidden_size, activation))
        # 输出前的隐藏层
        layers.append(self._repeat_layer(hidden_size, hidden_size//2, activation, dropout_rate=0.05))
        # 输出层
        layers.append(nn.Linear(hidden_size//2, action_size))
        
        self.net = nn.Sequential(*layers)
        
        # 使用Xavier初始化权重
        self._init_weights()
        
    def _repeat_layer(self, input_size, output_size, activation='LeakyReLU', use_layer_norm=True, dropout_rate=0.1):
        """创建一个可重复使用的网络层，包括线性层、归一化、激活函数和Dropout"""
        layers = [nn.Linear(input_size, output_size)]
        
        if use_layer_norm:
            layers.append(nn.LayerNorm(output_size))
            
        # 获取激活函数
        if activation == 'LeakyReLU':
            layers.append(nn.LeakyReLU(0.1))
        elif activation == 'ReLU':
            layers.append(nn.ReLU())
        elif activation == 'GELU':
            layers.append(nn.GELU())
        elif activation == 'ELU':
            layers.append(nn.ELU())
        elif activation == 'SELU':
            layers.append(nn.SELU())
        elif activation == 'Tanh':
            layers.append(nn.Tanh())
        else:
            # 默认使用LeakyReLU
            layers.append(nn.LeakyReLU(0.1))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
            
        return nn.Sequential(*layers)
        
    def _init_weights(self):
        """使用Xavier初始化权重，提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
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