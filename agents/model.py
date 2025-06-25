import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


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
            random_flag = False
        else:
            # 贪婪策略，但使用softmax生成更好的概率分布
            temperature = 0.1  # 增加温度使分布更平滑
            scaled_q_values = q_values / temperature
            exp_q = np.exp(scaled_q_values - np.max(scaled_q_values))  # 减去最大值以避免数值溢出
            probs = exp_q / np.sum(exp_q)
            random_flag = True

        return probs, random_flag
        
    def train(self, observation, action, reward, next_observation, done,
              target_network, optimizer, gamma=0.99, count=0, update_freq=1000, tau=0.005, losses=None, clip_grad_norm=1.0):
        """训练网络
        
        参数:
            observation: 当前观察状态
            action: 采取的动作
            reward: 获得的奖励
            next_observation: 下一个观察状态
            done: 是否结束
            target_network: 目标网络
            optimizer: 优化器
            gamma: 折扣因子
            count: 当前训练计数
            update_freq: 目标网络更新频率
            tau: 软更新系数
            losses: 损失记录列表
            clip_grad_norm: 梯度裁剪的最大范数，None表示不进行梯度裁剪
        
        返回:
            loss: 当前批次的损失值
        """
        device = next(self.parameters()).device
        
        # 将数据转换为tensor并移动到对应设备
        observation = torch.FloatTensor(observation).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_observation = torch.FloatTensor(next_observation).to(device)
        done = torch.FloatTensor(done).to(device)
        
        # 计算当前Q值
        q_values = self.forward(observation)
        next_q_values = target_network.forward(next_observation)
        argmax_actions = self.forward(next_observation).max(1)[1].detach()
        next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        expected_q_value = reward + gamma * ( 1 - done) * next_q_value
        
        # 计算损失
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        if losses is not None:
            losses.append(loss.item())
        
        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪 - 防止梯度爆炸
        if clip_grad_norm is not None and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
        
        optimizer.step() 
        
        # 定期更新目标网络
        if count % update_freq == 0:
            # 软更新目标网络
            for target_param, param in zip(target_network.parameters(), self.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        return loss.item()


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
    
    def train(self, observation, action, policy_accuracies, optimizer, clip_grad_norm=1.0):
        """训练策略网络
        
        参数:
            observation: 观察状态
            action: 实际采取的动作
            policy_accuracies: 策略准确率记录列表
            optimizer: 优化器
            clip_grad_norm: 梯度裁剪的最大范数，None表示不进行梯度裁剪
        
        返回:
            loss: 当前批次的损失值
        """
        device = next(self.parameters()).device
        
        # 将数据转换为tensor并移动到对应设备
        observation = torch.FloatTensor(observation).to(device)
        action = torch.LongTensor(action).to(device)

        # 计算策略概率
        probs = self.forward(observation)
        log_prob = probs.gather(1, action.unsqueeze(1)).squeeze(1).log()
        
        # 计算准确率 (预测的最高概率动作与实际动作匹配的比例)
        pred_actions = probs.argmax(dim=1)
        accuracy = (pred_actions == action).float().mean().item()
        policy_accuracies.append(accuracy)
        
        # 计算损失 (负对数似然)
        loss = -log_prob.mean()
        
        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪 - 防止梯度爆炸
        if clip_grad_norm is not None and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        return loss.item()