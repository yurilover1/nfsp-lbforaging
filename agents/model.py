import torch
import torch.nn as nn
import torch.nn.functional as F


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

class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, output_dim=6):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LeakyReLU(0.1))
            prev_dim = h
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(prev_dim, output_dim)
    def forward(self, x, temperature=1.0):
        x = self.net(x)
        logits = self.out(x)
        if temperature != 1.0:
            logits = logits / temperature
        return F.softmax(logits, dim=-1)

class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LeakyReLU(0.1))
            prev_dim = h
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(prev_dim, 1)
    def forward(self, x):
        x = self.net(x)
        return self.out(x)