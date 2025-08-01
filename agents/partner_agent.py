# 5*5 grid hidden_size (128,128)
# 6*6 grid hidden_size (128,128,128)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# from .utils import action_mask

def action_mask(probs, legal_actions):
    """
    过滤不合法的动作，只保留合法动作的概率

    参数:
        probs: 动作概率分布 (torch张量，形状为[batch_size, action_dim])
        legal_actions: 合法动作列表

    返回:
        过滤后的概率分布 (与输入相同形状的torch张量)
    """
    probs = probs.clone().detach()

    # 保存原始设备和形状信息
    _, action_dim = probs.shape

    # 确保legal_actions非空
    if not legal_actions:
        print("警告: legal_actions为空，仅使用NONE动作(索引0)")
        legal_actions = [0]

    # 使用softmax确保概率为正值，保持相对大小关系
    probs = torch.softmax(probs, dim=-1)

    # 创建掩码张量
    mask = torch.zeros_like(probs)

    # 设置合法动作的掩码
    for action in legal_actions:
        if 0 <= action < action_dim:
            mask[:, action] = 1.0

    # 应用掩码，将不合法动作的概率设为0
    masked_probs = probs * mask

    # 重新归一化，确保概率和为1
    prob_sums = masked_probs.sum(dim=1, keepdim=True)
    masked_probs = masked_probs / (prob_sums + 1e-10)  # 添加小值避免除零

    return masked_probs

def _extract_obs(obs):
    if isinstance(obs, tuple):
        return obs[0]
    return obs

class SimpleAgent2(nn.Module):
    def __init__(self, input_dim: object, hidden_dims: object, output_dim: object, device: object = 'cpu') -> None:
        super(SimpleAgent2, self).__init__()
        self.device = device
        self.hidden_dims = hidden_dims
        self.name = "SimpleAgent2"  # 添加名称属性
        
        # 添加动作历史记录
        self.action_history = []
        self.max_history_length = 1000  # 最大历史长度

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU()
            ))
            current_dim = hidden_dim

        self.fc1s = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.to(self.device)

    def forward(self, s):
        import torch
        import numpy as np
        if isinstance(s, dict):
            s = s['obs']
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float()
        x = s.to(self.device)
        for layer in self.fc1s:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def select_action(self, observation, is_training=False):
        obs = observation['obs'] if isinstance(observation, dict) else observation
        legal_actions = observation['actions'] if isinstance(observation, dict) and 'actions' in observation else None
        action, _ = self._select_action(obs, legal_actions, deterministic=True)
        
        # 记录动作
        action_value = int(action) if isinstance(action, torch.Tensor) else int(action)
        self.action_history.append(action_value)
        # 限制历史长度
        if len(self.action_history) > self.max_history_length:
            self.action_history = self.action_history[-self.max_history_length:]
            
        return int(action)

    def _select_action(self, s, legal_actions, deterministic=False):
        import numpy as np
        assert isinstance(legal_actions, (list, tuple, np.ndarray)), f"legal_actions type: {type(legal_actions)}, value: {legal_actions}"
        with torch.no_grad():
            act_probs = action_mask(self.forward(s), legal_actions)
            if deterministic:
                actions = torch.argmax(act_probs, dim=-1)
            else:
                act_dist = torch.distributions.Categorical(act_probs)
                actions = act_dist.sample()
        if isinstance(actions, torch.Tensor):
            action_value = int(actions.item())
        else:
            action_value = int(actions)
        assert action_value in list(legal_actions), f"[ERROR][SimpleAgent2._select_action] 采样动作{action_value}不在合法动作{legal_actions}中"
        return actions.detach(), act_probs.detach()
    
    
    def load_model(self, path):
        model_data = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(model_data)
    
    def step(self, obs_dict):
        """
        接收观测并返回动作
        
        参数:
            obs_dict: 包含观测和可用动作的字典
            
        返回:
            选择的动作
        """
        # 获取观测
        obs = _extract_obs(obs_dict)
        legal_actions = obs_dict['actions'] if isinstance(obs_dict, dict) and 'actions' in obs_dict else list(range(6))
        
        # 处理观测
        if isinstance(obs, (list, tuple)):
            # 如果是元组，则取第一个元素（针对多智能体环境）
            obs = obs[0]
        
        # 转换为Tensor（最高效的方式）
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        else:
            # 如果已经是tensor，确保形状正确
            obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 选择动作
        action, _ = self._select_action(obs, legal_actions, deterministic=True)
        action_value = int(action) if isinstance(action, torch.Tensor) else int(action)
        
        # 记录动作
        self.action_history.append(action_value)
        # 限制历史长度
        if len(self.action_history) > self.max_history_length:
            self.action_history = self.action_history[-self.max_history_length:]
        
        return action_value
    
    def get_last_actions(self):
        """获取最近的动作历史"""
        return self.action_history.copy()
    
    def choose_policy_mode(self):
        """兼容NFSP接口"""
        pass
