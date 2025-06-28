# 5*5 grid hidden_size (128,128)
# 6*6 grid hidden_size (128,128,128)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        legal_actions = [0]  # 只使用NONE动作(索引0)，而不是所有动作
    
    # 创建掩码张量
    mask = torch.zeros_like(probs)
    
    # 设置合法动作的掩码
    for action in legal_actions:
        if 0 <= action < action_dim:
            mask[:, action] = 1.0
    
    # 应用掩码，将不合法动作的概率设为0
    masked_probs = probs * mask
    
    # 处理可能存在的NaN或inf值
    masked_probs = torch.nan_to_num(masked_probs, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 计算每个样本的概率和
    prob_sums = masked_probs.sum(dim=1, keepdim=True)
    
    zero_indices = (prob_sums <= 1e-10).squeeze()
    
    valid_indices = ~zero_indices
    if valid_indices.any():
        masked_probs[valid_indices] = masked_probs[valid_indices] / prob_sums[valid_indices]
    
    # 处理极端情况：如果某个样本的所有概率仍然为0
    if zero_indices.any():
        # 默认使用第一个动作（通常是NONE动作）
        if legal_actions:
            uniform_probs = torch.zeros_like(probs)
            legal_actions_count = len(legal_actions)
            
            for action in legal_actions:
                if 0 <= action < action_dim:
                    uniform_probs[:, action] = 1.0 / legal_actions_count
            
            if zero_indices.dim() == 0:
                if zero_indices:
                    masked_probs = uniform_probs
            else:
                masked_probs[zero_indices] = uniform_probs[zero_indices]
        else:
            # 如果没有合法动作，使用第一个动作
            if zero_indices.dim() == 0:
                if zero_indices:
                    masked_probs[:, 0] = 1.0
            else:
                masked_probs[zero_indices, 0] = 1.0
    
    return masked_probs

class SimpleAgent2(nn.Module):
    def __init__(self, input_dim: object, hidden_dims: object, output_dim: object, device: object = 'cpu') -> None:
        super(SimpleAgent2, self).__init__()
        self.device = device
        self.hidden_dims = hidden_dims
        self.name = "SimpleAgent2"  # 添加名称属性

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
        x = s.to(self.device)
        for layer in self.fc1s:
            x = layer(x)
        act_dist = self.output_layer(x)
        act_probs = F.softmax(act_dist, dim=-1)
        return act_probs

    def select_action(self, s, legal_actions, deterministic=False):
        with torch.no_grad():
            act_probs = action_mask(self.forward(s), legal_actions)
            if deterministic:
                actions = torch.argmax(act_probs, dim=-1)
            else:
                act_dist = torch.distributions.Categorical(act_probs)
                actions = act_dist.sample()
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
        obs = obs_dict['obs'] if isinstance(obs_dict, dict) else obs_dict
        legal_actions = obs_dict['actions']
        # 处理观测
        if isinstance(obs, (list, tuple)):
            # 如果是元组，则取第一个元素（针对多智能体环境）
            obs = obs[0]
        
        # 转换为Tensor（最高效的方式）
        obs = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)

        # 选择动作
        action, _ = self.select_action(obs, legal_actions, deterministic=True)
        action = int(action)
        
        return action
    
    def choose_policy_mode(self):
        """兼容NFSP接口"""
        pass
