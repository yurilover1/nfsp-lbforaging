# 5*5 grid hidden_size (128,128)
# 6*6 grid hidden_size (128,128,128)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleAgent2(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, device='cpu'):
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

    def select_action(self, s, deterministic=False):
        with torch.no_grad():
            act_probs = self.forward(s)
            if deterministic:
                actions = torch.argmax(act_probs, dim=-1)
            else:
                act_dist = torch.distributions.Categorical(act_probs)  
                actions = act_dist.sample()
        return actions.detach().tolist(), act_probs.detach()
    
    
    def load_model(self, path):
        model_data = torch.load(path)
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
        obs = obs_dict['obs']
        
        # 处理观测
        if isinstance(obs, (list, tuple)):
            # 如果是元组，则取第一个元素（针对多智能体环境）
            obs = obs[0]
        
        # 转换为Tensor
        if isinstance(obs, np.ndarray):
            # 如果是3层观测格式 [3, 5, 5]，则需要展平
            if len(obs.shape) == 3:
                obs = obs[0] * 1 + obs[1] * 2 + obs[2] * 3
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        else:
            obs = torch.FloatTensor([obs]).unsqueeze(0).to(self.device)
        
        # 选择动作
        action, _ = self.select_action(obs)
        
        # 确保返回标量动作
        if isinstance(action, list) and len(action) == 1:
            action = action[0]
        
        return action
    
    def choose_policy_mode(self):
        """兼容NFSP接口"""
        pass
