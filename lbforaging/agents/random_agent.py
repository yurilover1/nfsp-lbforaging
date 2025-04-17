import random
import numpy as np
from lbforaging.agents import BaseAgent


class RandomAgent(BaseAgent):
    name = "Random Agent"

    def step(self, obs):
        """根据观察选择随机动作"""
        # 支持不同格式的输入
        if hasattr(obs, 'actions'):
            # 原始的BaseAgent格式
            return random.choice(obs.actions)
        elif isinstance(obs, dict) and 'actions' in obs:
            # 兼容字典格式
            return np.random.choice(obs['actions'])
        elif isinstance(obs, dict) and 'legal_actions' in obs:
            # 兼容其他字典格式
            return np.random.choice(obs['legal_actions'])
        else:
            # 默认所有动作都是合法的 (0-5)
            return np.random.choice(range(6))
    
    def eval_step(self, obs):
        """评估时的动作选择，返回动作和概率分布"""
        # 获取合法动作
        if hasattr(obs, 'actions'):
            actions = obs.actions
        elif isinstance(obs, dict) and 'actions' in obs:
            actions = obs['actions']
        elif isinstance(obs, dict) and 'legal_actions' in obs:
            actions = obs['legal_actions']
        else:
            # 默认所有动作都是合法的
            actions = list(range(6))
            
        # 选择一个随机动作
        action = np.random.choice(actions)
        
        # 创建均匀概率分布
        probs = np.zeros(6)  # 假设最多6个动作
        for i in actions:
            probs[i] = 1/len(actions)
            
        return action, probs
