import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np

from agents.partner_agent import SimpleAgent2

# 添加从nfsp_run.py移动的函数
logger = logging.getLogger(__name__)


def teammate_generate(action_size, device, id=None):
    if id is None:
        id = random.randint(0, 7)
    model_path = f'./partners/agents_for_5x5/agent_{id}_1.pt'
    teammate_agent = SimpleAgent2(
        input_dim=12,  # 使用正确的输入维度12而不是state_size
        hidden_dims=[128, 128],
        output_dim=action_size,
        device=device
    )
    teammate_agent.load_model(model_path)
    return teammate_agent

def calculate_state_size(env):
    """计算环境的状态大小"""
    try:
        # 获取一个示例观测
        obs, _ = env.reset()
        
        # 如果观测是元组（多智能体环境），取第一个智能体的观测
        first_obs = _extract_obs(obs)
            
        # 检查观测形状以判断观测模式
        if isinstance(first_obs, np.ndarray):
            # 检查是否为三维数组
            if len(first_obs.shape) == 3:
                # 检查是否为普通网格观测模式
                print(f"检测到网格观测模式: 形状={first_obs.shape}")
                return first_obs.size  # 返回展平后的大小
            
            # 普通一维观测
            print(f"检测到普通观测模式: 形状={first_obs.shape}")
            return first_obs.size
            
        # 尝试从环境获取observation_space
        if hasattr(env, 'observation_space'):
            # 尝试获取第一个观测空间的形状
            obs_shape = env.observation_space[0].shape
            print(f"从observation_space获取的形状: {obs_shape}")
            
            # 如果是多维形状
            if isinstance(obs_shape, tuple) and len(obs_shape) > 1:
                # 计算总大小
                return np.prod(obs_shape)
            elif isinstance(obs_shape, tuple) and len(obs_shape) > 0:
                return obs_shape[0]
        
        # 默认状态大小
        print("无法确定状态大小，使用默认值100")
        return 100
        
    except Exception as e:
        print(f"计算状态大小时出错: {e}")
        # 默认状态大小
        return 100

def _extract_obs(obs):
    if isinstance(obs, tuple):
        return obs[0]
    return obs

