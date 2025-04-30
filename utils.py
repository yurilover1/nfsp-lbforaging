import numpy as np
import time
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import logging
from tqdm import tqdm

# 添加从nfsp_run.py移动的函数
logger = logging.getLogger(__name__)

def calculate_state_size(env):
    """计算环境的状态大小"""
    try:
        # 获取一个示例观测
        obs, _ = env.reset()
        
        # 如果观测是元组（多智能体环境），取第一个智能体的观测
        if isinstance(obs, tuple):
            first_obs = obs[0]
        else:
            first_obs = obs
            
        # 检查观测形状以判断观测模式
        if isinstance(first_obs, np.ndarray):
            # 检查是否为三维数组
            if len(first_obs.shape) == 3:
                # 检查是否为三层观测模式 [3, 5, 5]
                if first_obs.shape[0] == 3 and first_obs.shape[1] == 5 and first_obs.shape[2] == 5:
                    print(f"检测到三层观测模式: 形状={first_obs.shape}")
                    return 3 * 5 * 5  # 返回展平后的大小
                
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

def plot_training_curve(history, num_episodes, eval_interval, nfsp_agents=None):
    """绘制训练曲线，包括四幅图：监督学习损失、策略准确率、强化学习损失和队伍总奖励"""
    # 数据预处理部分 - 将所有数据处理逻辑放在绘图之前
    
    # 预处理变量初始化
    smooth_losses = None
    smooth_acc = None
    smooth_rl = None
    team_rewards = None
    smooth_rewards = None
    
    # 只使用第一个NFSP智能体数据
    if nfsp_agents and len(nfsp_agents) > 0:
        agent = nfsp_agents[0]
        
        # 处理监督学习损失
        if len(agent.losses) > 0:
            losses = np.array(agent.losses)
            
            # 降采样
            if len(losses) > 1000:
                step = len(losses) // 1000
                losses = losses[::step]
            
            # 平滑处理
            window_size = min(20, len(losses))
            if window_size > 1:
                smooth_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        
        # 处理策略准确率
        if hasattr(agent, 'policy_accuracies') and len(agent.policy_accuracies) > 0:
            accuracies = np.array(agent.policy_accuracies)
            
            # 降采样
            if len(accuracies) > 1000:
                step = len(accuracies) // 1000
                accuracies = accuracies[::step]
            
            # 平滑处理
            window_size = min(20, len(accuracies))
            if window_size > 1:
                smooth_acc = np.convolve(accuracies, np.ones(window_size)/window_size, mode='valid')
        
        # 处理强化学习损失
        if len(agent.RLlosses) > 0:
            rl_losses = np.array(agent.RLlosses)
            
            # 降采样
            if len(rl_losses) > 1000:
                step = len(rl_losses) // 1000
                rl_losses = rl_losses[::step]
            
            # 平滑处理
            window_size = min(20, len(rl_losses))
            if window_size > 1:
                smooth_rl = np.convolve(rl_losses, np.ones(window_size)/window_size, mode='valid')
    
    # 处理团队奖励数据
    raw_rewards = history['episode_rewards']
    if len(raw_rewards) > 0:
        # 检查数据格式并转换为numpy数组
        if isinstance(raw_rewards[0], (list, np.ndarray)):
            # 多智能体情况，计算团队总奖励
            rewards = np.array([reward for reward in raw_rewards])
            team_rewards = rewards.sum(axis=1)
            
            # 平滑处理团队奖励
            window_size = min(50, len(team_rewards) // 10)  # 使用更大的窗口来平滑奖励曲线
            if window_size > 1:
                smooth_rewards = np.convolve(team_rewards, np.ones(window_size)/window_size, mode='valid')
        else:
            # 单智能体情况
            team_rewards = np.array(raw_rewards)
    
    # 创建图形
    plt.figure(figsize=(16, 16))
    
    # 1. 监督学习损失 (左上角)
    plt.subplot(2, 2, 1)
    if smooth_losses is not None:
        plt.plot(smooth_losses, 'k-', linewidth=2, label='SL Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Supervised Learning Loss')
    plt.grid(True)
    plt.legend()
    
    # 2. 策略准确率 (右上角)
    plt.subplot(2, 2, 2)
    if smooth_acc is not None:
        plt.plot(smooth_acc, 'k-', linewidth=2, label='Policy Accuracy')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Policy Accuracy')
    plt.grid(True)
    plt.legend()
    
    # 3. 强化学习损失 (左下角)
    plt.subplot(2, 2, 3)
    if smooth_rl is not None:
        plt.plot(smooth_rl, 'k-', linewidth=2, label='RL Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Reinforcement Learning Loss')
    plt.grid(True)
    plt.legend()
    
    # 4. 队伍总奖励 (右下角)
    plt.subplot(2, 2, 4)
    if team_rewards is not None:
        window_size = min(50, len(team_rewards) // 10)
        # 画出原始数据（较浅的颜色）
        plt.plot(np.arange(len(team_rewards)), team_rewards, 'b-', alpha=0.3, label='Raw Team Reward')
        
        # 如果有平滑数据，也画出平滑后的数据
        if smooth_rewards is not None:
            plt.plot(np.arange(len(smooth_rewards)) + window_size//2, smooth_rewards, 'b-', 
                     linewidth=2, label='Smoothed Team Reward')
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Team Total Reward')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./results/training_curve.png')
    plt.close()

def save_history(history, nfsp_agents):
    """保存训练历史数据"""
    # 创建保存目录
    os.makedirs("./results", exist_ok=True)
    
    # 准备保存数据
    data_to_save = {
        'episode_rewards': history['episode_rewards'],
        'eval_rewards': history['eval_rewards'],
        'eval_episodes': history['eval_episodes'],
    }
    
    # 添加可利用度数据（如果存在）
    if 'exploitability' in history and history['exploitability']:
        data_to_save['exploitability'] = history['exploitability']
    
    # 添加监督学习损失
    if 'sl_losses' in history and history['sl_losses']:
        data_to_save['sl_losses'] = history['sl_losses']
        
    # 添加强化学习损失
    if 'rl_losses' in history and history['rl_losses']:
        data_to_save['rl_losses'] = history['rl_losses']
        
    # 添加策略准确率
    if 'policy_accuracies' in history and history['policy_accuracies']:
        data_to_save['policy_accuracies'] = history['policy_accuracies']
    
    # 保存为numpy格式
    try:
        np.savez('./results/training_history.npz', **data_to_save)
        print("训练历史保存成功: ./results/training_history.npz")
    except Exception as e:
        print(f"保存训练历史时出错: {e}")
        import traceback
        traceback.print_exc()

