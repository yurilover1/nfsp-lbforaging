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

def plot_training_curve(history, num_episodes, eval_interval, main_agents=None, teamate_id=0, type='ppo'):
    """绘制训练曲线，显示评估奖励随批次变化的趋势以反映模型收敛速度与性能"""
    # 检查是否有评估数据
    if 'eval_rewards' not in history or len(history['eval_rewards']) == 0:
        print("没有评估数据，无法绘制评估曲线")
        return
     # 只使用第一个NFSP智能体数据
    if main_agents and len(main_agents) > 0:
        agent = main_agents[0]
        
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
        
    # 准备评估数据
    eval_rewards = history['eval_rewards']
    eval_batches = history['eval_batches']
    
    # 创建图形
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 1, 1)

    plt.plot(smooth_losses, 'b-o', linewidth=2, label='smooth SL losses')
    plt.title('Trends in assessment incentives by batch', fontsize=16)
    plt.xlabel('batch', fontsize=14)
    plt.ylabel('losses', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.subplot(2, 1, 2)
    plt.plot(eval_batches, eval_rewards, 'r-o', linewidth=2, label='agent rewards')
    
    # 设置图表标题和标签
    plt.title('Trends in assessment incentives by batch', fontsize=16)
    plt.xlabel('batch', fontsize=14)
    plt.ylabel('rewards', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 添加可选的平滑曲线（如果有足够的数据点）
    if len(eval_batches) > 5:
        # 使用简单的移动平均来平滑曲线
        window_size = min(10, len(eval_batches) // 2)
        if window_size > 1:
            # 对单智能体奖励进行平滑处理
            smooth_rewards = []
            for i in range(len(eval_rewards) - window_size + 1):
                smooth_rewards.append(np.mean(eval_rewards[i:i+window_size]))
            
            # 由于窗口平滑，x坐标需要调整
            smooth_x = eval_batches[:len(smooth_rewards)]
            plt.plot(smooth_x, smooth_rewards, 'b-', linewidth=2.5, 
                     label='smooth rewards')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f'./results/eval_performance_curve_{teamate_id}_{type}.png')
    print(f"评估性能曲线已保存至: ./results/eval_performance_curve_{teamate_id}_{type}.png")
    plt.close()

def save_history(history, nfsp_agents):
    """保存训练历史数据"""
    # 创建保存目录
    os.makedirs("./results", exist_ok=True)
    
    # 准备保存数据
    data_to_save = {
        'episode_rewards': history['episode_rewards'],
        'eval_rewards': history['eval_rewards'],
        'eval_batches': history['eval_batches'],
    }
    
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

