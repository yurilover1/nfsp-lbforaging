import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime


def action_mask(probs, legal_actions):
    """
    过滤不合法的动作，只保留合法动作的概率
    
    参数:
        probs: 动作概率分布
        legal_actions: 合法动作列表
    
    返回:
        过滤后的概率分布
    """
    # 确保legal_actions非空
    if not legal_actions:
        print("警告: legal_actions为空，仅使用NONE动作(索引0)")
        legal_actions = [0]  # 只使用NONE动作(索引0)，而不是所有动作
    
    # 处理probs中可能存在的NaN或inf值
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 创建掩码数组，初始化为全0
    masked_probs = np.zeros(len(probs))
    
    # 只设置合法动作的概率
    for action in legal_actions:
        if 0 <= action < len(probs):  # 确保动作索引有效
            masked_probs[action] = probs[action]
    
    # 如果所有概率和为0，则采用均匀分布
    prob_sum = np.sum(masked_probs)
    if prob_sum <= 1e-10:  # 使用一个很小的阈值而不是精确的0
        for action in legal_actions:
            if 0 <= action < len(masked_probs):  # 确保动作索引有效
                masked_probs[action] = 1.0 / len(legal_actions)
    else:
        # 重新归一化概率分布
        masked_probs = masked_probs / prob_sum
    
    # 最后再次检查是否包含NaN或inf
    masked_probs = np.nan_to_num(masked_probs, nan=1.0/len(legal_actions), posinf=1.0, neginf=0.0)
    
    # 确保概率和为1
    if np.sum(masked_probs) <= 1e-10:  # 极端情况下仍然可能概率和为0
        if legal_actions:
            # 使用均匀分布
            for action in legal_actions:
                if 0 <= action < len(masked_probs):
                    masked_probs[action] = 1.0 / len(legal_actions)
        else:
            # 如果一切都失败了，使用NONE动作（索引0）
            masked_probs[0] = 1.0
    else:
        # 再次归一化，确保和为1
        masked_probs = masked_probs / np.sum(masked_probs)
    
    return masked_probs

def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """计算广义优势估计（GAE）"""
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lambda_ * (1 - dones[t]) * last_gae
    
    returns = advantages + values
    return advantages, returns

def create_log_dir(base_dir="logs"):
    """创建日志目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, f"ppo_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

class Logger:
    """训练日志记录器"""
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "value_losses": [],
            "policy_losses": [],
            "entropy_losses": []
        }
    
    def log_metrics(self, metrics_dict, step):
        """记录指标"""
        for key, value in metrics_dict.items():
            self.writer.add_scalar(key, value, step)
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def save_metrics(self):
        """保存指标到文件"""
        metrics_file = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f)
    
    def close(self):
        """关闭日志记录器"""
        self.writer.close()
        self.save_metrics()

def create_checkpoint_dir(base_dir="checkpoints"):
    """创建检查点目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(base_dir, f"ppo_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 为每个智能体创建子目录
    for i in range(2):
        agent_dir = os.path.join(checkpoint_dir, f"agent_{i}")
        os.makedirs(agent_dir, exist_ok=True)
    
    return checkpoint_dir

def save_checkpoint(agent, optimizer, episode, checkpoint_dir):
    """保存检查点"""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pt")
    torch.save({
        'episode': episode,
        'model_state_dict': agent.actor_critic.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    return checkpoint_path

def load_checkpoint(agent, optimizer, checkpoint_path):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path)
    agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode']