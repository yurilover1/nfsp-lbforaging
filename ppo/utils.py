import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import json
from datetime import datetime

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