import numpy as np
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)

def evaluate(env, agents, eval_episodes=100, eval_env=None):
    """
    评估智能体性能
    
    参数:
        env: 游戏环境
        agents: 要评估的智能体列表
        eval_episodes: 评估回合数
        eval_env: 用于评估的环境，如果为None，则使用原环境的副本
    
    返回:
        返回每个智能体的平均奖励
    """
    if eval_env is None:
        # 创建一个新的环境用于评估
        eval_env = gym.make(env.unwrapped.spec.id, render_mode=None)

    # 使用环境的run方法执行评估
    agents[0].policy_mode='best'
    total_rewards = 0
    for _ in range(eval_episodes):
        _, payoffs, steps = eval_env.run(agents, is_training=False)
        total_rewards += payoffs.sum()
    # 计算平均奖励
    avg_rewards = total_rewards/ eval_episodes

    return avg_rewards.sum()