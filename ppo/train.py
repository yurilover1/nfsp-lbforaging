import os
import gymnasium as gym
import numpy as np
import torch
import lbforaging
from ppo.agent import PPOAgent
from ppo.utils import (
    compute_gae, create_log_dir, Logger,
    create_checkpoint_dir, save_checkpoint
)
from partners.agent import SimpleAgent2
import argparse
from tqdm import tqdm
import time

# 训练参数
NUM_EPISODES = 10000
MAX_STEPS = 100
SAVE_INTERVAL = 100
EVAL_INTERVAL = 50
NUM_EVAL_EPISODES = 10

def train_ppo_agents(env, agents, num_episodes=10000, eval_interval=100, render=False, render_interval=100, checkpoint_dir=None, logger=None):
    """PPO训练函数，提供与main.py兼容的接口
    
    参数:
        env: 游戏环境
        agents: 智能体列表（可以是单个PPO或多个PPO）
        num_episodes: 训练回合数
        eval_interval: 评估间隔
        render: 是否渲染
        render_interval: 渲染间隔
        checkpoint_dir: 检查点保存目录，如果为None则创建新目录
        logger: 日志记录器，如果为None则创建新记录器
    
    返回:
        训练后的智能体列表
    """
    # 过滤出PPO智能体
    ppo_agents = [agent for agent in agents if hasattr(agent, 'update')]
    if not ppo_agents:
        print("未发现PPO智能体，无法进行训练")
        return agents
    
    # 创建日志记录器
    if logger is None:
        log_dir = create_log_dir()
        logger = Logger(log_dir)
    
    # 创建检查点目录
    if checkpoint_dir is None:
        checkpoint_dir = create_checkpoint_dir()
        # 为每个智能体创建子目录
        for i in range(len(ppo_agents)):
            agent_dir = os.path.join(checkpoint_dir, f"agent_{i}")
            os.makedirs(agent_dir, exist_ok=True)

    # 训练循环
    for episode in tqdm(range(num_episodes)):
        # 判断是否渲染当前回合
        should_render = render and episode % render_interval == 0
        
        # 使用env.run执行回合，与NFSP保持一致的接口
        if should_render:
            print(f"\n渲染回合 {episode}...")
            trajectories, payoffs, steps = env.run(
                agents, 
                is_training=True, 
                render=True, 
                sleep_time=0.5
            )
        else:
            trajectories, payoffs, steps = env.run(
                agents, 
                is_training=True, 
                render=False
            )
        
        # 记录轨迹 (agents中自动存储了轨迹)
        # 调用每个PPO智能体的train方法更新策略
        for agent in ppo_agents:
            agent.train()
        
        # 记录指标
        metrics = {
            "episode_length": steps,
            "total_reward": sum(payoffs)
        }
        
        # 添加每个智能体的奖励
        for i, reward in enumerate(payoffs):
            metrics[f"episode_rewards_{i}"] = reward
        
        logger.log_metrics(metrics, episode)
        
        # 保存检查点
        if (episode + 1) % SAVE_INTERVAL == 0:
            for i, agent in enumerate(ppo_agents):
                save_checkpoint(
                    agent,
                    agent.optimizer,
                    episode,
                    os.path.join(checkpoint_dir, f"agent_{i}")
                )
        
        # 评估
        if (episode + 1) % eval_interval == 0:
            eval_rewards = evaluate(env, agents, NUM_EVAL_EPISODES)
            
            eval_metrics = {
                "eval_total_reward": sum(eval_rewards)
            }
            
            # 添加每个智能体的评估奖励
            for i, reward in enumerate(eval_rewards):
                eval_metrics[f"eval_rewards_{i}"] = reward
                
            logger.log_metrics(eval_metrics, episode)
    
    # 训练结束，保存最终模型
    print("PPO训练完成，保存最终模型...")
    for i, agent in enumerate(ppo_agents):
        agent.save_model(os.path.join(checkpoint_dir, f"agent_{i}", "final_model.pt"))
    
    # 关闭日志记录器
    logger.close()
    
    return agents

def evaluate(env, agents, num_episodes):
    """评估函数，使用env.run方法"""
    total_rewards = np.zeros(len(agents))
    
    for _ in range(num_episodes):
        _, payoffs, _ = env.run(agents, is_training=False)
        total_rewards += np.array(payoffs)
    
    # 计算平均奖励
    eval_rewards = total_rewards / num_episodes
    return eval_rewards

def train(args):
    """命令行训练入口函数"""
    # 创建环境
    env = gym.make(
        "Foraging-5x5-2p-2f-v3",
        render_mode="human" if args.render else None,
        # force_coop=True,
        max_episode_steps=MAX_STEPS
    )
    
    # 设置Player信息
    class PlayerInfo:
        def __init__(self, level):
            self.level = level  # 智能体等级
    
    # 创建两个PPO智能体进行训练
    agents = [
        PPOAgent(
            input_dim=12,  # 输入维度为12
            hidden_dims=[128, 128],
            output_dim=6,  # 动作空间为6
            device=args.device,
            player=PlayerInfo(i)
        ) for i in range(2)
    ]
    
    # 调用训练函数
    train_ppo_agents(
        env, 
        agents, 
        num_episodes=NUM_EPISODES, 
        eval_interval=EVAL_INTERVAL,
        render=args.render,
        render_interval=100
    )
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    
    train(args) 