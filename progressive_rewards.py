#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import argparse
from tqdm import tqdm
import logging

import lbforaging
from lbforaging.foraging.environment_3d import ForagingEnv3D
from lbforaging.agents import NFSPAgent
from utils import calculate_state_size, evaluate, plot_training_curve, save_history

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressiveRewardWrapper:
    """
    递进式奖励包装器，根据训练进度增加奖励
    """
    def __init__(self, env, total_episodes=3000, reward_scale_max=3.0):
        self.env = env
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.base_reward_scale = env.food_reward_scale
        self.reward_scale_max = reward_scale_max
        
        # 保存原始环境的run方法
        self.original_run = env.run
        
        # 重载环境的run方法
        def wrapped_run(*args, **kwargs):
            # 更新当前回合数
            self.current_episode += 1
            
            # 计算动态奖励缩放系数（从1.0逐渐增加到reward_scale_max）
            progress = min(1.0, self.current_episode / (self.total_episodes * 0.8))
            current_scale = 1.0 + progress * (self.reward_scale_max - 1.0)
            
            # 临时修改环境的奖励比例
            original_scale = self.env.food_reward_scale
            self.env.food_reward_scale = self.base_reward_scale * current_scale
            
            # 调用原始的run方法
            trajectories, payoffs = self.original_run(*args, **kwargs)
            
            # 恢复环境的奖励比例
            self.env.food_reward_scale = original_scale
            
            # 打印进度信息（每100回合）
            if self.current_episode % 100 == 0:
                logger.info(f"回合 {self.current_episode}/{self.total_episodes}, 奖励比例: {current_scale:.2f}")
            
            return trajectories, payoffs
        
        # 替换环境的run方法
        self.env.run = wrapped_run
    
    def __getattr__(self, name):
        """转发所有其他属性访问到原始环境"""
        return getattr(self.env, name)

def train_with_progressive_rewards(args):
    """使用递进式奖励训练智能体"""
    # 创建3D环境
    env = ForagingEnv3D(
        n_rows=4,
        n_cols=4,
        n_depth=4,
        num_agents=2,
        num_food=3,
        max_player_level=3,
        min_player_level=2,
        max_food_level=1,
        min_food_level=1,
        sight=4,
        force_coop=True,  # 强制合作，增加挑战
        grid_observation=True,
        penalty=0.02,
        step_reward_factor=0.1,
        step_reward_threshold=0.01,
        food_reward_scale=10.0,  # 基础奖励
        proximity_factor=0.1,
        enable_proximity_reward=True,
        enable_step_reward=True
    )
    
    # 包装环境，实现递进式奖励
    env = ProgressiveRewardWrapper(env, total_episodes=args.episodes, reward_scale_max=args.reward_scale_max)
    
    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        env.seed(args.seed)
    
    # 计算状态空间大小
    state_size = calculate_state_size(env)
    logger.info(f"状态空间大小: {state_size}")
    action_size = 7  # 3D环境有7个动作
    
    # 创建保存目录
    save_dir = "./models_progressive"
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建智能体
    agents = []
    for i in range(env.num_agents):
        agent_dir = os.path.join(save_dir, f"agent{i}")
        os.makedirs(agent_dir, exist_ok=True)
        
        agent = NFSPAgent(
            player=env.players[i],
            state_size=state_size,
            action_size=action_size,
            epsilon_init=0.9,  # 开始时高探索
            epsilon_decay=args.episodes // 3,  # 更快的探索衰减
            epsilon_min=0.05,  # 降低最小探索率，以促进收敛
            update_freq=50,
            sl_lr=0.01,
            rl_lr=0.01,
            sl_buffer_size=10000,
            rl_buffer_size=20000,
            rl_start=100,
            sl_start=100,
            train_freq=1,
            gamma=0.99,
            eta=0.1,  # 降低平均策略的使用概率
            rl_batch_size=64,
            sl_batch_size=128,
            hidden_units=1024,
            layers=6,
            device=args.device
        )
        agents.append(agent)
    
    # 训练历史记录
    history = {
        'episode_rewards': [],
        'eval_rewards': [],
        'eval_episodes': [],
        'exploitability': [],
        'sl_losses': [],
        'rl_losses': [],
        'policy_accuracies': []
    }
    
    # 创建结果目录
    os.makedirs("./results_progressive", exist_ok=True)
    
    # 用于记录100回合的奖励
    recent_rewards = []
    
    # 设置初始批次
    total_batches = args.episodes // 100
    
    print(f"\n开始训练 - 总共 {args.episodes} 回合 ({total_batches} 批次)...\n")
    
    # 外层循环处理每个批次
    for batch in range(total_batches):
        # 为每个批次创建一个tqdm进度条
        batch_start_time = time.time()
        batch_size = min(100, args.episodes - batch * 100)  # 处理最后一个不完整批次
        
        # 创建当前批次的进度条
        with tqdm(total=batch_size, desc=f"批次 {batch+1}/{total_batches}",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            # 记录当前批次的奖励
            batch_rewards = []
            
            # 处理当前批次中的每个回合
            for i in range(batch_size):
                episode = batch * 100 + i
                
                # 在每个回合开始时选择策略模式
                for agent in agents:
                    agent.choose_policy_mode()
                
                # 判断是否需要在本回合渲染
                should_render = args.render and episode % args.render_interval == 0
                
                if should_render:
                    print(f"\n渲染回合 {episode}...")
                    
                    # 运行一个完整回合，并渲染
                    trajectories, payoffs = env.run(
                        agents,
                        is_training=True,
                        render=True,
                        sleep_time=0.5,
                        render_mode='3d'
                    )
                    # 暂停进度条更新
                    pbar.clear()
                
                else:
                    # 正常训练，不渲染
                    trajectories, payoffs = env.run(
                        agents,
                        is_training=True,
                        render=False
                    )
                
                # 记录每个回合的奖励
                history['episode_rewards'].append(payoffs)
                
                # 计算团队总奖励（智能体奖励之和）
                team_reward = sum(payoffs)
                recent_rewards.append(team_reward)
                batch_rewards.append(team_reward)
                
                # 更新进度条
                avg_batch_reward = sum(batch_rewards) / len(batch_rewards)
                elapsed_time = time.time() - batch_start_time
                pbar.set_postfix({
                    '奖励': f'{team_reward:.2f}',
                    '平均': f'{avg_batch_reward:.2f}',
                    '用时': f'{elapsed_time:.1f}s'
                })
                pbar.update(1)
                
                # 存储轨迹并训练
                for j in range(env.num_agents):
                    for ts in trajectories[j]:
                        if len(ts) > 0:
                            agents[j].add_traj(ts)
                    agents[j].train()
                
                # 每10个回合保存损失和准确率
                if episode % 10 == 0 and len(agents) > 0:
                    # 使用第一个智能体的损失数据
                    agent = agents[0]
                    # 保存SL损失
                    if hasattr(agent, 'losses') and len(agent.losses) > 0:
                        history['sl_losses'].append(agent.losses[-1])
                    # 保存RL损失
                    if hasattr(agent, 'RLlosses') and len(agent.RLlosses) > 0:
                        history['rl_losses'].append(agent.RLlosses[-1])
                    # 保存策略准确率
                    if hasattr(agent, 'policy_accuracies') and len(agent.policy_accuracies) > 0:
                        history['policy_accuracies'].append(agent.policy_accuracies[-1])
                
                # 定期评估
                if episode % args.eval_interval == 0 or episode == args.episodes - 1:
                    try:
                        pbar.clear()  # 暂时清除进度条显示
                        
                        # 计算普通奖励和可利用度
                        eval_rewards, exploitability = evaluate(
                            env,
                            agents,
                            num_episodes=10,
                            calculate_exploitability=True
                        )
                        
                        # 确保eval_rewards是一个单一值，并记录当前回合数
                        agent0_reward = eval_rewards[0]
                        history['eval_rewards'].append(agent0_reward)
                        history['eval_episodes'].append(episode)  # 记录评估回合
                        history['exploitability'].append(exploitability)  # 记录可利用度
                        
                        # 恢复进度条显示
                        pbar.display()
                    except Exception as e:
                        print(f"\n评估过程中发生错误: {e}")
                        import traceback
                        traceback.print_exc()
            
            # 批次完成后显示平均奖励
            avg_reward = np.mean(recent_rewards)
            elapsed_time = time.time() - batch_start_time
            
            # 关闭当前进度条
            pbar.close()
            
            # 打印批次完成信息
            batch_summary = f"✅ 批次 {batch+1}/{total_batches} 完成 | 平均奖励: {avg_reward:.4f} | 用时: {elapsed_time:.1f}秒 | 总进度: {(batch+1)/total_batches*100:.1f}%"
            print(f"\033[92m{batch_summary}\033[0m\n")
            
            # 重置recent_rewards列表
            recent_rewards = []
    
    print("\n训练完成！\n")
    
    # 保存模型
    for i, agent in enumerate(agents):
        agent_dir = os.path.join(save_dir, f"agent{i}")
        agent.save_models(path=agent_dir)
        print(f"已保存智能体{i}的模型到 {agent_dir}")
    
    # 绘制训练曲线
    plot_training_curve(history, args.episodes, args.eval_interval, agents)
    
    # 保存训练历史记录
    np.savez('./results_progressive/training_history.npz', **history)
    print("训练历史已保存到 ./results_progressive/training_history.npz")
    
    return agents, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用递进式奖励训练NFSP智能体")
    parser.add_argument("--episodes", type=int, default=3000, help="训练回合数")
    parser.add_argument("--render", action="store_true", help="启用环境渲染")
    parser.add_argument("--render_interval", type=int, default=500, help="渲染间隔")
    parser.add_argument("--eval_interval", type=int, default=200, help="评估间隔")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cpu:0", help="训练设备")
    parser.add_argument("--reward_scale_max", type=float, default=3.0, help="最终奖励缩放因子")
    
    args = parser.parse_args()
    train_with_progressive_rewards(args) 