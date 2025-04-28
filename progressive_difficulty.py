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

class ProgressiveDifficultyEnv:
    """
    渐进式难度环境包装器，随着训练进度增加游戏难度
    """
    def __init__(self, 
                 total_episodes=3000, 
                 initial_difficulty={
                     'food_level': 1,
                     'num_food': 2,
                     'sight': 5,
                     'penalty': 0.01
                 }, 
                 final_difficulty={
                     'food_level': 3,
                     'num_food': 5,
                     'sight': 3,
                     'penalty': 0.1
                 }):
        # 基本参数
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.env = None
        
        # 记录当前的难度参数
        self.current_difficulty = initial_difficulty.copy()
        
        # 记录难度历史
        self.difficulty_history = {k: [] for k in initial_difficulty.keys()}
        
    def create_env(self):
        """创建带有当前难度参数的环境"""
        # 如果已有环境，先关闭
        if self.env is not None:
            self.env.close()
        
        # 创建3D环境，使用当前难度参数
        self.env = ForagingEnv3D(
            n_rows=4,
            n_cols=4,
            n_depth=4,
            num_agents=2,
            num_food=self.current_difficulty['num_food'],
            max_player_level=3,
            min_player_level=2,
            max_food_level=self.current_difficulty['food_level'],
            min_food_level=self.current_difficulty['food_level'],
            sight=self.current_difficulty['sight'],
            force_coop=True,  # 强制合作
            grid_observation=True,
            penalty=self.current_difficulty['penalty'],
            step_reward_factor=0.1,
            step_reward_threshold=0.01,
            food_reward_scale=15.0,
            proximity_factor=0.1,
            enable_proximity_reward=True,
            enable_step_reward=True
        )
        return self.env
    
    def update_difficulty(self):
        """更新难度参数，根据当前训练进度"""
        self.current_episode += 1
        
        # 计算当前完成的训练比例（最多80%，以保持一段时间的最终难度）
        progress = min(0.8, self.current_episode / self.total_episodes)
        
        # 更新每个难度参数
        for param in self.initial_difficulty:
            # 线性插值计算当前难度
            initial_value = self.initial_difficulty[param]
            final_value = self.final_difficulty[param]
            
            # 更新当前参数值
            self.current_difficulty[param] = initial_value + progress * (final_value - initial_value)
            
            # 对整数参数进行取整
            if param in ['num_food', 'food_level', 'sight']:
                self.current_difficulty[param] = int(round(self.current_difficulty[param]))
            
            # 记录难度历史
            if self.current_episode % 10 == 0:
                self.difficulty_history[param].append(self.current_difficulty[param])
        
        # 每100回合打印一次当前难度
        if self.current_episode % 100 == 0:
            logger.info(f"回合 {self.current_episode}/{self.total_episodes}, 当前难度: {self.current_difficulty}")
        
        # 更新环境参数
        if self.env is not None:
            # 更新食物等级
            self.env.max_food_level = np.array([self.current_difficulty['food_level']] * self.env.num_food)
            self.env.min_food_level = np.array([self.current_difficulty['food_level']] * self.env.num_food)
            
            # 更新视野
            self.env.sight = self.current_difficulty['sight']
            
            # 更新惩罚
            self.env.penalty = self.current_difficulty['penalty']
            
            # 每500回合或当食物数量变化时重新创建环境
            if self.current_episode % 500 == 0 or self.env.num_food != self.current_difficulty['num_food']:
                self.create_env()

def train_with_progressive_difficulty(args):
    """使用渐进式难度训练智能体"""
    # 创建渐进式难度环境
    prog_env = ProgressiveDifficultyEnv(
        total_episodes=args.episodes,
        initial_difficulty={
            'food_level': 1,
            'num_food': 2,
            'sight': 5,
            'penalty': 0.01
        },
        final_difficulty={
            'food_level': args.max_food_level,
            'num_food': args.max_food_num,
            'sight': args.min_sight,
            'penalty': args.max_penalty
        }
    )
    
    # 初始化环境
    env = prog_env.create_env()
    
    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        env.seed(args.seed)
    
    # 计算状态空间大小
    state_size = calculate_state_size(env)
    logger.info(f"状态空间大小: {state_size}")
    action_size = 7  # 3D环境有7个动作
    
    # 创建保存目录
    save_dir = "./models_progressive_difficulty"
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
            epsilon_init=0.9,  # 高初始探索率
            epsilon_decay=args.episodes // 4,  # 较快的探索衰减
            epsilon_min=0.1,  # 较低的最小探索率
            update_freq=50,
            sl_lr=0.01,
            rl_lr=0.01,
            sl_buffer_size=12000,
            rl_buffer_size=20000,
            rl_start=100,
            sl_start=100,
            train_freq=1,
            gamma=0.99,
            eta=0.2,  # 平均策略的使用概率
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
        'policy_accuracies': [],
        'difficulty_history': prog_env.difficulty_history
    }
    
    # 创建结果目录
    os.makedirs("./results_progressive_difficulty", exist_ok=True)
    
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
                
                # 更新环境难度
                prog_env.update_difficulty()
                
                # 在每个回合开始时选择策略模式
                for agent in agents:
                    agent.choose_policy_mode()
                
                # 判断是否需要在本回合渲染
                should_render = args.render and episode % args.render_interval == 0
                
                if should_render:
                    print(f"\n渲染回合 {episode}...")
                    print(f"当前难度: 食物等级={prog_env.current_difficulty['food_level']}, "
                          f"食物数量={prog_env.current_difficulty['num_food']}, "
                          f"视野={prog_env.current_difficulty['sight']}, "
                          f"惩罚={prog_env.current_difficulty['penalty']:.3f}")
                    
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
                
                # 显示难度信息
                difficulty_info = f"F{prog_env.current_difficulty['food_level']}"
                difficulty_info += f"|N{prog_env.current_difficulty['num_food']}"
                difficulty_info += f"|S{prog_env.current_difficulty['sight']}"
                
                pbar.set_postfix({
                    '奖励': f'{team_reward:.2f}',
                    '平均': f'{avg_batch_reward:.2f}',
                    '难度': difficulty_info,
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
                        
                        # 创建临时评估环境（使用当前参数）
                        eval_env = ForagingEnv3D(
                            n_rows=4,
                            n_cols=4,
                            n_depth=4,
                            num_agents=2,
                            num_food=prog_env.current_difficulty['num_food'],
                            max_player_level=3,
                            min_player_level=2,
                            max_food_level=prog_env.current_difficulty['food_level'],
                            min_food_level=prog_env.current_difficulty['food_level'],
                            sight=prog_env.current_difficulty['sight'],
                            force_coop=True,
                            grid_observation=True,
                            penalty=prog_env.current_difficulty['penalty'],
                            step_reward_factor=0.1,
                            step_reward_threshold=0.01,
                            food_reward_scale=15.0,
                            proximity_factor=0.1,
                            enable_proximity_reward=True,
                            enable_step_reward=True
                        )
                        
                        # 计算普通奖励和可利用度
                        eval_rewards, exploitability = evaluate(
                            env,
                            agents,
                            num_episodes=10,
                            calculate_exploitability=True,
                            eval_env=eval_env
                        )
                        
                        # 确保eval_rewards是一个单一值，并记录当前回合数
                        agent0_reward = eval_rewards[0]
                        history['eval_rewards'].append(agent0_reward)
                        history['eval_episodes'].append(episode)  # 记录评估回合
                        history['exploitability'].append(exploitability)  # 记录可利用度
                        
                        # 关闭临时评估环境
                        eval_env.close()
                        
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
            difficulty_str = f"食物等级={prog_env.current_difficulty['food_level']}, "
            difficulty_str += f"食物数量={prog_env.current_difficulty['num_food']}, "
            difficulty_str += f"视野={prog_env.current_difficulty['sight']}, "
            difficulty_str += f"惩罚={prog_env.current_difficulty['penalty']:.3f}"
            
            batch_summary = f"✅ 批次 {batch+1}/{total_batches} 完成 | 平均奖励: {avg_reward:.4f} | 当前难度: {difficulty_str} | 用时: {elapsed_time:.1f}秒"
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
    np.savez('./results_progressive_difficulty/training_history.npz', **history)
    print("训练历史已保存到 ./results_progressive_difficulty/training_history.npz")
    
    # 绘制难度参数变化曲线
    plot_difficulty_curves(prog_env.difficulty_history, args.episodes)
    
    return agents, history

def plot_difficulty_curves(difficulty_history, total_episodes):
    """绘制难度参数变化曲线"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    
    # 遍历每个难度参数
    for i, (param_name, param_values) in enumerate(difficulty_history.items(), 1):
        plt.subplot(2, 2, i)
        episodes = np.arange(len(param_values)) * 10  # 记录频率为每10回合
        plt.plot(episodes, param_values)
        plt.xlabel('Episodes')
        plt.ylabel(param_name.capitalize())
        plt.title(f'{param_name.capitalize()} Progression')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./results_progressive_difficulty/difficulty_curves.png')
    print("难度变化曲线已保存到 ./results_progressive_difficulty/difficulty_curves.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用渐进式难度训练NFSP智能体")
    parser.add_argument("--episodes", type=int, default=3000, help="训练回合数")
    parser.add_argument("--render", action="store_true", help="启用环境渲染")
    parser.add_argument("--render_interval", type=int, default=500, help="渲染间隔")
    parser.add_argument("--eval_interval", type=int, default=200, help="评估间隔")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cpu:0", help="训练设备")
    parser.add_argument("--max_food_level", type=int, default=3, help="最终食物等级")
    parser.add_argument("--max_food_num", type=int, default=5, help="最终食物数量")
    parser.add_argument("--min_sight", type=int, default=3, help="最低视野范围")
    parser.add_argument("--max_penalty", type=float, default=0.1, help="最大动作惩罚")
    
    args = parser.parse_args()
    
    # 导入matplotlib以绘制难度曲线
    import matplotlib.pyplot as plt
    
    train_with_progressive_difficulty(args) 