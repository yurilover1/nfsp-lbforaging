#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import argparse
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from datetime import datetime

import lbforaging
from lbforaging.foraging.environment_3d import ForagingEnv3D
from lbforaging.agents import NFSPAgent
from utils import calculate_state_size, evaluate, plot_training_curve, save_history, test_agents

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingCheckpoint:
    """
    训练检查点管理器，用于保存和加载训练状态
    """
    def __init__(self, checkpoint_dir="./checkpoints", save_interval=1000, max_checkpoints=5):
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.checkpoint_history = []
        
        # 确保检查点目录存在
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def should_save(self, episode):
        """判断是否应该保存检查点"""
        return episode % self.save_interval == 0
    
    def save(self, agents, history, episode):
        """保存当前训练状态"""
        # 创建检查点目录
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{episode}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 保存每个智能体的模型
        for i, agent in enumerate(agents):
            agent_dir = os.path.join(checkpoint_path, f"agent{i}")
            os.makedirs(agent_dir, exist_ok=True)
            agent.save_models(path=agent_dir)
        
        # 保存训练历史
        np.savez(os.path.join(checkpoint_path, "history.npz"), **history)
        
        # 记录检查点
        self.checkpoint_history.append({"episode": episode, "path": checkpoint_path})
        
        # 如果检查点数量超过最大值，删除最早的检查点
        if len(self.checkpoint_history) > self.max_checkpoints:
            oldest = self.checkpoint_history.pop(0)
            # 这里可以添加删除最旧检查点的代码，但为了安全起见，我们暂时不自动删除
        
        logger.info(f"已保存检查点: {checkpoint_path}")
        return checkpoint_path
    
    def load_latest(self):
        """加载最新的检查点"""
        # 获取检查点目录中的所有检查点
        checkpoints = []
        if os.path.exists(self.checkpoint_dir):
            for d in os.listdir(self.checkpoint_dir):
                if d.startswith("checkpoint_"):
                    try:
                        episode = int(d.split("_")[1])
                        checkpoints.append((episode, os.path.join(self.checkpoint_dir, d)))
                    except:
                        pass
        
        if not checkpoints:
            return None, 0
        
        # 按回合数排序，获取最新的检查点
        checkpoints.sort(key=lambda x: x[0])
        latest_episode, latest_path = checkpoints[-1]
        
        return latest_path, latest_episode

def train_with_long_horizon(args):
    """使用更长的训练周期训练NFSP智能体"""
    # 创建检查点管理器
    checkpoint_manager = TrainingCheckpoint(
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.checkpoint_interval,
        max_checkpoints=args.max_checkpoints
    )
    
    # 检查是否有检查点需要加载
    latest_checkpoint, start_episode = checkpoint_manager.load_latest()
    if latest_checkpoint and args.resume:
        logger.info(f"找到检查点，从回合 {start_episode} 继续训练")
        resume_training = True
    else:
        logger.info("从头开始训练")
        start_episode = 0
        resume_training = False
    
    # 创建环境
    env = ForagingEnv3D(
        n_rows=4,
        n_cols=4,
        n_depth=4,
        num_agents=2,
        num_food=4,
        max_player_level=3,
        min_player_level=2,
        max_food_level=2,
        min_food_level=1,
        sight=4,
        force_coop=True,  # 强制合作
        grid_observation=True,
        penalty=0.02,
        step_reward_factor=0.05,
        step_reward_threshold=0.01,
        food_reward_scale=10.0,
        proximity_factor=0.05,
        enable_proximity_reward=True,
        enable_step_reward=True
    )
    
    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        env.seed(args.seed)
    
    # 计算状态空间大小
    state_size = calculate_state_size(env)
    logger.info(f"状态空间大小: {state_size}")
    action_size = 7  # 3D环境有7个动作
    
    # 创建保存目录
    save_dir = "./models_long_training"
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建评估记录目录
    eval_dir = "./eval_results"
    os.makedirs(eval_dir, exist_ok=True)
    
    # 创建智能体
    agents = []
    
    # 设置不同的衰减率，以观察不同衰减策略的效果
    epsilon_decay_values = [args.episodes // 3, args.episodes // 4]  # 为两个智能体设置不同的衰减率
    
    for i in range(env.num_agents):
        agent_dir = os.path.join(save_dir, f"agent{i}")
        os.makedirs(agent_dir, exist_ok=True)
        
        # 选择当前智能体的衰减率
        epsilon_decay = epsilon_decay_values[i % len(epsilon_decay_values)]
        
        agent = NFSPAgent(
            player=env.players[i],
            state_size=state_size,
            action_size=action_size,
            epsilon_init=0.9,  # 高初始探索率
            epsilon_decay=epsilon_decay,
            epsilon_min=0.05,  # 较低的最小探索率
            update_freq=40,
            sl_lr=0.005,  # 降低学习率以提高稳定性
            rl_lr=0.005,
            sl_buffer_size=20000,  # 增大缓冲区大小
            rl_buffer_size=30000,
            rl_start=500,  # 延迟开始学习，收集更多经验
            sl_start=500,
            train_freq=2,  # 减少训练频率，增强数据多样性
            gamma=0.99,
            eta=0.1,  # 降低平均策略的使用概率
            rl_batch_size=128,  # 增大批次大小
            sl_batch_size=256,
            hidden_units=1024,
            layers=6,
            device=args.device
        )
        agents.append(agent)
    
    # 如果有检查点，从检查点加载模型和历史
    if resume_training:
        history = {}
        # 加载每个智能体的模型
        for i, agent in enumerate(agents):
            agent_checkpoint_dir = os.path.join(latest_checkpoint, f"agent{i}")
            if os.path.exists(agent_checkpoint_dir):
                try:
                    agent.load_models(path=agent_checkpoint_dir)
                    logger.info(f"已加载智能体{i}的模型从 {agent_checkpoint_dir}")
                except Exception as e:
                    logger.error(f"加载智能体{i}的模型时出错: {e}")
        
        # 加载训练历史
        history_path = os.path.join(latest_checkpoint, "history.npz")
        if os.path.exists(history_path):
            try:
                checkpoint_history = dict(np.load(history_path))
                # 转换numpy数组为列表
                for k, v in checkpoint_history.items():
                    if isinstance(v, np.ndarray):
                        history[k] = v.tolist() if v.dtype == np.object else v
                    else:
                        history[k] = v
                logger.info(f"已加载训练历史从 {history_path}")
            except Exception as e:
                logger.error(f"加载训练历史时出错: {e}")
                history = {
                    'episode_rewards': [],
                    'eval_rewards': [],
                    'eval_episodes': [],
                    'exploitability': [],
                    'sl_losses': [],
                    'rl_losses': [],
                    'policy_accuracies': []
                }
    else:
        # 初始化空的训练历史
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
    results_dir = "./results_long_training"
    os.makedirs(results_dir, exist_ok=True)
    
    # 用于记录100回合的奖励
    recent_rewards = []
    
    # 计算剩余回合数和批次
    remaining_episodes = args.episodes - start_episode
    total_batches = (remaining_episodes + 99) // 100  # 向上取整
    
    print(f"\n开始训练 - 总共 {args.episodes} 回合，从回合 {start_episode} 继续，剩余 {remaining_episodes} 回合 ({total_batches} 批次)...\n")
    
    # 创建保存周期性评估结果的列表
    periodic_eval_results = []
    
    # 记录训练开始时间
    training_start_time = time.time()
    last_checkpoint_time = training_start_time
    
    # 外层循环处理每个批次
    for batch in range(total_batches):
        # 计算当前批次的开始回合
        batch_start_episode = start_episode + batch * 100
        
        # 为每个批次创建一个tqdm进度条
        batch_start_time = time.time()
        batch_size = min(100, args.episodes - batch_start_episode)  # 处理最后一个不完整批次
        
        # 创建当前批次的进度条
        with tqdm(total=batch_size, desc=f"批次 {batch+1}/{total_batches}",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            # 记录当前批次的奖励
            batch_rewards = []
            
            # 处理当前批次中的每个回合
            for i in range(batch_size):
                episode = batch_start_episode + i
                
                # 计算当前探索率
                current_epsilon = agents[0].epsilon(agents[0].count)
                
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
                    'ε': f'{current_epsilon:.2f}',
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
                            num_episodes=20,  # 增加评估回合以提高准确性
                            calculate_exploitability=True
                        )
                        
                        # 确保eval_rewards是一个单一值，并记录当前回合数
                        agent0_reward = eval_rewards[0]
                        history['eval_rewards'].append(agent0_reward)
                        history['eval_episodes'].append(episode)  # 记录评估回合
                        history['exploitability'].append(exploitability)  # 记录可利用度
                        
                        # 打印评估信息
                        print(f"\n评估 - 回合 {episode}/{args.episodes}: 奖励={eval_rewards}, 可利用度={exploitability:.4f}")
                        
                        # 恢复进度条显示
                        pbar.display()
                    except Exception as e:
                        print(f"\n评估过程中发生错误: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 周期性进行更全面的评估
                if episode % args.thorough_eval_interval == 0 and episode > 0:
                    try:
                        pbar.clear()  # 暂时清除进度条显示
                        
                        print(f"\n进行周期性评估 - 回合 {episode}...")
                        
                        # 运行更多回合的评估
                        eval_results = test_agents(
                            env,
                            agents,
                            eval_episodes=50,  # 更多回合来确保评估稳定性
                            eval_explo=True,
                            render=False,
                            num_demo_episodes=0  # 不渲染
                        )
                        
                        # 记录评估结果
                        eval_results['episode'] = episode
                        eval_results['time'] = time.time() - training_start_time
                        periodic_eval_results.append(eval_results)
                        
                        # 保存周期性评估结果
                        np.savez(
                            os.path.join(eval_dir, f"eval_episode_{episode}.npz"),
                            episode=episode,
                            mean_reward=eval_results['mean_reward'],
                            agent_rewards=eval_results['agent_rewards'],
                            exploitability=eval_results['exploitability'],
                            time=eval_results['time']
                        )
                        
                        # 绘制周期性评估趋势图
                        if len(periodic_eval_results) > 1:
                            plot_periodic_eval_trends(periodic_eval_results, eval_dir)
                        
                        # 恢复进度条显示
                        pbar.display()
                    except Exception as e:
                        print(f"\n周期性评估时出错: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 检查是否需要保存检查点
                if checkpoint_manager.should_save(episode):
                    checkpoint_path = checkpoint_manager.save(agents, history, episode)
                    # 记录检查点保存时间
                    checkpoint_time = time.time()
                    time_since_last_checkpoint = checkpoint_time - last_checkpoint_time
                    last_checkpoint_time = checkpoint_time
                    
                    # 打印检查点信息
                    print(f"\n已保存检查点: {checkpoint_path}")
                    print(f"自上次检查点以来的训练时间: {time_since_last_checkpoint:.1f}秒")
                    
                    # 计算预估完成时间
                    elapsed_time = time.time() - training_start_time
                    progress = episode / args.episodes
                    if progress > 0:
                        estimated_total_time = elapsed_time / progress
                        remaining_time = estimated_total_time - elapsed_time
                        estimated_finish_time = time.time() + remaining_time
                        finish_time_str = datetime.fromtimestamp(estimated_finish_time).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"预估完成时间: {finish_time_str} (还需 {remaining_time / 3600:.1f} 小时)")
            
            # 批次完成后显示平均奖励
            avg_reward = np.mean(recent_rewards)
            elapsed_time = time.time() - batch_start_time
            
            # 关闭当前进度条
            pbar.close()
            
            # 计算总训练时间
            total_elapsed_time = time.time() - training_start_time
            hours, remainder = divmod(total_elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # 打印批次完成信息
            batch_summary = f"✅ 批次 {batch+1}/{total_batches} 完成 | 平均奖励: {avg_reward:.4f} | 批次用时: {elapsed_time:.1f}秒 | 总训练时间: {int(hours)}h {int(minutes)}m {int(seconds)}s"
            print(f"\033[92m{batch_summary}\033[0m\n")
            
            # 重置recent_rewards列表
            recent_rewards = []
            
            # 每批次结束后，保存当前的训练历史图表
            if batch % 5 == 0 or batch == total_batches - 1:
                try:
                    # 绘制训练曲线
                    plot_training_curve(history, batch_start_episode + batch_size, args.eval_interval, agents)
                    # 保存训练历史
                    np.savez(os.path.join(results_dir, "training_history.npz"), **history)
                    print(f"已保存训练曲线和历史数据 (批次 {batch+1}/{total_batches})")
                except Exception as e:
                    print(f"保存训练曲线时出错: {e}")
    
    # 训练完成，计算总时间
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n训练完成！")
    print(f"总训练时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    
    # 保存最终模型
    for i, agent in enumerate(agents):
        agent_dir = os.path.join(save_dir, f"agent{i}")
        agent.save_models(path=agent_dir)
        print(f"已保存智能体{i}的最终模型到 {agent_dir}")
    
    # 绘制最终训练曲线
    plot_training_curve(history, args.episodes, args.eval_interval, agents)
    
    # 保存最终训练历史记录
    np.savez(os.path.join(results_dir, "training_history.npz"), **history)
    print(f"已保存最终训练历史到 {os.path.join(results_dir, 'training_history.npz')}")
    
    # 绘制周期性评估趋势图
    if len(periodic_eval_results) > 1:
        plot_periodic_eval_trends(periodic_eval_results, eval_dir)
    
    # 运行最终评估
    print("\n运行最终评估...")
    final_eval_results = test_agents(
        env,
        agents,
        eval_episodes=100,  # 大量回合以确保评估准确性
        eval_explo=True,
        render=args.render,
        num_demo_episodes=5 if args.render else 0
    )
    
    # 保存最终评估结果
    np.savez(
        os.path.join(eval_dir, "final_evaluation.npz"),
        mean_reward=final_eval_results['mean_reward'],
        agent_rewards=final_eval_results['agent_rewards'],
        exploitability=final_eval_results['exploitability'],
        training_time=total_training_time
    )
    
    print(f"\n最终评估结果:")
    print(f"平均团队奖励: {final_eval_results['mean_reward']:.4f}")
    print(f"智能体奖励: {final_eval_results['agent_rewards']}")
    print(f"可利用度: {final_eval_results['exploitability']:.4f}")
    
    return agents, history, final_eval_results

def plot_periodic_eval_trends(eval_results, eval_dir):
    """绘制周期性评估趋势图"""
    # 提取数据
    episodes = [res['episode'] for res in eval_results]
    rewards = [res['mean_reward'] for res in eval_results]
    exploitabilities = [res.get('exploitability', 0) for res in eval_results]
    times = [res.get('time', 0) / 3600 for res in eval_results]  # 转换为小时
    
    plt.figure(figsize=(15, 10))
    
    # 绘制奖励趋势
    plt.subplot(2, 2, 1)
    plt.plot(episodes, rewards, 'bo-')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Team Reward')
    plt.title('Reward Trend Over Long Training')
    plt.grid(True)
    
    # 绘制可利用度趋势
    plt.subplot(2, 2, 2)
    plt.plot(episodes, exploitabilities, 'ro-')
    plt.xlabel('Episodes')
    plt.ylabel('Exploitability')
    plt.title('Exploitability Trend Over Long Training')
    plt.grid(True)
    
    # 绘制奖励与训练时间关系
    plt.subplot(2, 2, 3)
    plt.plot(times, rewards, 'go-')
    plt.xlabel('Training Time (hours)')
    plt.ylabel('Mean Team Reward')
    plt.title('Reward vs Training Time')
    plt.grid(True)
    
    # 绘制可利用度与训练时间关系
    plt.subplot(2, 2, 4)
    plt.plot(times, exploitabilities, 'mo-')
    plt.xlabel('Training Time (hours)')
    plt.ylabel('Exploitability')
    plt.title('Exploitability vs Training Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'eval_trends.png'))
    print(f"已保存评估趋势图到 {os.path.join(eval_dir, 'eval_trends.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用更长训练周期训练NFSP智能体")
    parser.add_argument("--episodes", type=int, default=10000, help="训练回合数")
    parser.add_argument("--render", action="store_true", help="启用环境渲染")
    parser.add_argument("--render_interval", type=int, default=1000, help="渲染间隔")
    parser.add_argument("--eval_interval", type=int, default=200, help="基本评估间隔")
    parser.add_argument("--thorough_eval_interval", type=int, default=1000, help="详细评估间隔")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cpu:0", help="训练设备")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点保存目录")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="检查点保存间隔")
    parser.add_argument("--max_checkpoints", type=int, default=5, help="最大检查点数量")
    parser.add_argument("--resume", action="store_true", help="从最近的检查点恢复训练")
    
    args = parser.parse_args()
    train_with_long_horizon(args) 