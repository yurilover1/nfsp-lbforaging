import csv
import json
import logging
import os
import random
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

from utils import teammate_generate

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def update_progress_bar(
    pbar: tqdm,
    team_reward: float,
    recent_rewards: List[float],
    batch_rewards: List[float],
    batch_start_time: float
) -> None:
    """
    更新进度条。
    Args:
        pbar (tqdm): 进度条对象。
        team_reward (float): 当前回合团队奖励。
        recent_rewards (List[float]): 最近奖励。
        batch_rewards (List[float]): 当前batch奖励。
        batch_start_time (float): 批次开始时间。
    Returns:
        None
    """
    if len(recent_rewards) > 100:
        recent_rewards.pop(0)
    pbar.set_postfix({
        'mean100': f'{np.mean(recent_rewards):.3f}',
        'batch_avg': f'{np.mean(batch_rewards):.3f}',
        'elapsed_time': f'{time.time() - batch_start_time:.1f}s'
    })
    pbar.update(1)

def run_episode(env, agents, render: bool = False) -> Tuple[Any, float, int, Dict[str, Any]]:
    """
    执行一个完整回合，收集轨迹数据并添加到agent的buffer中。
    Args:
        env: 环境对象。
        agents: 智能体列表。
        render (bool): 是否渲染。
    Returns:
        轨迹, 团队奖励, 步数, 奖励细节
    """
    # 参考rollout_and_train的逻辑，手动执行游戏循环并收集轨迹数据
    reset_result, _ = env.reset()
    if isinstance(reset_result, tuple):
        obs = reset_result[0] if len(reset_result) > 0 else reset_result
    else:
        obs = reset_result
    
    done = False
    step = 0
    episode_reward = 0
    episode_trajs = []
    max_steps = 200  # 最大步数限制
    
    # 获取主智能体（第一个智能体）
    main_agent = agents[0] if agents else None
    teammate = agents[1] if len(agents) > 1 else None
    
    while not done and step < max_steps:
        try:
            # 处理观察数据
            if isinstance(obs, list) and len(obs) >= 2:
                main_obs = obs[0]
                teammate_obs = obs[1] if len(obs) > 1 else obs[0]
            else:
                main_obs = obs
                teammate_obs = obs
            
            # 主智能体选择动作
            if main_agent and hasattr(main_agent, 'select_action'):
                main_action = main_agent.select_action(main_obs)
                # 将状态-动作对添加到SL memory中
                if hasattr(main_agent, 'sl_memory') and hasattr(main_agent, '_preprocess_state'):
                    try:
                        processed_state = main_agent._preprocess_state(main_obs)
                        main_agent.sl_memory.add((processed_state, main_action))
                    except:
                        pass  # 如果预处理失败，跳过
            else:
                main_action = 0  # 默认动作
            
            # 队友选择动作
            if teammate and hasattr(teammate, 'step'):
                teammate_action = teammate.step(teammate_obs)
            else:
                teammate_action = 0  # 默认动作
            
            # 执行动作
            actions = [main_action, teammate_action]
            step_result = env.step(actions)
            
            # 解析step结果
            if isinstance(step_result, tuple):
                if len(step_result) == 5:
                    next_obs, reward, done, truncated, info = step_result
                elif len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                    truncated = False
                elif len(step_result) == 3:
                    next_obs, reward, done = step_result
                    truncated, info = False, {}
                else:
                    break
            else:
                break
            
            if render:
                env.render()
            
            episode_reward += reward
            
            # 收集轨迹数据（针对主智能体）
            if main_agent:
                try:
                    # 处理观察数据格式
                    if isinstance(main_obs, list):
                        obs_array = main_obs
                    else:
                        obs_array = main_obs
                    
                    if isinstance(next_obs, list) and len(next_obs) >= 1:
                        next_obs_array = next_obs[0]
                    else:
                        next_obs_array = next_obs
                    
                    # 获取有效动作列表
                    valid_actions = getattr(env, '_valid_actions', [[0, 1, 2, 3, 4, 5]])[0] if hasattr(env, '_valid_actions') else list(range(6))
                    
                    # 创建轨迹数据格式，参考rollout_and_train的格式
                    obs_dict = {
                        'obs': obs_array, 
                        'actions': list(a.value if hasattr(a, 'value') else a for a in valid_actions)
                    }
                    next_obs_dict = {
                        'obs': next_obs_array, 
                        'actions': list(a.value if hasattr(a, 'value') else a for a in valid_actions)
                    }
                    
                    traj = [obs_dict, main_action, reward, next_obs_dict, done]
                    episode_trajs.append(traj)
                except Exception as e:
                    # 如果轨迹数据收集失败，跳过
                    pass
            
            obs = next_obs
            step += 1
            
        except Exception as e:
            logger.warning(f"Episode step {step} 执行失败: {e}")
            break
    
    # 将收集的轨迹数据添加到主智能体的buffer中
    if main_agent and episode_trajs:
        try:
            for traj in episode_trajs:
                if hasattr(main_agent, 'add_traj2buffer'):
                    main_agent.add_traj2buffer(traj)
                elif hasattr(main_agent, 'rl_agent') and hasattr(main_agent.rl_agent, 'add_traj2buffer'):
                    main_agent.rl_agent.add_traj2buffer(traj)
        except Exception as e:
            logger.warning(f"轨迹数据添加到buffer失败: {e}")
    
    # 构造奖励细节
    reward_detail = {
        'total': episode_reward,
        'base': episode_reward if episode_reward > 0 else 0.0,
        'attraction': 0.0,  # 简化处理
        'step': -0.01 * step  # 步数惩罚
    }
    
    return episode_trajs, episode_reward, step, reward_detail

def train_agents(
    env,
    agent,
    num_episodes: int = 5000,
    eval_interval: int = 100,
    render: bool = False,
    render_interval: int = 100,
    layer_num: int = 7,
    train_freq: int = 50,  # 降低训练频率，从200改为50
    batch_size: int = 500   # 减少批次大小，从1000改为500
) -> Dict[str, Any]:
    """
    训练智能体与预加载队友合作。
    Args:
        env: 环境对象。
        agent: 主智能体。
        num_episodes (int): 训练回合数。
        eval_interval (int): 
        render (bool): 是否渲染（已禁用）。
        render_interval (int): 渲染间隔（已禁用）。
        layer_num (int): 网络层数。
        train_freq (int): 训练频率。
        batch_size (int): 每批次回合数。
    Returns:
        history (dict): 训练历史。
    """
    history = {
        'episode_rewards': [],
        'eval_rewards': [],
        'eval_batches': [],
        'exploitability': [],
        'sl_losses': [],
        'rl_losses': [],
        'policy_accuracies': [],
    }
    os.makedirs("./results", exist_ok=True)
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train_debug.csv')
    csv_header = [
        'episode', 'reward_total', 'reward_base', 'reward_attraction', 'reward_step',
        'rl_loss', 'actor_loss', 'critic_loss', 'sl_loss', 'policy_entropy', 'entropy_history', 'buffer_size',
        'eta', 'policy_mode', 'param_mean', 'param_std', 'steps'
    ]
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        recent_rewards: List[float] = []
        total_batches = (num_episodes + train_freq - 1) // train_freq  # 按train_freq分批
        logger.info(f"\n开始训练 - 总共 {num_episodes} 回合 ({total_batches} 批次)...\n")
        
        # 预加载8个队友
        teammate_pool = [teammate_generate(6, device=agent.device, id=i) for i in range(8)]
        
        # 统计变量
        episode_count = 0
        
        for batch_idx in range(total_batches):
            batch_start_episode = batch_idx * train_freq
            batch_end_episode = min(batch_start_episode + train_freq, num_episodes)
            current_batch_size = batch_end_episode - batch_start_episode
            
            batch_start_time = time.time()
            batch_rewards: List[float] = []
            batch_steps: List[int] = []
            batch_actor_losses: List[float] = []
            batch_critic_losses: List[float] = []
            batch_total_losses: List[float] = []
            
            # 清空PPO buffer开始新的batch
            if hasattr(agent, 'rl_agent'):
                agent.rl_agent.clear_trajectory()
            
            logger.info(f"开始批次 {batch_idx+1}/{total_batches} (回合 {batch_start_episode+1}-{batch_end_episode})...")
            
            with tqdm(total=current_batch_size, desc=f"Batch {batch_idx+1}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for i in range(current_batch_size):
                    episode = batch_start_episode + i
                    episode_count += 1
                    
                    # 随机选择队友
                    teammate = random.choice(teammate_pool)
                    
                    # 执行游戏并收集轨迹数据（run_episode已经处理了轨迹数据收集）
                    try:
                        agents = [agent, teammate]
                        trajectory, team_reward, steps, reward_detail = run_episode(env, agents, render=render)
                        
                        history['episode_rewards'].append(team_reward)
                        batch_rewards.append(team_reward)
                        batch_steps.append(steps)
                        recent_rewards.append(team_reward)
                        
                        update_progress_bar(pbar, team_reward, recent_rewards, batch_rewards, batch_start_time)
                        
                    except Exception as e:
                        logger.warning(f"Episode {episode} 执行失败: {e}")
                        # 使用默认值
                        team_reward = 0.0
                        steps = 0
                        reward_detail = {}
                        batch_rewards.append(team_reward)
                        batch_steps.append(steps)
            
            # 在每个batch结束后执行训练（参考rollout_and_train的逻辑）
            logger.info(f"批次 {batch_idx+1} 数据收集完成，开始训练...")
            
            # 执行PPO训练
            if hasattr(agent, 'rl_agent'):
                buffer_size = len(agent.rl_agent.states) if hasattr(agent.rl_agent, 'states') else 0
                logger.info(f"PPO缓冲区大小: {buffer_size}")
                
                if buffer_size > 0:
                    agent.rl_agent.update()
                    logger.info("PPO训练完成")
                    
                    # 获取训练损失
                    if hasattr(agent.rl_agent, 'actor_losses') and agent.rl_agent.actor_losses:
                        batch_actor_losses = [agent.rl_agent.actor_losses[-1]] * current_batch_size
                    else:
                        batch_actor_losses = [None] * current_batch_size
                        
                    if hasattr(agent.rl_agent, 'critic_losses') and agent.rl_agent.critic_losses:
                        batch_critic_losses = [agent.rl_agent.critic_losses[-1]] * current_batch_size
                    else:
                        batch_critic_losses = [None] * current_batch_size
                        
                    if hasattr(agent.rl_agent, 'losses') and agent.rl_agent.losses:
                        batch_total_losses = [agent.rl_agent.losses[-1]] * current_batch_size
                    else:
                        batch_total_losses = [None] * current_batch_size
                else:
                    logger.warning("PPO缓冲区为空，跳过训练")
                    batch_actor_losses = [None] * current_batch_size
                    batch_critic_losses = [None] * current_batch_size
                    batch_total_losses = [None] * current_batch_size
            
            # 执行SL训练
            if hasattr(agent, 'sl_train'):
                try:
                    agent.sl_train()
                    logger.info("SL训练完成")
                except Exception as e:
                    logger.warning(f"SL训练失败: {e}")
            
            # 记录每个episode的详细日志
            for i, (reward, steps, actor_loss, critic_loss, total_loss) in enumerate(zip(
                batch_rewards, batch_steps, batch_actor_losses, batch_critic_losses, batch_total_losses
            )):
                episode = batch_start_episode + i
                
                # 计算奖励分解
                reward_detail = {} if 'reward_detail' not in locals() else reward_detail
                
                # 统计参数均值/方差
                params = [p.data.cpu().numpy().flatten() for p in agent.rl_agent.actor.parameters() 
                         if hasattr(agent, 'rl_agent') and hasattr(agent.rl_agent, 'actor') and p.requires_grad]
                if params:
                    all_params = np.concatenate(params)
                    param_mean = all_params.mean()
                    param_std = all_params.std()
                else:
                    param_mean = None
                    param_std = None
                
                row = {
                    'episode': episode + 1,
                    'reward_total': reward_detail.get('total', reward) if reward_detail else reward,
                    'reward_base': reward_detail.get('base', 0.0) if reward_detail else 0.0,
                    'reward_attraction': reward_detail.get('attraction', 0.0) if reward_detail else 0.0,
                    'reward_step': reward_detail.get('step', 0.0) if reward_detail else 0.0,
                    'rl_loss': total_loss,
                    'actor_loss': actor_loss,
                    'critic_loss': critic_loss,
                    'sl_loss': agent.losses[-1] if hasattr(agent, 'losses') and agent.losses else None,
                    'policy_entropy': agent.rl_agent.entropies[-1] if hasattr(agent, 'rl_agent') and hasattr(agent.rl_agent, 'entropies') and agent.rl_agent.entropies else None,
                    'entropy_history': json.dumps(agent.rl_agent.entropies[-5:]) if hasattr(agent, 'rl_agent') and hasattr(agent.rl_agent, 'entropies') and agent.rl_agent.entropies else None,
                    'buffer_size': len(agent.rl_agent.states) if hasattr(agent, 'rl_agent') and hasattr(agent.rl_agent, 'states') else None,
                    'eta': getattr(agent, 'eta', None),
                    'policy_mode': getattr(agent, 'policy_mode', None),
                    'param_mean': param_mean,
                    'param_std': param_std,
                    'steps': steps
                }
                writer.writerow(row)
                
                # 更新history
                if total_loss is not None:
                    history['rl_losses'].append(total_loss)
                if hasattr(agent, 'losses') and agent.losses:
                    history['sl_losses'].append(agent.losses[-1])
                if hasattr(agent, 'policy_accuracies') and agent.policy_accuracies:
                    history['policy_accuracies'].append(agent.policy_accuracies[-1])
            
            csvfile.flush()
            
            # 输出批次统计信息
            avg_reward = np.mean(batch_rewards) if batch_rewards else 0.0
            avg_steps = np.mean(batch_steps) if batch_steps else 0.0
            avg_actor_loss = np.mean([x for x in batch_actor_losses if x is not None]) if batch_actor_losses else None
            avg_critic_loss = np.mean([x for x in batch_critic_losses if x is not None]) if batch_critic_losses else None
            avg_total_loss = np.mean([x for x in batch_total_losses if x is not None]) if batch_total_losses else None
            
            logger.info(f"✅ 批次 {batch_idx+1}/{total_batches} 完成 | "
                       f"平均奖励: {avg_reward:.4f} | 平均步数: {avg_steps:.2f} | "
                       f"Actor Loss: {avg_actor_loss:.4f if avg_actor_loss else 'N/A'} | "
                       f"Critic Loss: {avg_critic_loss:.4f if avg_critic_loss else 'N/A'} | "
                       f"Total Loss: {avg_total_loss:.4f if avg_total_loss else 'N/A'} | "
                       f"总进度: {100*(batch_end_episode/num_episodes):.1f}%\n")
            
            # 评估
            if (batch_end_episode) % eval_interval == 0:
                logger.info(f"执行评估 (回合 {batch_end_episode})...")
                # 评估逻辑略
                # ...
        
        # 如果还有剩余的episodes
        if episode_count < num_episodes:
            logger.info(f"处理剩余 {num_episodes - episode_count} 个回合...")
            # 处理剩余episodes的逻辑可以在这里添加
                
    return history
