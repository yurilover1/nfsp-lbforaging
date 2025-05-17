import numpy as np
import time
import os
import matplotlib.pyplot as plt
import gymnasium as gym
import logging
from tqdm import tqdm
from evaluate import evaluate
from utils import save_history, plot_training_curve

logger = logging.getLogger(__name__)

def train_agents(env, agents, num_episodes=5000, eval_interval=100, render=False, render_interval=100,
                 teamate_id=0):
    """训练智能体与预加载的SimpleAgent2队友合作
    
    参数:
        env: 游戏环境
        agents: 智能体列表，第一个为主智能体，第二个为SimpleAgent2
        num_episodes: 训练回合数
        eval_interval: 评估间隔
        render: 是否渲染
        render_interval: 渲染间隔
        teamate_id: 队友ID
    """
    # 判断智能体类型
    nfsp_agents = [agent for agent in agents if agent.name.startswith('NFSP')]
    ppo_agents = [agent for agent in agents if agent.name.startswith('PPO')]
    teammate_agents = [agent for agent in agents if not hasattr(agent, 'add_traj')]
    
    # 训练历史记录
    history = {
        'episode_rewards': [],
        'eval_rewards': [],
        'eval_batches': [],  # 记录每次评估对应的回合数
        'exploitability': [],  # 可利用度记录
        'sl_losses': [],       # 监督学习损失
        'rl_losses': [],       # 强化学习损失
        'policy_accuracies': [], # 策略准确率
    }
    
    # 创建结果目录
    os.makedirs("./results", exist_ok=True)
    
    # 用于记录100回合的奖励
    recent_rewards = []
    
    # 设置初始批次
    total_batches = num_episodes // 100
    
    print(f"\n开始训练 - 总共 {num_episodes} 回合 ({total_batches} 批次)...\n")
    print(f"正在训练 {len(nfsp_agents)} 个NFSP智能体, {len(ppo_agents)} 个PPO智能体与 {len(teammate_agents)} 个SimpleAgent2队友合作\n")

    # 外层循环处理每个批次
    for batch in range(total_batches):
        # 为每个批次创建一个tqdm进度条
        batch_start_time = time.time()
        batch_size = min(100, num_episodes - batch * 100)  # 处理最后一个不完整批次
        
        # 创建当前批次的进度条
        with tqdm(total=batch_size, desc=f"批次 {batch+1}/{total_batches}", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            # 记录当前批次的奖励
            batch_rewards = []
            steps_list = []
            
            # 处理当前批次中的每个回合
            for i in range(batch_size):
                episode = batch * 100 + i
                
                # 在每个回合开始时选择策略模式（仅对NFSP智能体）
                for agent in nfsp_agents:
                    agent.choose_policy_mode()

                # 判断是否需要在本回合渲染
                should_render = render and episode % render_interval == 0
                
                if should_render:
                    print(f"\n渲染回合 {episode}...")
                   
                    # 运行一个完整回合，并渲染
                    trajectories, payoffs, steps = env.run(agents, is_training=True, render=True, 
                                                           sleep_time=0.5)
                    # 暂停进度条更新
                    pbar.clear()

                else:
                    # 正常训练，不渲染
                    trajectories, payoffs, steps = env.run( agents, is_training=True, render=False)
                
                # 记录每个回合的奖励
                history['episode_rewards'].append(payoffs)
                steps_list.append(steps)
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
                for j, agent in enumerate(nfsp_agents + ppo_agents):
                    agent_idx = agents.index(agent)  # 找到当前NFSP智能体在agents列表中的索引
                    for ts in trajectories[agent_idx]:
                        if len(ts) > 0:
                            # ts结构：[obs_dict, action, reward, next_obs_dict, done]
                            # 首先获取预处理后的状态
                            obs = agent._preprocess_state(ts[0])
                            action = ts[1]
                            reward = ts[2]
                            next_obs = None if ts[4] else agent._preprocess_state(ts[3])
                            done = ts[4]
                            
                            # 如果是终止状态，则使用当前状态作为下一状态
                            if next_obs is None:
                                next_obs = obs
                            
                            # 直接添加轨迹，_preprocess_state中会处理数据格式转换
                            agent.add_traj([obs, action, reward, next_obs, done])

                    agent.train()
                
                # 每10个回合保存损失和准确率
                if episode % 10 == 0 and len(nfsp_agents) > 0:
                    # 使用第一个NFSP智能体的损失数据
                    agent = nfsp_agents[0]
                    # 保存SL损失
                    if len(agent.losses) > 0:
                        history['sl_losses'].append(agent.losses[-1])
                    # 保存RL损失
                    if len(agent.RLlosses) > 0:
                        history['rl_losses'].append(agent.RLlosses[-1])
                    # 保存策略准确率
                    if hasattr(agent, 'policy_accuracies') and len(agent.policy_accuracies) > 0:
                        history['policy_accuracies'].append(agent.policy_accuracies[-1])

            
            
            # 批次完成后显示平均奖励
            avg_reward = np.mean(recent_rewards)
            elapsed_time = time.time() - batch_start_time
            avg_steps = np.mean(steps_list)
            # 关闭当前进度条
            pbar.close()
            
            # 打印批次完成信息（使用彩色文本和表情符号使其更明显）
            batch_summary = (f"✅ 批次 {batch+1}/{total_batches} 完成 | "
                             f"平均奖励: {avg_reward:.4f} | "
                             f"用时: {elapsed_time:.1f}秒 | "
                             f"步数：{avg_steps} | "
                             f"总进度: {(batch+1)/total_batches*100:.1f}%")
            print(f"\033[92m{batch_summary}\033[0m\n")
            
            # 重置recent_rewards列表
            recent_rewards = []
            
            # 评估
            if batch % 1 == 0:
                print(f"\n执行评估 ( 批次 {batch})...")
                
                # 评估智能体性能
                eval_result = evaluate(
                    env, 
                    agents, 
                    eval_episodes=100,  # 评估回合数
                    calculate_exploitability=False
                )
                rewards = eval_result
                
                # 记录评估结果
                history['eval_rewards'].append(rewards)
                history['eval_batches'].append(batch)
                
                # 打印评估结果
                print(f"评估结果 (批次 {batch}):")
                print(f"团队平均奖励: {np.mean(rewards):.4f}")
            print("-" * 50)
    
    print("\n训练完成！\n")
    
    # 保存模型（NFSP智能体）
    for agent in nfsp_agents:
        agent.save_models(teamate_id)
    
    # 保存模型（PPO智能体）
    for i, agent in enumerate(ppo_agents):
        model_path = os.path.join("./models", f"ppo_agent_{teamate_id}_model.pt")
        agent.save_model(model_path)
        print(f"PPO智能体模型已保存到 {model_path}")
    
    # 绘制训练曲线
    plot_training_curve(history, num_episodes, eval_interval, 
                        nfsp_agents + ppo_agents, teamate_id, 
                        type='ppo' if len(ppo_agents) > 0 else 'nfsp')
    
    # 保存训练历史记录
    save_history(history, nfsp_agents + ppo_agents)
    
    return agents

