import numpy as np
import time
import os
import matplotlib.pyplot as plt
import gymnasium as gym
import logging
from tqdm import tqdm
from evaluate import evaluate
from random import randint
from utils import save_history, plot_training_curve, teammate_generate

logger = logging.getLogger(__name__)

def update_progress_bar(pbar, payoffs, recent_rewards, batch_rewards, batch_start_time):
    """更新进度条显示
    
    参数:
        pbar: tqdm进度条对象
        team_reward: 当前回合的团队奖励
        batch_rewards: 当前批次的奖励列表
        batch_start_time: 批次开始时间
    """
    # 计算当前回合的团队奖励
    team_reward = sum(payoffs)
    recent_rewards.append(team_reward)
    batch_rewards.append(team_reward)
    
    # 计算当前批次的平均奖励
    avg_batch_reward = sum(batch_rewards) / len(batch_rewards)
    
    # 计算当前批次所用时间
    elapsed_time = time.time() - batch_start_time
    
    # 更新进度条
    pbar.set_postfix({
        '奖励': f'{team_reward:.2f}', 
        '平均': f'{avg_batch_reward:.2f}',
        '用时': f'{elapsed_time:.1f}s'
    })
    pbar.update(1)

def train_agents(env, agent, num_episodes=5000, eval_interval=100, render=False, render_interval=100, layer_num=7):
    """训练智能体与预加载的SimpleAgent2队友合作
    
    参数:
        env: 游戏环境
        agent: 主智能体
        num_episodes: 训练回合数
        eval_interval: 评估间隔
        render: 是否渲染
        render_interval: 渲染间隔
    """
   
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
            
            agents = [agent, teammate_generate(6, device='cpu',  id = randint(0, 7))]
            
            # 处理当前批次中的每个回合
            for i in range(batch_size):
                episode = batch * 100 + i

                # 判断是否需要在本回合渲染
                should_render = render and episode % render_interval == 0

                agent.choose_policy_mode()
                
                if should_render:
                    print(f"\n渲染回合 {episode}...")
                   
                    # 运行一个完整回合，并渲染
                    _, payoffs, steps = env.run(agents, is_training=True, render=True, 
                                                           sleep_time=0.5)
                    # 暂停进度条更新
                    pbar.clear()

                else:
                    # 正常运行，不渲染
                    _, payoffs, steps = env.run(agents, is_training=True, render=False)
                
                # 记录每个回合的奖励
                history['episode_rewards'].append(payoffs)
                steps_list.append(steps)
                
                # 更新进度条
                update_progress_bar(pbar, payoffs, recent_rewards, batch_rewards, batch_start_time)

                if episode % agent.train_freq == 0:
                    agent.rl_train(5)
                    # agent.sl_train(50)



            # 关闭当前进度条
            pbar.close()
            
            # 打印批次完成信息（使用彩色文本和表情符号使其更明显）
            batch_summary = (f"✅ 批次 {batch+1}/{total_batches} 完成 | "
                             f"平均奖励: {np.mean(recent_rewards):.4f} | "
                             f"步数：{np.mean(steps_list)} | "
                             f"总进度: {(batch+1)/total_batches*100:.1f}%")
            print(f"\033[92m{batch_summary}\033[0m\n")
            
            # 重置recent_rewards列表
            recent_rewards = []
            
            # 评估
            print(f"\n执行评估 ( 批次 {batch})...")
            
            rewards = evaluate(env, agents)
            
            # 记录评估结果
            history['eval_rewards'].append(rewards)
            history['eval_batches'].append(batch)
            
            # 打印评估结果
            print(f"评估结果 (批次 {batch}):")
            print(f"团队平均奖励: {np.sum(rewards):.4f}")
            print("-" * 50)
    
    print("\n训练完成！\n")

    # 保存模型（NFSP智能体）
    agent.save_models()

    # 保存训练历史记录
    history['rl_losses'].extend(agent.RLlosses)
    save_history(history, agent, layer_num=layer_num)
    
    return agent

