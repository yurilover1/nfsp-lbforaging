import numpy as np
import time
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import logging
from tqdm import tqdm

# 添加从nfsp_run.py移动的函数
logger = logging.getLogger(__name__)

def calculate_state_size(env):
    """计算环境的状态大小"""
    try:
        # 获取一个示例观测
        obs, _ = env.reset()
        
        # 如果观测是元组（多智能体环境），取第一个智能体的观测
        if isinstance(obs, tuple):
            first_obs = obs[0]
        else:
            first_obs = obs
            
        # 检查观测形状以判断观测模式
        if isinstance(first_obs, np.ndarray):
            # 检查是否为三维数组
            if len(first_obs.shape) == 3:
                # 检查是否为三层观测模式 [3, 5, 5]
                if first_obs.shape[0] == 3 and first_obs.shape[1] == 5 and first_obs.shape[2] == 5:
                    print(f"检测到三层观测模式: 形状={first_obs.shape}")
                    return 3 * 5 * 5  # 返回展平后的大小
                
                # 检查是否为普通网格观测模式
                print(f"检测到网格观测模式: 形状={first_obs.shape}")
                return first_obs.size  # 返回展平后的大小
            
            # 普通一维观测
            print(f"检测到普通观测模式: 形状={first_obs.shape}")
            return first_obs.size
            
        # 尝试从环境获取observation_space
        if hasattr(env, 'observation_space'):
            # 尝试获取第一个观测空间的形状
            obs_shape = env.observation_space[0].shape
            print(f"从observation_space获取的形状: {obs_shape}")
            
            # 如果是多维形状
            if isinstance(obs_shape, tuple) and len(obs_shape) > 1:
                # 计算总大小
                return np.prod(obs_shape)
            elif isinstance(obs_shape, tuple) and len(obs_shape) > 0:
                return obs_shape[0]
        
        # 默认状态大小
        print("无法确定状态大小，使用默认值100")
        return 100
        
    except Exception as e:
        print(f"计算状态大小时出错: {e}")
        # 默认状态大小
        return 100

def evaluate(env, agents, num_episodes=100, calculate_exploitability=False, eval_env=None):
    """
    评估智能体性能
    
    参数:
        env: 游戏环境
        agents: 要评估的智能体列表
        num_episodes: 评估回合数
        calculate_exploitability: 是否计算可利用度
        eval_env: 用于评估的环境，如果为None，则使用原环境的副本
    
    返回:
        如果calculate_exploitability为False，返回每个智能体的平均奖励；
        否则，返回(平均奖励，可利用度)元组
    """
    if eval_env is None:
        # 创建一个新的环境用于评估
        eval_env = gym.make(env.unwrapped.spec.id, render_mode=None)
    
    # 检查是否可以使用简化的评估方法
    if hasattr(eval_env, 'run'):
        # 使用环境的run方法执行评估
        total_rewards = np.zeros(len(agents))
        for _ in range(num_episodes):
            _, payoffs = eval_env.run(agents, is_training=False)
            total_rewards += payoffs
            
        # 计算平均奖励
        avg_rewards = total_rewards / num_episodes
        
        # 如果需要计算可利用度
        if calculate_exploitability and hasattr(agents[0], 'evaluate_team_exploitability'):
            # 使用第一个智能体的方法评估团队可利用度
            exploitability, _ = agents[0].evaluate_team_exploitability(eval_env, agents, num_episodes=num_episodes//2)
            return avg_rewards, exploitability
        
        return avg_rewards
    

def train_agents(env, nfsp_agents, num_episodes=5000, eval_interval=100, render=False, render_interval=100):
    """训练NFSP智能体"""
    # 用于评估的环境，传递与主环境相同的渲染模式
    eval_env = gym.make("Foraging-6x6-2p-3f-v3", sight=3, 
                   grid_observation=False,  three_layer_obs=True, render_mode=None)
    
    # 训练历史记录
    history = {
        'episode_rewards': [],
        'eval_rewards': [],
        'eval_episodes': [],  # 记录每次评估对应的回合数
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
    current_batch = 0
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
            
            # 处理当前批次中的每个回合
            for i in range(batch_size):
                episode = batch * 100 + i
                
                # 在每个回合开始时选择策略模式
                for agent in nfsp_agents:
                    agent.choose_policy_mode()

                # 判断是否需要在本回合渲染
                should_render = render and episode % render_interval == 0
                
                if should_render:
                    print(f"\n渲染回合 {episode}...")
                   
                    # 运行一个完整回合，并渲染
                    trajectories, payoffs = env.run(
                        nfsp_agents,
                        is_training=True,
                        render=True, 
                        sleep_time=0.5
                    )
                    # 暂停进度条更新
                    pbar.clear()

                else:
                    # 正常训练，不渲染
                    trajectories, payoffs = env.run(
                        nfsp_agents, 
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
                for j in range(env.n_agents):
                    for ts in trajectories[j]:
                        if len(ts) > 0:
                            # ts结构：[obs_dict, action, reward, next_obs_dict, done]
                            # 首先获取预处理后的状态
                            obs = nfsp_agents[j]._preprocess_state(ts[0])
                            action = ts[1]
                            reward = ts[2]
                            next_obs = None if ts[4] else nfsp_agents[j]._preprocess_state(ts[3])
                            done = ts[4]
                            
                            # 如果是终止状态，则使用当前状态作为下一状态
                            if next_obs is None:
                                next_obs = obs
                            
                            # 直接添加轨迹，_preprocess_state中会处理数据格式转换
                            nfsp_agents[j].add_traj([obs, action, reward, next_obs, done])
                            nfsp_agents[j].train()
                
                # 每10个回合保存损失和准确率
                if episode % 10 == 0 and len(nfsp_agents) > 0:
                    # 使用第一个智能体的损失数据
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
                
                # 定期评估
                if episode % eval_interval == 0 or episode == num_episodes - 1:
                    try:
                        pbar.clear()  # 暂时清除进度条显示
                        
                        # 计算普通奖励和可利用度
                        eval_rewards, exploitability = evaluate(
                            env, 
                            nfsp_agents, 
                            num_episodes=10, 
                            calculate_exploitability=True,
                            eval_env=eval_env
                        )
                        
                        # 确保eval_rewards是一个单一值，并记录当前回合数
                        agent0_reward = eval_rewards[0]
                        history['eval_rewards'].append(agent0_reward)
                        history['eval_episodes'].append(episode)  # 记录评估回合
                        history['exploitability'].append(exploitability)  # 记录可利用度
                        
                        # 打印评估信息
                        # print(f"\n评估 - 回合 {episode}/{num_episodes}: 奖励={eval_rewards}, 可利用度={exploitability:.4f}")
                        
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
            
            # 打印批次完成信息（使用彩色文本和表情符号使其更明显）
            batch_summary = f"✅ 批次 {batch+1}/{total_batches} 完成 | 平均奖励: {avg_reward:.4f} | 用时: {elapsed_time:.1f}秒 | 总进度: {(batch+1)/total_batches*100:.1f}%"
            print(f"\033[92m{batch_summary}\033[0m\n")
            
            # 重置recent_rewards列表
            recent_rewards = []
    
    print("\n训练完成！\n")
    
    # 保存模型
    for agent in nfsp_agents:
        agent.save_models()
    
    # 绘制训练曲线
    plot_training_curve(history, num_episodes, eval_interval, nfsp_agents)
    
    # 保存训练历史记录
    save_history(history, nfsp_agents)
    
    return nfsp_agents

def plot_training_curve(history, num_episodes, eval_interval, nfsp_agents=None):
    """绘制训练曲线，包括四幅图：监督学习损失、策略准确率、强化学习损失和队伍总奖励"""
    # 处理中文显示问题
    use_english = True
    plt.figure(figsize=(16, 16))
    
    # 1. 监督学习损失 (左上角)
    plt.subplot(2, 2, 1)
    
    if nfsp_agents and len(nfsp_agents) > 0:
        
        # 计算所有智能体的平均损失
        all_losses = []
        max_length = 0
        
        # 确定最大长度并收集数据
        for agent in nfsp_agents:
            if len(agent.losses) > max_length:
                max_length = len(agent.losses)
        
        # 初始化累积数组
        if max_length > 0:
            avg_losses = np.zeros(max_length)
            count = np.zeros(max_length)
            
            # 累加每个智能体的损失
            for agent in nfsp_agents:
                losses = np.array(agent.losses)
                avg_losses[:len(losses)] += losses
                count[:len(losses)] += 1
            
            # 计算平均值（避免除零错误）
            valid_indices = count > 0
            avg_losses[valid_indices] /= count[valid_indices]
            
            # 降采样
            if len(avg_losses) > 1000:
                step = len(avg_losses) // 1000
                avg_losses = avg_losses[::step]
            
            # 平滑处理
            window_size = min(20, len(avg_losses))
            if window_size > 1:
                smooth_avg = np.convolve(avg_losses, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smooth_avg, 'k-', linewidth=2, 
                         label='Average SL Loss' if use_english else '平均SL损失')
                
        plt.xlabel('Training Steps' if use_english else '训练步数')
        plt.ylabel('Loss' if use_english else '损失')
        plt.title('Supervised Learning Loss' if use_english else '监督学习损失')
        plt.grid(True)
        plt.legend()
    
    # 2. 策略准确率 (右上角)
    plt.subplot(2, 2, 2)
    
    if nfsp_agents and len(nfsp_agents) > 0:
        # 如果智能体数量较多，也绘制平均准确率
        if len(nfsp_agents) > 1:
            # 计算所有智能体的平均准确率
            max_length = 0
            
            # 确定最大长度
            for agent in nfsp_agents:
                if hasattr(agent, 'policy_accuracies') and len(agent.policy_accuracies) > max_length:
                    max_length = len(agent.policy_accuracies)
            
            # 初始化累积数组
            if max_length > 0:
                avg_acc = np.zeros(max_length)
                count = np.zeros(max_length)
                
                # 累加每个智能体的准确率
                for agent in nfsp_agents:
                    if hasattr(agent, 'policy_accuracies'):
                        acc = np.array(agent.policy_accuracies)
                        avg_acc[:len(acc)] += acc
                        count[:len(acc)] += 1
                
                # 计算平均值
                valid_indices = count > 0
                avg_acc[valid_indices] /= count[valid_indices]
                
                # 降采样
                if len(avg_acc) > 1000:
                    step = len(avg_acc) // 1000
                    avg_acc = avg_acc[::step]
                
                # 平滑处理
                window_size = min(20, len(avg_acc))
                if window_size > 1:
                    smooth_avg = np.convolve(avg_acc, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(smooth_avg, 'k-', linewidth=2, 
                             label='Average Accuracy' if use_english else '平均准确率')
        
        plt.xlabel('Training Steps' if use_english else '训练步数')
        plt.ylabel('Accuracy' if use_english else '准确率')
        plt.title('Policy Accuracy' if use_english else '策略准确率')
        plt.grid(True)
        plt.legend()
    
    # 3. 强化学习损失 (左下角)
    plt.subplot(2, 2, 3)
    
    if nfsp_agents and len(nfsp_agents) > 0:
        
        # 计算所有智能体的平均RL损失
        max_length = 0
        
        # 确定最大长度
        for agent in nfsp_agents:
            if len(agent.RLlosses) > max_length:
                max_length = len(agent.RLlosses)
        
        # 初始化累积数组
        if max_length > 0:
            avg_rl = np.zeros(max_length)
            count = np.zeros(max_length)
            
            # 累加每个智能体的RL损失
            for agent in nfsp_agents:
                rl = np.array(agent.RLlosses)
                avg_rl[:len(rl)] += rl
                count[:len(rl)] += 1
            
            # 计算平均值
            valid_indices = count > 0
            avg_rl[valid_indices] /= count[valid_indices]
            
            # 降采样
            if len(avg_rl) > 1000:
                step = len(avg_rl) // 1000
                avg_rl = avg_rl[::step]
            
            # 平滑处理
            window_size = min(20, len(avg_rl))
            if window_size > 1:
                smooth_avg = np.convolve(avg_rl, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smooth_avg, 'k-', linewidth=2, 
                         label='Average RL Loss' if use_english else '平均RL损失')
        
        plt.xlabel('Training Steps' if use_english else '训练步数')
        plt.ylabel('Loss' if use_english else '损失')
        plt.title('Reinforcement Learning Loss' if use_english else '强化学习损失')
        plt.grid(True)
        plt.legend()
    
    # 4. 队伍总奖励 (右下角)
    plt.subplot(2, 2, 4)
    raw_rewards = history['episode_rewards']
    
    if len(raw_rewards) > 0:
        # 检查数据格式并转换为numpy数组
        if isinstance(raw_rewards[0], (list, np.ndarray)):
            # 多智能体情况，绘制团队总奖励
            rewards = np.array([reward for reward in raw_rewards])
            team_rewards = rewards.sum(axis=1)
             # 平滑处理团队奖励
            window_size = min(50, len(team_rewards) // 10)  # 使用更大的窗口来平滑奖励曲线
            if window_size > 1:
                smooth_rewards = np.convolve(team_rewards, np.ones(window_size)/window_size, mode='valid')
                # 画出原始数据（较浅的颜色）
                plt.plot(np.arange(len(team_rewards)), team_rewards, 'b-', alpha=0.3, 
                         label='Raw Team Reward' if use_english else '原始队伍奖励')
                # 画出平滑后的数据（较深的颜色）
                plt.plot(np.arange(len(smooth_rewards)) + window_size//2, smooth_rewards, 'b-', linewidth=2, 
                         label='Smoothed Team Reward' if use_english else '平滑队伍奖励')
            else:
                # 如果数据量太少，无法平滑，就直接绘制原始数据
                plt.plot(np.arange(len(team_rewards)), team_rewards, 'b-', 
                         label='Team Total Reward' if use_english else '队伍总奖励')
            
        else:
            # 单智能体情况
            rewards = np.array(raw_rewards)
            plt.plot(np.arange(len(rewards)), rewards, 'b-', 
                     label='Agent Reward' if use_english else '智能体奖励')
    
    plt.xlabel('Episodes' if use_english else '回合')
    plt.ylabel('Reward' if use_english else '奖励')
    plt.title('Team Total Reward' if use_english else '队伍总奖励')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./results/training_curve.png')
    plt.close()

def save_history(history, nfsp_agents):
    """保存训练历史数据"""
    # 创建保存目录
    os.makedirs("./results", exist_ok=True)
    
    # 准备保存数据
    data_to_save = {
        'episode_rewards': history['episode_rewards'],
        'eval_rewards': history['eval_rewards'],
        'eval_episodes': history['eval_episodes'],
    }
    
    # 添加可利用度数据（如果存在）
    if 'exploitability' in history and history['exploitability']:
        data_to_save['exploitability'] = history['exploitability']
    
    # 添加监督学习损失
    if 'sl_losses' in history and history['sl_losses']:
        data_to_save['sl_losses'] = history['sl_losses']
        
    # 添加强化学习损失
    if 'rl_losses' in history and history['rl_losses']:
        data_to_save['rl_losses'] = history['rl_losses']
        
    # 添加策略准确率
    if 'policy_accuracies' in history and history['policy_accuracies']:
        data_to_save['policy_accuracies'] = history['policy_accuracies']
    
    # 添加合作指标数据
    if 'coop_metrics' in history and history['coop_metrics']:
        # 将字典列表转换为结构化的数据格式
        coop_metrics = history['coop_metrics']
        
        # 提取各个指标
        episodes = [m['episode'] for m in coop_metrics]
        coop_potential = [m['coop_potential'] for m in coop_metrics]
        coop_food_count = [m['coop_food_count'] for m in coop_metrics]
        player_clusters = [m['player_clusters'] for m in coop_metrics]
        avg_cluster_size = [m['avg_cluster_size'] for m in coop_metrics]
        food_remaining = [m['food_remaining'] for m in coop_metrics]
        food_count = [m['food_count'] for m in coop_metrics]
        
        # 添加到保存数据中
        data_to_save['coop_episodes'] = episodes
        data_to_save['coop_potential'] = coop_potential
        data_to_save['coop_food_count'] = coop_food_count
        data_to_save['player_clusters'] = player_clusters
        data_to_save['avg_cluster_size'] = avg_cluster_size
        data_to_save['food_remaining'] = food_remaining
        data_to_save['food_count'] = food_count
        
        # 单独保存合作指标数据，方便后续分析
        np.savez('./results/coop_metrics.npz', 
                episodes=episodes,
                coop_potential=coop_potential,
                coop_food_count=coop_food_count,
                player_clusters=player_clusters,
                avg_cluster_size=avg_cluster_size,
                food_remaining=food_remaining,
                food_count=food_count)
    
    # 保存为numpy格式
    try:
        np.savez('./results/training_history.npz', **data_to_save)
        print("训练历史保存成功: ./results/training_history.npz")
        if 'coop_metrics' in history and history['coop_metrics']:
            print("合作指标数据保存成功: ./results/coop_metrics.npz")
    except Exception as e:
        print(f"保存训练历史时出错: {e}")
        import traceback
        traceback.print_exc()

def plot_cooperation_metrics(history=None, file_path=None):
    """
    绘制合作指标图表
    
    参数:
        history: 训练历史字典，包含合作指标
        file_path: 已保存的合作指标数据文件路径
    """
    # 处理中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 加载数据
    if history and 'coop_metrics' in history and history['coop_metrics']:
        metrics = history['coop_metrics']
        episodes = [m['episode'] for m in metrics]
        coop_potential = [m['coop_potential'] for m in metrics]
        coop_food_count = [m['coop_food_count'] for m in metrics]
        player_clusters = [m['player_clusters'] for m in metrics]
        avg_cluster_size = [m['avg_cluster_size'] for m in metrics]
    elif file_path:
        try:
            data = np.load(file_path)
            episodes = data['episodes']
            coop_potential = data['coop_potential']
            coop_food_count = data['coop_food_count']
            player_clusters = data['player_clusters']
            avg_cluster_size = data['avg_cluster_size']
        except Exception as e:
            print(f"加载合作指标数据时出错: {e}")
            return
    else:
        print("未提供合作指标数据")
        return
    
    # 创建图表
    plt.figure(figsize=(16, 12))
    
    # 1. 合作潜力与食物数量
    plt.subplot(2, 2, 1)
    plt.plot(episodes, coop_potential, 'r-', label='合作潜力')
    plt.plot(episodes, coop_food_count, 'b-', label='需合作食物数')
    plt.xlabel('回合')
    plt.ylabel('值')
    plt.title('合作潜力与需要合作的食物数量')
    plt.grid(True)
    plt.legend()
    
    # 2. 玩家聚集组数
    plt.subplot(2, 2, 2)
    plt.plot(episodes, player_clusters, 'g-', label='玩家聚集组数')
    plt.xlabel('回合')
    plt.ylabel('组数')
    plt.title('玩家聚集组数变化')
    plt.grid(True)
    plt.legend()
    
    # 3. 平均聚集组大小
    plt.subplot(2, 2, 3)
    # 过滤掉0值（可能表示没有聚集）
    filtered_episodes = []
    filtered_sizes = []
    for i, size in enumerate(avg_cluster_size):
        if size > 0:
            filtered_episodes.append(episodes[i])
            filtered_sizes.append(size)
    
    plt.plot(filtered_episodes, filtered_sizes, 'm-', label='平均组大小')
    plt.xlabel('回合')
    plt.ylabel('平均组大小')
    plt.title('玩家聚集组平均大小')
    plt.grid(True)
    plt.legend()
    
    # 4. 合作指标随时间变化的平滑曲线
    plt.subplot(2, 2, 4)
    window_size = min(50, len(episodes) // 10) if len(episodes) > 50 else 10
    
    # 平滑处理
    def smooth(data, window_size):
        if window_size < 2:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    if window_size >= 2:
        smooth_episodes = episodes[window_size-1:]
        smooth_food_count = smooth(coop_food_count, window_size)
        smooth_clusters = smooth(player_clusters, window_size)
        
        # 计算协作率：聚集组数/需要合作的食物数
        coop_rate = []
        for i in range(len(coop_food_count)):
            if coop_food_count[i] > 0:
                coop_rate.append(min(player_clusters[i] / coop_food_count[i], 1.0))
            else:
                coop_rate.append(0)
        
        smooth_coop_rate = smooth(coop_rate, window_size)
        
        plt.plot(smooth_episodes, smooth_food_count, 'b-', label='平滑食物数')
        plt.plot(smooth_episodes, smooth_clusters, 'g-', label='平滑聚集组数')
        plt.plot(smooth_episodes, smooth_coop_rate, 'r-', label='协作率')
        plt.xlabel('回合')
        plt.ylabel('值')
        plt.title('合作指标趋势分析')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('./results/cooperation_metrics.png')
    print("合作指标图表已保存: ./results/cooperation_metrics.png")
    plt.close()

def test_agents(env, nfsp_agents, eval_episodes=100, eval_explo=False, render=True, num_demo_episodes=5):
    """
    测试NFSP智能体性能
    
    参数:
        env: 游戏环境
        nfsp_agents: 要测试的NFSP智能体列表
        eval_episodes: 评估回合数
        eval_explo: 是否评估团队可利用度
        render: 是否渲染演示回合
        num_demo_episodes: 演示回合数量
        
    返回:
        测试结果字典，包含平均奖励、智能体奖励和可利用度（如果计算）
    """
    print("\n测试模式 - 加载预训练模型并展示性能...\n")
    
    # 加载预训练模型
    loaded_models = []
    for agent in nfsp_agents:
        if agent.load_models():
            loaded_models.append("成功")
        else:
            loaded_models.append("失败")
    
    print(f"模型加载状态: {loaded_models}")
    
    results = {}
    
    # 如果指定了评估可利用度
    if eval_explo:
        print("\n评估团队可利用度...\n")
        
        # 创建评估环境
        eval_env = gym.make(env.unwrapped.spec.id, render_mode=None)
        
        # 评估团队可利用度
        rewards, exploitability = evaluate(
            env, 
            nfsp_agents, 
            num_episodes=eval_episodes, 
            calculate_exploitability=True,
            eval_env=eval_env
        )
        
        print(f"\n团队平均奖励: {np.mean(rewards):.4f}")
        print(f"各智能体奖励: {rewards}")
        print(f"团队可利用度: {exploitability:.4f}")
        print("\n注: 可利用度越低表示策略越接近最优协作策略\n")
        
        # 保存评估结果
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        np.savez(
            f"{results_dir}/exploitability_eval.npz",
            rewards=rewards,
            exploitability=exploitability
        )
        
        print(f"评估结果已保存到 {results_dir}/exploitability_eval.npz")
        
        # 保存结果
        results = {
            'rewards': rewards,
            'mean_reward': np.mean(rewards),
            'exploitability': exploitability
        }
    
    # 如果需要渲染演示回合
    if render and num_demo_episodes > 0:
        # 展示演示回合
        print(f"\n展示{num_demo_episodes}个回合的表现:\n")
        demo_rewards = []
        coop_metrics = []
        
        for episode in range(num_demo_episodes):
            print(f"回合 {episode + 1}...")
            
            # 为每个回合创建新的环境实例
            test_env = gym.make(env.unwrapped.spec.id, render_mode="human")
            
            # 为新环境设置智能体控制器
            for i, agent in enumerate(nfsp_agents):
                if i < len(test_env.players):
                    test_env.players[i].set_controller(agent)
            
            # 运行一个回合并渲染
            _, rewards, info = test_env.run(None, is_training=False, render=True, sleep_time=0.5)
            demo_rewards.append(rewards)
            
            # 收集合作指标
            if info:
                coop_metrics.append(info)
                
                # 打印回合合作指标
                print(f"回合 {episode + 1} 奖励: {rewards}")
                if 'coop_food_count' in info:
                    print(f"需要合作食物: {info['coop_food_count']:.1f}, 玩家聚集组数: {info['player_clusters']:.1f}")
                if 'avg_cluster_size' in info and info['avg_cluster_size'] > 0:
                    print(f"平均组大小: {info['avg_cluster_size']:.2f}")
            else:
                print(f"回合 {episode + 1} 奖励: {rewards}")
            
            # 关闭环境
            test_env.close()
            time.sleep(1.0)  # 在回合之间暂停
        
        print("\n演示完成！\n")
        
        # 将演示回合的奖励和合作指标添加到结果中
        results['demo_rewards'] = demo_rewards
        results['demo_coop_metrics'] = coop_metrics
    
    return results