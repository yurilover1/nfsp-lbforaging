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
    

def train_agents(env, agents, num_episodes=5000, eval_interval=100, render=False, render_interval=100):
    """训练NFSP智能体与预加载的SimpleAgent2队友合作
    
    参数:
        env: 游戏环境
        agents: 智能体列表，第一个为NFSP，第二个为SimpleAgent2
        num_episodes: 训练回合数
        eval_interval: 评估间隔
        render: 是否渲染
        render_interval: 渲染间隔
    """
    # 判断智能体类型
    nfsp_agents = [agent for agent in agents if hasattr(agent, 'add_traj')]
    teammate_agents = [agent for agent in agents if not hasattr(agent, 'add_traj')]
    
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
    print(f"正在训练 {len(nfsp_agents)} 个NFSP智能体与 {len(teammate_agents)} 个SimpleAgent2队友合作\n")
    
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
                    trajectories, payoffs, steps = env.run(
                        agents,
                        is_training=True,
                        render=True, 
                        sleep_time=0.5
                    )
                    # 暂停进度条更新
                    pbar.clear()

                else:
                    # 正常训练，不渲染
                    trajectories, payoffs, steps = env.run(
                        agents, 
                        is_training=True,
                        render=False
                    )
                
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
                
                # 存储轨迹并训练（仅对NFSP智能体）
                for j, agent in enumerate(nfsp_agents):
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
    
    print("\n训练完成！\n")
    
    # 保存模型（仅NFSP智能体）
    for agent in nfsp_agents:
        agent.save_models()
    
    # 绘制训练曲线
    plot_training_curve(history, num_episodes, eval_interval, nfsp_agents)
    
    # 保存训练历史记录
    save_history(history, nfsp_agents)
    
    return agents

def plot_training_curve(history, num_episodes, eval_interval, nfsp_agents=None):
    """绘制训练曲线，包括四幅图：监督学习损失、策略准确率、强化学习损失和队伍总奖励"""
    # 数据预处理部分 - 将所有数据处理逻辑放在绘图之前
    
    # 预处理变量初始化
    smooth_losses = None
    smooth_acc = None
    smooth_rl = None
    team_rewards = None
    smooth_rewards = None
    
    # 只使用第一个NFSP智能体数据
    if nfsp_agents and len(nfsp_agents) > 0:
        agent = nfsp_agents[0]
        
        # 处理监督学习损失
        if len(agent.losses) > 0:
            losses = np.array(agent.losses)
            
            # 降采样
            if len(losses) > 1000:
                step = len(losses) // 1000
                losses = losses[::step]
            
            # 平滑处理
            window_size = min(20, len(losses))
            if window_size > 1:
                smooth_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        
        # 处理策略准确率
        if hasattr(agent, 'policy_accuracies') and len(agent.policy_accuracies) > 0:
            accuracies = np.array(agent.policy_accuracies)
            
            # 降采样
            if len(accuracies) > 1000:
                step = len(accuracies) // 1000
                accuracies = accuracies[::step]
            
            # 平滑处理
            window_size = min(20, len(accuracies))
            if window_size > 1:
                smooth_acc = np.convolve(accuracies, np.ones(window_size)/window_size, mode='valid')
        
        # 处理强化学习损失
        if len(agent.RLlosses) > 0:
            rl_losses = np.array(agent.RLlosses)
            
            # 降采样
            if len(rl_losses) > 1000:
                step = len(rl_losses) // 1000
                rl_losses = rl_losses[::step]
            
            # 平滑处理
            window_size = min(20, len(rl_losses))
            if window_size > 1:
                smooth_rl = np.convolve(rl_losses, np.ones(window_size)/window_size, mode='valid')
    
    # 处理团队奖励数据
    raw_rewards = history['episode_rewards']
    if len(raw_rewards) > 0:
        # 检查数据格式并转换为numpy数组
        if isinstance(raw_rewards[0], (list, np.ndarray)):
            # 多智能体情况，计算团队总奖励
            rewards = np.array([reward for reward in raw_rewards])
            team_rewards = rewards.sum(axis=1)
            
            # 平滑处理团队奖励
            window_size = min(50, len(team_rewards) // 10)  # 使用更大的窗口来平滑奖励曲线
            if window_size > 1:
                smooth_rewards = np.convolve(team_rewards, np.ones(window_size)/window_size, mode='valid')
        else:
            # 单智能体情况
            team_rewards = np.array(raw_rewards)
    
    # 创建图形
    plt.figure(figsize=(16, 16))
    
    # 1. 监督学习损失 (左上角)
    plt.subplot(2, 2, 1)
    if smooth_losses is not None:
        plt.plot(smooth_losses, 'k-', linewidth=2, label='SL Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Supervised Learning Loss')
    plt.grid(True)
    plt.legend()
    
    # 2. 策略准确率 (右上角)
    plt.subplot(2, 2, 2)
    if smooth_acc is not None:
        plt.plot(smooth_acc, 'k-', linewidth=2, label='Policy Accuracy')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Policy Accuracy')
    plt.grid(True)
    plt.legend()
    
    # 3. 强化学习损失 (左下角)
    plt.subplot(2, 2, 3)
    if smooth_rl is not None:
        plt.plot(smooth_rl, 'k-', linewidth=2, label='RL Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Reinforcement Learning Loss')
    plt.grid(True)
    plt.legend()
    
    # 4. 队伍总奖励 (右下角)
    plt.subplot(2, 2, 4)
    if team_rewards is not None:
        window_size = min(50, len(team_rewards) // 10)
        # 画出原始数据（较浅的颜色）
        plt.plot(np.arange(len(team_rewards)), team_rewards, 'b-', alpha=0.3, label='Raw Team Reward')
        
        # 如果有平滑数据，也画出平滑后的数据
        if smooth_rewards is not None:
            plt.plot(np.arange(len(smooth_rewards)) + window_size//2, smooth_rewards, 'b-', 
                     linewidth=2, label='Smoothed Team Reward')
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Team Total Reward')
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
    
    # 保存为numpy格式
    try:
        np.savez('./results/training_history.npz', **data_to_save)
        print("训练历史保存成功: ./results/training_history.npz")
    except Exception as e:
        print(f"保存训练历史时出错: {e}")
        import traceback
        traceback.print_exc()

def test_agents(env, agents, eval_episodes=100, eval_explo=False, render=True, num_demo_episodes=5):
    """
    测试NFSP智能体与SimpleAgent2队友合作性能
    
    参数:
        env: 游戏环境
        agents: 智能体列表，可以包含NFSP和SimpleAgent2
        eval_episodes: 评估回合数
        eval_explo: 是否评估团队可利用度
        render: 是否渲染演示回合
        num_demo_episodes: 演示回合数量
        
    返回:
        测试结果字典，包含平均奖励、智能体奖励和可利用度（如果计算）
    """
    # 区分NFSP智能体和SimpleAgent2智能体
    nfsp_agents = [agent for agent in agents if hasattr(agent, 'add_traj')]
    teammate_agents = [agent for agent in agents if not hasattr(agent, 'add_traj')]
    
    print("\n测试模式 - 加载预训练模型并展示性能...\n")
    print(f"正在测试 {len(nfsp_agents)} 个NFSP智能体与 {len(teammate_agents)} 个SimpleAgent2队友合作\n")
    
    # 加载预训练模型(只加载NFSP智能体模型)
    loaded_models = []
    for agent in nfsp_agents:
        if agent.load_models():
            loaded_models.append("成功")
        else:
            loaded_models.append("失败")
    
    print(f"NFSP模型加载状态: {loaded_models}")
    
    results = {}
    
    # 如果指定了评估可利用度
    if eval_explo:
        print("\n评估团队可利用度...\n")
        
        # 创建评估环境
        eval_env = gym.make(env.unwrapped.spec.id, render_mode=None)
        
        # 评估团队可利用度
        rewards, exploitability = evaluate(
            env, 
            agents, 
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
            for i, agent in enumerate(agents):
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