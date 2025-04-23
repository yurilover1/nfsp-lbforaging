import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import logging
import lbforaging  # noqa
from lbforaging.agents  import RandomAgent

# 添加从nfsp_run.py移动的函数
logger = logging.getLogger(__name__)

def calculate_state_size(env):
    """计算状态空间大小"""
    # 重置环境获取观察
    obss, _ = env.reset()
    
    # 从观察空间获取信息
    obs_shape = env.observation_space[0].shape[0]  # 观察空间的总维度
    
    # 通过Gymnasium的公开接口获取场地大小和智能体数量
    field_size = env.field_size[0] * env.field_size[1]  # 场地大小 (rows * cols)
    n_agents = env.n_agents  # 智能体数量
    
    # 计算最终状态大小
    # 场地大小 + 每个智能体4个特征 (x, y, level, is_self)
    state_size = field_size + (4 * n_agents)
    
    print(f"状态空间大小: field_size={field_size}, n_agents={n_agents}, total={state_size}")
    print(f"观察空间形状: {obs_shape}")
    
    # 使用观察空间大小作为状态大小，确保维度匹配
    return obs_shape

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
    eval_env = gym.make("Foraging-5x5-2p-1f-v3", render_mode=None)
    
    # 训练历史记录
    history = {
        'episode_rewards': [],
        'eval_rewards': [],
        'eval_episodes': [],  # 记录每次评估对应的回合数
        'exploitability': [],  # 可利用度记录
        'sl_losses': [],       # 监督学习损失
        'rl_losses': [],       # 强化学习损失
        'policy_accuracies': [] # 策略准确率
    }
    
    # 创建结果目录
    os.makedirs("./results", exist_ok=True)
    
    # 用于记录100回合的奖励
    recent_rewards = []
    
    # 设置初始批次
    current_batch = 0
    total_batches = num_episodes // 100
    
    # 记录最后显示的状态文本长度
    last_status_length = 0
    
    print(f"\n开始训练 - 总共 {num_episodes} 回合 ({total_batches} 批次)...\n")
    
    for episode in range(num_episodes):
        # 在每个回合开始时选择策略模式
        for agent in nfsp_agents:
            agent.choose_policy_mode()

        # 判断是否需要在本回合渲染
        should_render = render and episode % render_interval == 0
        
        if should_render:
            # 清除当前行
            if last_status_length > 0:
                print(" " * last_status_length, end="\r", flush=True)
            print(f"\n渲染回合 {episode}...")
            # 为渲染创建专门的环境实例
            render_env = gym.make("Foraging-5x5-2p-1f-v3", render_mode="human")
            
            # 设置智能体控制器
            for i, agent in enumerate(nfsp_agents):
                if i < len(render_env.players):
                    render_env.players[i].set_controller(agent)
            
            # 运行一个完整回合，并渲染
            trajectories, payoffs = render_env.run(
                None,  # 已经设置了控制器，所以传None
                is_training=True,
                render=True, 
                sleep_time=0.5
            )
            
            # 关闭渲染环境
            render_env.close()
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
        
        # 计算当前100轮内的进度（0-99）
        current_progress = episode % 100
        progress_percent = current_progress + 1  # 进度从1%到100%
        
        # 创建动态进度条
        bar_length = 20  # 减少进度条长度，避免行太长
        block = int(bar_length * progress_percent / 100)
        progress_bar = '█' * block + '░' * (bar_length - block)
        
        # 计算总体完成进度
        overall_progress = (episode + 1) / num_episodes * 100
        
        # 使用\r在同一行更新进度，固定状态文本的宽度以确保完全覆盖前一行
        status_text = f"批次: {current_batch+1}/{total_batches} | 进度: [{progress_bar}] {progress_percent:3d}%| 奖励: {team_reward:.2f}"
        
        # 先清除前一行，再打印新状态
        clear_line = " " * max(last_status_length, len(status_text))
        print(clear_line, end="\r", flush=True)
        print(status_text, end="\r", flush=True)
        
        # 更新状态行长度
        last_status_length = len(status_text)
        
        # 每100轮显示平均奖励并重置进度条
        if (episode + 1) % 100 == 0:
            # 计算最近100回合的平均奖励
            avg_reward = np.mean(recent_rewards)
            
            # 清除当前行并打印批次完成信息
            print(" " * last_status_length, end="\r", flush=True)
            print(f"\n✓ 完成批次 {current_batch+1}/{total_batches} | 平均团队奖励: {avg_reward:.4f} | 总进度: {overall_progress:.1f}%\n")
            
            # 更新批次计数
            current_batch += 1
            
            # 重置recent_rewards列表
            recent_rewards = []
            
            # 重置状态行长度（因为下一行会从头开始）
            last_status_length = 0
        
        # 存储轨迹并训练
        for i in range(env.n_agents):
            for ts in trajectories[i]:
                if len(ts) > 0:
                    # ts结构：[obs_dict, action, reward, next_obs_dict, done]
                    # 首先获取预处理后的状态
                    obs = nfsp_agents[i]._preprocess_state(ts[0])
                    action = ts[1]
                    reward = ts[2]
                    next_obs = None if ts[4] else nfsp_agents[i]._preprocess_state(ts[3])
                    done = ts[4]
                    
                    # 如果是终止状态，则使用当前状态作为下一状态
                    if next_obs is None:
                        next_obs = obs
                    
                    # 直接添加轨迹，_preprocess_state中会处理数据格式转换
                    nfsp_agents[i].add_traj([obs, action, reward, next_obs, done])
                    nfsp_agents[i].train()
        
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
                # 清除当前行，确保评估信息从新行开始
                if last_status_length > 0:
                    print(" " * last_status_length, end="\r", flush=True)
                
                # 重置状态行长度
                last_status_length = 0
                
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
                print(f"\n评估 - 回合 {episode}/{num_episodes}: 奖励={eval_rewards}, 可利用度={exploitability:.4f}\n")
            except Exception as e:
                print(f"\n评估过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
    
    # 训练结束，清空最后一行
    if last_status_length > 0:
        print(" " * last_status_length, end="\r", flush=True)
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
    
    # 保存为numpy格式
    try:
        np.savez('./results/training_history.npz', **data_to_save)
        print("训练历史保存成功: ./results/training_history.npz")
    except Exception as e:
        print(f"保存训练历史时出错: {e}")
        import traceback
        traceback.print_exc()

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
        
        for episode in range(num_demo_episodes):
            print(f"回合 {episode + 1}...")
            
            # 为每个回合创建新的环境实例
            test_env = gym.make(env.unwrapped.spec.id, render_mode="human")
            
            # 为新环境设置智能体控制器
            for i, agent in enumerate(nfsp_agents):
                if i < len(test_env.players):
                    test_env.players[i].set_controller(agent)
            
            # 运行一个回合并渲染
            _, rewards = test_env.run(None, is_training=False, render=True, sleep_time=0.5)
            demo_rewards.append(rewards)
            print(f"回合 {episode + 1} 奖励: {rewards}")
            
            # 关闭环境
            test_env.close()
            time.sleep(1.0)  # 在回合之间暂停
        
        print("\n演示完成！\n")
        
        # 将演示回合的奖励添加到结果中
        results['demo_rewards'] = demo_rewards
    
    return results