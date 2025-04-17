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
        'exploitability': []  # 新增可利用度记录
    }
    
    # 创建结果目录
    os.makedirs("./results", exist_ok=True)
    
    for episode in range(num_episodes):
        # 在每个回合开始时选择策略模式
        for agent in nfsp_agents:
            agent.choose_policy_mode()

        # 判断是否需要在本回合渲染
        should_render = render and episode % render_interval == 0
        
        if should_render:
            print(f"渲染回合 {episode}...")
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
        print(f"回合 {episode} 奖励: {payoffs}")
        history['episode_rewards'].append(payoffs)
        
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
        
        # 定期评估
        if episode % eval_interval == 0 or episode == num_episodes - 1:
            try:
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
                
                # 打印进度
                print(f"Episode {episode}/{num_episodes}: 训练奖励={payoffs}, 评估奖励={agent0_reward}, 可利用度={exploitability:.4f}")
            except Exception as e:
                print(f"评估过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
    
    # 训练结束，保存模型
    for agent in nfsp_agents:
        agent.save_models()
        agent.plot_losses()
    
    # 绘制训练曲线
    plot_training_curve(history, num_episodes, eval_interval, nfsp_agents)
    
    # 保存训练历史记录
    save_history(history, nfsp_agents)
    
    return nfsp_agents

def plot_training_curve(history, num_episodes, eval_interval, nfsp_agents=None):
    """绘制训练曲线"""
    plt.figure(figsize=(16, 16))
    
    # 1. 训练奖励 (左上角)
    plt.subplot(2, 2, 1)
    raw_rewards = history['episode_rewards']
    
    # 打印训练奖励数据用于调试
    print(f"原始训练奖励数据: {raw_rewards}")
    
    if len(raw_rewards) > 0:
        # 检查数据格式并转换为numpy数组
        if isinstance(raw_rewards[0], (list, np.ndarray)):
            # 多智能体情况，每个元素是包含多个奖励的列表或数组
            rewards = np.array([reward for reward in raw_rewards])
            n_agents = rewards.shape[1]
            
            # 直接绘制每个智能体的奖励
            for i in range(n_agents):
                plt.plot(np.arange(len(rewards)), rewards[:, i], 'o-', label=f'智能体 {i+1}')
        else:
            # 单智能体情况，每个元素是单个奖励值
            rewards = np.array(raw_rewards)
            plt.plot(np.arange(len(rewards)), rewards, 'o-', label='智能体')
    else:
        print("训练奖励数据为空，无法绘制")
    
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.title('训练奖励')
    plt.grid(True)
    plt.legend()
    
    # 2. 评估奖励 (右上角)
    plt.subplot(2, 2, 2)
    eval_rewards = history['eval_rewards']
    eval_episodes = history['eval_episodes']
    
    # 打印调试信息
    print(f"评估数据: episodes={eval_episodes}, rewards={eval_rewards}")
    print(f"维度: x_shape={np.array(eval_episodes).shape}, y_shape={np.array(eval_rewards).shape}")
    
    # 确保x轴和y轴长度一致
    if len(eval_episodes) > 0 and len(eval_rewards) > 0:
        # 使用实际记录的回合数作为x轴数据
        plt.plot(eval_episodes, eval_rewards, 'ro-', label='评估奖励')
        
        # 添加趋势线
        if len(eval_episodes) > 1:
            z = np.polyfit(eval_episodes, eval_rewards, 1)
            p = np.poly1d(z)
            plt.plot(eval_episodes, p(eval_episodes), "b--", label='趋势线')
            
        plt.xlabel('回合')
        plt.ylabel('评估奖励')
        plt.title('与随机智能体对战的评估奖励')
        plt.grid(True)
        plt.legend()
    else:
        print("评估数据为空，跳过绘制评估奖励曲线")
    
    # 3. 可利用度 (左下角)
    plt.subplot(2, 2, 3)
    exploitability = history.get('exploitability', [])
    
    if len(eval_episodes) > 0 and len(exploitability) > 0:
        plt.plot(eval_episodes, exploitability, 'go-', label='可利用度')
        
        # 添加趋势线
        if len(eval_episodes) > 1:
            z = np.polyfit(eval_episodes, exploitability, 1)
            p = np.poly1d(z)
            plt.plot(eval_episodes, p(eval_episodes), "m--", label='趋势线')
            
        plt.xlabel('回合')
        plt.ylabel('可利用度')
        plt.title('团队协作可利用度')
        plt.grid(True)
        plt.legend()
    else:
        print("可利用度数据为空，跳过绘制可利用度曲线")
    
    # 4. RL损失和SL损失 (右下角)
    if nfsp_agents and len(nfsp_agents) > 0:
        plt.subplot(2, 2, 4)
        agent = nfsp_agents[0]  # 使用第一个智能体的损失
        
        # 创建两个Y轴，左侧显示RL损失，右侧显示SL损失
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        if len(agent.RLlosses) > 0:
            # 绘制RL损失曲线
            rl_losses = np.array(agent.RLlosses)
            # 如果损失数据过多，进行下采样
            if len(rl_losses) > 1000:
                step = len(rl_losses) // 1000
                rl_losses = rl_losses[::step]
            
            # 平滑处理
            window_size = min(20, len(rl_losses))
            if window_size > 1:
                smooth_rl_losses = np.convolve(rl_losses, np.ones(window_size)/window_size, mode='valid')
                ax1.plot(smooth_rl_losses, 'b-', label='平滑RL损失')
            ax1.plot(rl_losses, 'b-', alpha=0.3, label='原始RL损失')
            ax1.set_xlabel('训练步数')
            ax1.set_ylabel('RL损失', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
        
        if len(agent.losses) > 0:
            # 绘制SL损失曲线
            sl_losses = np.array(agent.losses)
            # 如果损失数据过多，进行下采样
            if len(sl_losses) > 1000:
                step = len(sl_losses) // 1000
                sl_losses = sl_losses[::step]
            
            # 平滑处理
            window_size = min(20, len(sl_losses))
            if window_size > 1:
                smooth_sl_losses = np.convolve(sl_losses, np.ones(window_size)/window_size, mode='valid')
                ax2.plot(smooth_sl_losses, 'r-', label='平滑SL损失')
            ax2.plot(sl_losses, 'r-', alpha=0.3, label='原始SL损失')
            ax2.set_ylabel('SL损失', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('RL和SL损失曲线')
        
        # 合并两个轴的图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.grid(True)
    
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
    if 'exploitability' in history:
        data_to_save['exploitability'] = history['exploitability']
    
    # 添加智能体损失数据
    if nfsp_agents and len(nfsp_agents) > 0:
        agent = nfsp_agents[0]  # 使用第一个智能体的数据
        data_to_save['rl_losses'] = agent.RLlosses
        data_to_save['sl_losses'] = agent.losses
        if hasattr(agent, 'policy_accuracies'):
            data_to_save['policy_accuracies'] = agent.policy_accuracies
    
    # 保存为numpy格式
    try:
        np.savez('./results/training_history.npz', **data_to_save)
        print("训练历史保存成功: ./results/training_history.npz")
    except Exception as e:
        print(f"保存训练历史时出错: {e}")
        import traceback
        traceback.print_exc()

def test_agents(env, nfsp_agents, eval_episodes=100, evaluate_exploitability=False, render=True, num_demo_episodes=5):
    """
    测试NFSP智能体性能
    
    参数:
        env: 游戏环境
        nfsp_agents: 要测试的NFSP智能体列表
        eval_episodes: 评估回合数
        evaluate_exploitability: 是否评估团队可利用度
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
    if evaluate_exploitability:
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