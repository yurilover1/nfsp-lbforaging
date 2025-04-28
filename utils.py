import numpy as np
import time
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import logging
from tqdm import tqdm
import gym
import numpy as np
from lbforaging.foraging.environment_3d import ForagingEnv3D


# 添加从nfsp_run.py移动的函数
logger = logging.getLogger(__name__)


def calculate_state_size(env_name, env_config=None):
    # 如果提供了环境配置，使用它创建环境
    if env_config:
        if 'Foraging3D-v0' in env_name:
            env = ForagingEnv3D(**env_config)
        else:
            env = gym.make(env_name, **env_config)
    else:
        env = gym.make(env_name)

    try:
        # 获取观测空间
        if 'Foraging3D-v0' in env_name:
            # 对于3D环境
            obs = env.reset()[0]
            if isinstance(obs, tuple):
                # 如果返回的是元组，尝试获取第一个智能体的观测
                obs = obs[0] if len(obs) > 0 else obs
        else:
            # 对于原始环境
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]  # gym >= 0.26.0 返回 (obs, info)
            else:
                obs = reset_result  # 旧版本直接返回 obs

        # 计算状态空间大小
        if hasattr(obs, 'shape'):
            # 对于numpy数组观测
            state_size = np.prod(obs.shape)
        elif isinstance(obs, tuple) and len(obs) == 2:
            # 对于元组观测 (player_obs, food_obs)
            state_size = np.prod(obs[0].shape) + np.prod(obs[1].shape)
        else:
            # 对于其他情况，尝试迭代观测
            try:
                state_size = sum(np.prod(o.shape) for o in obs)
            except:
                # 如果以上都失败，使用观测空间的形状
                state_size = np.prod(env.observation_space.shape)

        return int(state_size)

    except Exception as e:
        print(f"计算状态大小时出错: {e}")
        # 如果出错，返回一个默认值
        return 1372  # 这是您之前获得的状态空间大小

    finally:
        env.close()


def calculate_state_size(env):
    """计算环境的状态大小"""
    try:
        # 检查是否使用3D环境
        is_3d_env = "ForagingEnv3D" in str(type(env.unwrapped))

        # 获取一个示例观测
        if hasattr(env, 'reset'):
            obs, info = env.reset()
            
            # 如果环境使用grid_observation，从get_agent_obs方法获取更准确的观测
            if is_3d_env and hasattr(env, 'grid_observation') and env.grid_observation:
                agent_obs = env.get_agent_obs()
                if agent_obs and isinstance(agent_obs, list) and len(agent_obs) > 0:
                    obs = agent_obs[0]  # 使用第一个智能体的观测
        else:
            obs, info = None, None

        # 如果无法获取观测，使用默认策略
        if obs is None:
            if is_3d_env:
                # 检查环境尺寸
                n_rows = getattr(env, 'n_rows', 4)
                n_cols = getattr(env, 'n_cols', 4)
                n_depth = getattr(env, 'n_depth', 4)
                # 3D环境：假设是网格观测(4, n_rows, n_cols, n_depth)，4个通道
                print(f"使用3D环境默认状态大小 (4,{n_rows},{n_cols},{n_depth})")
                return 4 * n_rows * n_cols * n_depth  # 3D网格观测的默认大小
            else:
                # 2D环境默认状态大小
                print("使用2D环境默认状态大小")
                return 100

        # 如果观测是元组（多智能体环境），取第一个智能体的观测
        if isinstance(obs, tuple):
            first_obs = obs[0]
        else:
            first_obs = obs

        # 检查观测形状以判断观测模式
        if isinstance(first_obs, np.ndarray):
            # 检查是否为3D环境的网格观测
            if is_3d_env and len(first_obs.shape) == 4:
                # 例如 [4, rows, cols, depth] 形状
                print(f"检测到3D网格观测模式: 形状={first_obs.shape}")
                return first_obs.size  # 返回展平后的大小

            # 检查是否为三维数组
            elif len(first_obs.shape) == 3:
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
            if isinstance(env.observation_space, gym.spaces.Tuple):
                obs_shape = env.observation_space[0].shape
            else:
                obs_shape = env.observation_space.shape

            print(f"从observation_space获取的形状: {obs_shape}")

            # 如果是多维形状
            if isinstance(obs_shape, tuple) and len(obs_shape) > 1:
                # 计算总大小
                return np.prod(obs_shape)
            elif isinstance(obs_shape, tuple) and len(obs_shape) > 0:
                return obs_shape[0]

        # 默认状态大小
        if is_3d_env:
            print("无法确定3D环境的状态大小，使用默认值(864)")
            return 864  # 4*6*6*6(默认3D网格观测大小)
        else:
            print("无法确定状态大小，使用默认值100")
            return 100

    except Exception as e:
        print(f"计算状态大小时出错: {e}")
        import traceback
        traceback.print_exc()
        # 检查是否为3D环境
        is_3d_env = "ForagingEnv3D" in str(type(env.unwrapped))
        if is_3d_env:
            # 3D环境默认状态大小
            return 864  # 4*6*6*6(默认3D网格观测大小)
        else:
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
        # 检查是否使用3D环境
        is_3d_env = "ForagingEnv3D" in str(type(env))

        if is_3d_env:
            # 为3D环境创建新的评估环境实例
            from lbforaging.foraging.environment_3d import ForagingEnv3D
            eval_env = ForagingEnv3D(
                n_rows=env.n_rows,
                n_cols=env.n_cols,
                n_depth=env.n_depth,
                num_agents=env.num_agents,
                num_food=env.num_food,
                max_player_level=env.max_player_level,
                min_player_level=env.min_player_level,
                max_food_level=env.max_food_level,
                min_food_level=env.min_food_level,
                sight=10,  # 确保全局视野 - 设置为比环境尺寸更大的值
                force_coop=env.force_coop,
                grid_observation=hasattr(env, 'grid_observation') and env.grid_observation,
                food_reward_scale=env.food_reward_scale,
                step_reward_factor=env.step_reward_factor,
                step_reward_threshold=env.step_reward_threshold
            )
        else:
            # 创建一个新的标准环境用于评估
            try:
                # 尝试使用gym.make
                import gymnasium as gym
                eval_env = gym.make(env.unwrapped.spec.id, render_mode=None)
            except (AttributeError, ImportError) as e:
                print(f"无法使用gym.make创建环境：{e}")
                # 作为后备选项，尝试复制环境
                import copy
                eval_env = copy.deepcopy(env)

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
            exploitability, _ = agents[0].evaluate_team_exploitability(eval_env, agents, num_episodes=num_episodes // 2)
            return avg_rewards, exploitability

        return avg_rewards
    else:
        # 环境不支持run方法，创建一个简单的评估循环
        print("警告：环境不支持run方法，使用手动评估循环")
        total_rewards = np.zeros(len(agents))

        for _ in range(num_episodes):
            # 重置环境
            if hasattr(eval_env, 'reset'):
                obs, _ = eval_env.reset()
            else:
                print("错误：环境没有reset方法")
                return np.zeros(len(agents))

            done = False
            episode_rewards = np.zeros(len(agents))

            while not done:
                # 收集动作
                actions = []
                for i, agent in enumerate(agents):
                    if hasattr(agent, 'step'):
                        action = agent.step(obs[i] if isinstance(obs, tuple) else obs)
                    else:
                        print(f"错误：智能体{i}没有step方法")
                        action = 0  # 使用默认动作
                    actions.append(action)

                # 执行动作
                if hasattr(eval_env, 'step'):
                    obs, rewards, done, _, _ = eval_env.step(actions)
                    episode_rewards += rewards
                else:
                    print("错误：环境没有step方法")
                    break

            total_rewards += episode_rewards

        avg_rewards = total_rewards / num_episodes

        # 计算可利用度（如果需要）
        if calculate_exploitability and hasattr(agents[0], 'evaluate_team_exploitability'):
            exploitability, _ = agents[0].evaluate_team_exploitability(eval_env, agents, num_episodes=num_episodes // 2)
            return avg_rewards, exploitability

        return avg_rewards




def train_agents(env, nfsp_agents, num_episodes=5000, eval_interval=100, render=False, render_interval=100, render_mode='human'):
    """训练NFSP智能体"""
    # 检查是否使用3D环境
    is_3d_env = "ForagingEnv3D" in str(type(env))

    # 用于评估的环境
    if is_3d_env:
        # 为3D环境创建新的评估环境实例
        from lbforaging.foraging.environment_3d import ForagingEnv3D
        eval_env = ForagingEnv3D(
            n_rows=env.n_rows,
            n_cols=env.n_cols,
            n_depth=env.n_depth,
            num_agents=env.num_agents,
            num_food=env.num_food,
            max_player_level=env.max_player_level,
            min_player_level=env.min_player_level,
            max_food_level=env.max_food_level,
            min_food_level=env.min_food_level,
            sight=10,  # 确保全局视野 - 设置为比环境尺寸更大的值
            force_coop=env.force_coop,
            grid_observation=hasattr(env, 'grid_observation') and env.grid_observation,
            food_reward_scale=env.food_reward_scale,
            step_reward_factor=env.step_reward_factor,
            step_reward_threshold=env.step_reward_threshold
        )
    else:
        try:
            # 尝试使用gym.make
            import gymnasium as gym
            eval_env = gym.make("Foraging-6x6-2p-3f-v3", sight=3,
                                grid_observation=False, three_layer_obs=True, render_mode=None)
        except Exception as e:
            print(f"无法创建评估环境：{e}")
            eval_env = env  # 使用原始环境作为后备选项
    
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
                        sleep_time=0.5,
                        render_mode=render_mode  # 添加渲染模式参数
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
                for j in range(env.num_agents):
                    for ts in trajectories[j]:
                        if len(ts) > 0:
                            # ts结构：[obs_dict, action, reward, next_obs_dict, done]
                            # 直接将整个观察字典传递给_preprocess_state方法，不需要提前预处理
                            obs_dict = ts[0]
                            action = ts[1]
                            reward = ts[2]
                            next_obs_dict = ts[3]
                            done = ts[4]
                            
                            # 直接添加轨迹，让_preprocess_state在内部处理数据格式转换
                            nfsp_agents[j].add_traj([obs_dict, action, reward, next_obs_dict, done])
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
    
    # 保存为numpy格式
    try:
        np.savez('./results/training_history.npz', **data_to_save)
        print("训练历史保存成功: ./results/training_history.npz")
    except Exception as e:
        print(f"保存训练历史时出错: {e}")
        import traceback
        traceback.print_exc()


def test_agents(env, nfsp_agents, eval_episodes=100, eval_explo=False, render=True, num_demo_episodes=5, render_mode='human', sleep_time=0.5):
    """
    测试NFSP智能体性能
    
    参数:
        env: 环境
        nfsp_agents: NFSP智能体列表
        eval_episodes: 评估回合数
        eval_explo: 是否评估可利用性
        render: 是否渲染
        num_demo_episodes: 渲染的回合数
        render_mode: 渲染模式
        sleep_time: 渲染时每步之间的延迟时间(秒)
        
    返回:
        测试结果字典
    """
    print(f"\n开始测试智能体性能 (总共 {eval_episodes} 回合)...")
    
    # 为评估设置智能体策略模式
    for agent in nfsp_agents:
        agent.policy_mode = agent.eval_mode  # 使用评估模式('average' 或 'best')
    
    # 测试部分
    
    # 评估智能体性能
    if render and num_demo_episodes > 0:
        print(f"\n运行 {num_demo_episodes} 个渲染回合...")
        
        # 渲染演示回合
        total_rewards = np.zeros(env.num_agents)  # 使用num_agents而不是n_agents
        
        for i in range(num_demo_episodes):
            print(f"\n渲染回合 {i+1}/{num_demo_episodes}...")
            # 执行演示回合，并渲染
            _, episode_rewards = env.run(
                nfsp_agents, 
                is_training=False, 
                render=True, 
                sleep_time=sleep_time,
                render_mode=render_mode
            )
            print(f"回合 {i+1} 奖励: {episode_rewards}")
            total_rewards += episode_rewards
            
        # 计算平均奖励
        avg_demo_rewards = total_rewards / num_demo_episodes
        print(f"\n演示回合平均奖励: {avg_demo_rewards}")
    
    # 更多评估（统计全部评估回合）
    remaining_episodes = eval_episodes - (num_demo_episodes if render else 0)
    
    if remaining_episodes > 0:
        print(f"\n运行 {remaining_episodes} 个评估回合...")
        
        # 不渲染的评估回合
        total_rewards = np.zeros(env.num_agents)
        
        for i in range(remaining_episodes):
            # 执行评估回合，不渲染
            _, episode_rewards = env.run(
                nfsp_agents, 
                is_training=False, 
                render=False
            )
            total_rewards += episode_rewards
            
        # 计算平均奖励
        avg_eval_rewards = total_rewards / remaining_episodes
        print(f"\n评估回合平均奖励: {avg_eval_rewards}")
    
    # 合并所有回合的奖励统计
    if num_demo_episodes > 0 and render and remaining_episodes > 0:
        # 如果同时有演示和评估回合
        total_episodes = num_demo_episodes + remaining_episodes
        combined_rewards = (avg_demo_rewards * num_demo_episodes + 
                          avg_eval_rewards * remaining_episodes) / total_episodes
        agent_rewards = combined_rewards
    elif num_demo_episodes > 0 and render:
        # 只有演示回合
        agent_rewards = avg_demo_rewards
    else:
        # 只有评估回合
        agent_rewards = avg_eval_rewards
    
    team_reward = np.sum(agent_rewards)
    print(f"\n整体评估结果:")
    print(f"团队总奖励: {team_reward:.4f}")
    print(f"每个智能体奖励: {agent_rewards}")
    
    # 如果需要评估可利用性
    exploitability = 0.0
    if eval_explo and hasattr(nfsp_agents[0], 'evaluate_team_exploitability'):
        try:
            exploitability, _ = nfsp_agents[0].evaluate_team_exploitability(env, nfsp_agents, num_episodes=50)
            print(f"团队可利用性: {exploitability:.4f}")
        except Exception as e:
            print(f"评估可利用性时出错: {e}")
    
    # 返回测试结果
    return {
        'mean_reward': team_reward,
        'agent_rewards': agent_rewards,
        'exploitability': exploitability if eval_explo else None
    }