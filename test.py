import numpy as np
import time
import os
import gymnasium as gym
import logging
from evaluate import evaluate

logger = logging.getLogger(__name__)

def test_agents(env, agents, eval_episodes=100, eval_explo=False, render=True, num_demo_episodes=5):
    """
    测试智能体与队友合作性能
    
    参数:
        env: 游戏环境
        agents: 智能体列表，可以包含NFSP, PPO和SimpleAgent2
        eval_episodes: 评估回合数
        eval_explo: 是否评估团队可利用度
        render: 是否渲染演示回合
        num_demo_episodes: 演示回合数量
        
    返回:
        测试结果字典，包含平均奖励、智能体奖励和可利用度（如果计算）
    """
    # 区分不同类型的智能体
    nfsp_agents = [agent for agent in agents if agent.name.startswith('NFSP')]
    ppo_agents = [agent for agent in agents if agent.name.startswith('PPO')]
    teammate_agents = [agent for agent in agents if not hasattr(agent, 'add_traj')]
    
    print("\n测试模式 - 加载预训练模型并展示性能...\n")
    print(f"正在测试 {len(nfsp_agents)} 个NFSP智能体, {len(ppo_agents)} 个PPO智能体与 {len(teammate_agents)} 个SimpleAgent2队友合作\n")
    
    # 记录模型加载状态
    loaded_models_info = []
    
    # 加载NFSP智能体模型
    if nfsp_agents:
        loaded_models = []
        for agent in nfsp_agents:
            if hasattr(agent, 'load_models') and agent.load_models():
                loaded_models.append("成功")
            else:
                loaded_models.append("失败")
        
        loaded_models_info.append(f"NFSP模型加载状态: {loaded_models}")
    
    # PPO智能体模型已在main.py中加载
    if ppo_agents:
        loaded_models_info.append("PPO模型已在主程序中加载")
    
    for info in loaded_models_info:
        print(info)
    
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
            eval_episodes=eval_episodes,
            calculate_exploitability=True,
            eval_env=eval_env
        )
        
        print(f"\n团队平均奖励: {np.mean(rewards):.4f}")
        
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
    else:
        # 如果不评估可利用度，仍需要评估性能
        print("\n评估团队性能...\n")
        
        # 创建评估环境
        eval_env = gym.make(env.unwrapped.spec.id, render_mode=None)
        
        # 评估团队奖励
        rewards = evaluate(
            env, 
            agents, 
            eval_episodes=eval_episodes, 
            calculate_exploitability=False,
            eval_env=eval_env
        )
        
        print(f"\n团队平均奖励: {rewards.sum():.4f}")
        # 保存评估结果
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        np.savez(
            f"{results_dir}/eval_results.npz",
            rewards=rewards
        )
        
        print(f"评估结果已保存到 {results_dir}/eval_results.npz")
        
        # 保存结果
        results = {
            'rewards': rewards
        }
    
    # 如果需要渲染演示回合
    if render and num_demo_episodes > 0:
        # 展示演示回合
        print(f"\n展示{num_demo_episodes}个回合的表现:\n")
        demo_rewards = []
        coop_metrics = []
        
        for episode in range(num_demo_episodes):
            print(f"回合 {episode + 1}...")
            
            # 检查是否有PPO智能体
            has_ppo = len(ppo_agents) > 0
            
            if has_ppo:
                # PPO智能体需要手动运行环境
                test_env = gym.make(env.unwrapped.spec.id, render_mode="human")
                obs, _ = test_env.reset()
                
                done = False
                truncated = False
                episode_rewards = np.zeros(len(agents))
                
                while not (done or truncated):
                    actions = []
                    # 对每个智能体获取动作
                    for i, agent in enumerate(agents):
                        if agent in ppo_agents:
                            # 使用PPO的select_action方法
                            action, _ = agent.select_action(obs[i])
                        else:
                            # 使用其他智能体的step方法
                            action = agent.step(obs[i])
                        
                        actions.append(action)
                    
                    # 执行动作
                    obs, rewards, done_list, truncated_list, info = test_env.step(actions)
                    
                    # 更新奖励
                    episode_rewards += np.array(rewards)
                    
                    # 更新done和truncated标志
                    if isinstance(done_list, (list, tuple)):
                        done = all(done_list)
                    else:
                        done = done_list
                        
                    if isinstance(truncated_list, (list, tuple)):
                        truncated = all(truncated_list)
                    else:
                        truncated = truncated_list
                    
                    # 渲染
                    test_env.render()
                    time.sleep(0.5)  # 降低速度以便观察
                
                rewards = episode_rewards
                demo_rewards.append(rewards)
                
                # 打印回合奖励
                print(f"回合 {episode + 1} 奖励: {rewards}")
                
                # 关闭环境
                test_env.close()
                
            else:
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
        if coop_metrics:
            results['demo_coop_metrics'] = coop_metrics
    
    return results 