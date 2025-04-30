import numpy as np
import time
import os
import gymnasium as gym
import logging
from evaluate import evaluate

logger = logging.getLogger(__name__)

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