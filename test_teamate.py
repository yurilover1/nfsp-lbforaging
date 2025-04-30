import os
import gymnasium as gym
import numpy as np
import torch
import lbforaging
from partners.agent import SimpleAgent2
import logging
import itertools

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_teammate_pair(agent_id_0, agent_id_1, num_episodes=10):
    """测试一对预训练智能体的性能"""
    # 创建环境
    env = gym.make("Foraging-5x5-2p-2f-v3", render_mode="human",
                   force_coop=True, max_episode_steps=100)
    
    # 加载两个预训练的智能体
    agent0 = SimpleAgent2(
        input_dim=12,  # 输入维度为12
        hidden_dims=[128, 128],
        output_dim=6,
        device="cpu"
    )
    model_path_0 = f'./partners/agents_for_5*5/agent_{agent_id_0}_0.pt'
    agent0.load_model(model_path_0)
    
    agent1 = SimpleAgent2(
        input_dim=12,  # 输入维度为12
        hidden_dims=[128, 128],
        output_dim=6,
        device="cpu"
    )
    model_path_1 = f'./partners/agents_for_5*5/agent_{agent_id_1}_1.pt'
    agent1.load_model(model_path_1)
    
    agents = [agent0, agent1]
    print(f"已加载智能体: {model_path_0} 和 {model_path_1}")
    
    # 运行测试回合
    total_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_rewards = [0, 0]
        done = False
        truncated = False
        
        while not (done or truncated):
            # 由于无法直接访问私有属性_valid_actions，直接使用观测作为输入
            actions = []
            for i, agent in enumerate(agents):
                observation = {
                    'obs': obs[i],
                    'actions': list(range(6))  # 默认可以执行0-5的动作
                }
                actions.append(agent.step(observation))
            
            # 执行动作
            obs, rewards, done, truncated, info = env.step(actions)
            
            # 累积奖励
            episode_rewards = [r1 + r2 for r1, r2 in zip(episode_rewards, rewards)]
            
            # 如果所有智能体都完成了，结束回合
            if isinstance(done, (list, tuple)):
                done = all(done)
            if isinstance(truncated, (list, tuple)):
                truncated = all(truncated)
        
        total_rewards.append(sum(episode_rewards))
        # print(f"回合 {episode+1}/{num_episodes}: 总奖励 = {sum(episode_rewards):.2f}, 智能体奖励 = [{episode_rewards[0]:.2f}, {episode_rewards[1]:.2f}]")
    
    # 打印总结
    avg_reward = np.mean(total_rewards)
    print(f"\n测试结果:")
    print(f"智能体组合: {agent_id_0}_{0} 和 {agent_id_1}_{1}")
    print(f"平均回合总奖励: {avg_reward:.2f}")
    print(f"最低回合奖励: {min(total_rewards):.2f}")
    print(f"最高回合奖励: {max(total_rewards):.2f}")
    print("-" * 50)
    
    return avg_reward

def test_multiple_pairs():
    """测试多对预训练智能体组合"""
    # 定义要测试的智能体ID
    agent_ids = [0, 1, 2, 3]
    
    # 测试参数
    num_episodes = 1000  # 每对智能体测试的回合数
    
    # 测试结果
    results = {}
    
    # 测试不同的智能体组合
    for agent_id_0 in range(7):
        pair_key = f"{agent_id_0}_{0}_and_{agent_id_0}_{1}"
        print(f"\n开始测试智能体组合: {pair_key}")
        avg_reward = test_teammate_pair(agent_id_0, agent_id_0, num_episodes)
        results[pair_key] = avg_reward
    
    # 打印所有结果的总结
    print("\n\n所有智能体组合的测试结果汇总:")
    print("=" * 70)
    print(f"{'智能体组合':<30} | {'平均回合总奖励':<15}")
    print("-" * 70)
    
    # 按平均奖励排序
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for pair, avg_reward in sorted_results:
        print(f"{pair:<30} | {avg_reward:<15.2f}")
    
    print("=" * 70)

if __name__ == "__main__":
    # 如果只想测试一对特定的智能体，取消下面这行的注释
    # test_teammate_pair(0, 1, num_episodes=10)
    
    # 测试多对智能体组合
    test_multiple_pairs()
