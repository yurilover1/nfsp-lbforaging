import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import logging
import lbforaging  # noqa
from lbforaging.agents import NFSPAgent, RandomAgent
from utils import calculate_state_size, evaluate, train_agents, plot_training_curve, save_history, test_agents


logger = logging.getLogger(__name__)

def main(args):
    """主函数，训练并测试NFSP智能体"""
    # 配置日志记录
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # 设置环境和智能体
    render_mode = "human" if args.render else None
    env = gym.make("Foraging-5x5-2p-1f-v3", render_mode=render_mode)
    
    # 计算状态空间大小
    state_size = calculate_state_size(env)
    action_size = 6  # (NONE, NORTH, SOUTH, WEST, EAST, LOAD)
    
    # 创建NFSP智能体
    nfsp_agents = []
    
    for i in range(env.n_agents):
        # 为每个智能体设置目录
        agent_dir = f"./models/agent{i}"
        os.makedirs(agent_dir, exist_ok=True)
        
        # 为每个智能体设置独特的种子
        seed = args.seed + i if args.seed is not None else None
        
        # 创建NFSP智能体
        agent = NFSPAgent(
            player=env.players[i],
            state_size=state_size,
            action_size=action_size,
            epsilon_init=0.6,                # 提高初始探索率
            epsilon_decay=10000,             # 减缓探索衰减速度
            epsilon_min=0.1,                 # 提高最小探索率
            update_freq=200,                 # 减少目标网络更新频率
            sl_lr=0.005,                     # 提高监督学习率
            rl_lr=0.005,                     # 提高强化学习率
            sl_buffer_size=10000,            # 减小监督学习缓冲区
            rl_buffer_size=10000,            # 减小强化学习缓冲区
            rl_start=100,                    # 减小RL缓冲区起始大小
            sl_start=100,                    # 减小SL缓冲区起始大小
            train_freq=1,                    # 每步都进行训练
            gamma=0.99,                      # 折扣因子
            eta=0.2,                         # 增加最优策略使用概率
        )
        nfsp_agents.append(agent)
    
    # 训练模式
    if not args.test:
        print("\n开始训练智能体...\n")
        trained_agents = train_agents(
            env, 
            nfsp_agents, 
            num_episodes=args.episodes, 
            eval_interval=args.eval_interval,
            render=args.render,
            render_interval=args.render_interval
        )
        print("\n训练完成！\n")
    # 测试模式
    else:
        # 使用utils中的test_agents函数进行测试
        test_results = test_agents(
            env,
            nfsp_agents,
            eval_episodes=args.eval_episodes,
            evaluate_exploitability=args.evaluate_exploitability,
            render=args.render,
            num_demo_episodes=5
        )
        
        # 打印测试结果摘要
        if 'mean_reward' in test_results:
            print(f"\n测试结果摘要:")
            print(f"平均团队奖励: {test_results['mean_reward']:.4f}")
            if 'exploitability' in test_results:
                print(f"团队可利用度: {test_results['exploitability']:.4f}")
            print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练和测试NFSP智能体")
    
    # 基本选项
    parser.add_argument("--episodes", type=int, default=5000, help="训练回合数")
    parser.add_argument("--render", action="store_true", help="启用环境渲染")
    parser.add_argument("--render_interval", type=int, default=100, help="训练期间渲染的回合间隔")
    parser.add_argument("--eval_interval", type=int, default=100, help="评估间隔（回合数）")
    parser.add_argument("--test", action="store_true", help="测试模式：加载预训练模型")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志")
    parser.add_argument("--evaluate_exploitability", action="store_true", help="评估团队可利用度")
    parser.add_argument("--eval_episodes", type=int, default=100, help="评估回合数")
    
    args = parser.parse_args()
    main(args)

