import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import logging
import lbforaging  # noqa
from lbforaging.agents import NFSPAgent
from utils import calculate_state_size, evaluate, train_agents, plot_training_curve, save_history, test_agents


logger = logging.getLogger(__name__)

def main(args):
    """主函数，训练并测试NFSP智能体"""
    # 配置日志记录
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # 设置环境和智能体
    render_mode = "human" if args.render else None
    env = gym.make("Foraging-6x6-2p-3f-v3", 
                   render_mode=render_mode, sight=3, 
                   three_layer_obs=True, force_coop=False)
    
    # 计算状态空间大小
    state_size = calculate_state_size(env)
    print(f"计算得到的状态空间大小: {state_size}")
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
            epsilon_init=0.4,                # 进一步增加初始探索率
            epsilon_decay=30000,             # 进一步延长探索期
            epsilon_min=0.05,                # 保持合理的最小探索率
            update_freq=200,                 # 减少目标网络更新频率，使训练更稳定
            sl_lr=0.0005,                    # 进一步降低监督学习率
            rl_lr=0.0002,                    # 进一步降低强化学习率
            sl_buffer_size=10000,            # 进一步扩大监督学习缓冲区
            rl_buffer_size=20000,            # 进一步扩大强化学习缓冲区
            rl_start=500,                    # 增加RL开始训练的样本数量
            sl_start=500,                    # 增加SL开始训练的样本数量
            train_freq=4,                    # 减少训练频率，允许更多探索
            gamma=0.99,                      # 维持折扣因子
            eta=0.2,                         # 增加平均策略比例，增强稳定性
            rl_batch_size=128,               # 减少批量大小，防止过拟合
            sl_batch_size=256,               # 减少批量大小，防止过拟合
            hidden_units=256,                # 减小隐藏单元，简化模型
            layers=args.layers,
            device=args.device
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
            eval_explo=args.eval_explo,
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
    parser.add_argument("--eval_explo", action="store_true", help="评估团队可利用度")
    parser.add_argument("--eval_episodes", type=int, default=100, help="评估回合数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备类型")
    parser.add_argument("--layers", type=int, default=5, help="神经网络层数")
    args = parser.parse_args()
    main(args)

