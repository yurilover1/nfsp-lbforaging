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
    env = gym.make("Foraging-5x5-2p-1f-v3", 
                   render_mode=render_mode, sight=2, 
                   grid_observation=True, force_coop=False)
    
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
            epsilon_init=0.3,                # 增加初始探索率以获取更多样本
            epsilon_decay=20000,             # 降低衰减速度，保持更长时间的探索
            epsilon_min=0.05,                # 保持合理的最小探索率
            update_freq=100,                 # 增加目标网络更新频率
            sl_lr=0.001,                     # 降低监督学习率以稳定训练
            rl_lr=0.0005,                    # 降低强化学习率以稳定训练
            sl_buffer_size=2000,            # 大幅增加监督学习缓冲区大小
            rl_buffer_size=5000,            # 大幅增加强化学习缓冲区大小
            rl_start=100,                   # 增加RL训练起始大小
            sl_start=100,                   # 增加SL训练起始大小
            train_freq=1,                    # 每步都训练
            gamma=0.99,                      # 维持折扣因子
            eta=0.1,                         # 降低平均策略比例，更注重最优策略
            rl_batch_size=256,               # 增加批量大小提高稳定性
            sl_batch_size=512,               # 增加批量大小提高稳定性
            hidden_units=256,                # 增加隐藏单元大小匹配修改后的模型
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
    
    args = parser.parse_args()
    main(args)

