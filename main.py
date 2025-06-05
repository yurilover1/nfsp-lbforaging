import os
import argparse
import gymnasium as gym
import logging
# import lbforaging  # noqa
from utils import *
from train import train_agents
from test import test_agents
import torch


logger = logging.getLogger(__name__)


def main(args):
    """主函数，训练并测试智能体"""
    # 配置日志记录
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # 设置环境和智能体
    render_mode = "human" if args.render else None
    env = gym.make("Foraging-5x5-2p-2f-v3", force_coop=args.force_coop, normalize_reward=False,
                   render_mode=render_mode, max_episode_steps=args.max_epi_steps)
    action_size = 6  # (NONE, NORTH, SOUTH, WEST, EAST, LOAD)

    args.agent_player = env.players[0]
    
    # 创建智能体
    agents = []
    # 创建主智能体（可选NFSP或PPO）
    agents.append(create_agent(args.agent_type, env.players,
        calculate_state_size(env), action_size, args.layers, args.device))
    agents.append(teammate_generate(action_size, id=0, device=args.device))
    
    # 训练模式
    if not args.test:
        print(f"\n开始训练智能体（类型: {args.agent_type}）...\n")
        # 使用train_agents函数进行训练
        _ = train_agents(
            env, 
            agents, 
            num_episodes=args.episodes, 
            eval_interval=args.eval_interval,
            render=args.render,
            render_interval=args.render_interval,
            teamate_id=0
        )
        print("\n训练完成！\n")
    # 测试模式
    else:
        agent0 = agents[0]
        # 加载预训练模型
        if args.agent_type.lower() == "nfsp":
            # 对于NFSP智能体，使用其专有加载方法
            if hasattr(agent0, 'load_models'):
                agent0.load_models("./models")
        elif args.agent_type.lower() == "ppo":
            # 对于PPO智能体，加载最近的检查点
            checkpoint_path = f"models/ppo_agent_0_model.pt"
            if checkpoint_path:
                agent0.load_model(checkpoint_path)
            else:
                print("警告: 未指定检查点路径，使用未训练的PPO智能体")
        # 使用test模块中的test_agents函数进行测试
        test_results = test_agents(
            env,
            agents,
            eval_episodes=args.eval_episodes,
            # eval_explo=args.eval_explo,
            render=args.render,
            num_demo_episodes=100
        )

        # 打印测试结果摘要
        if 'mean_reward' in test_results:
            print(f"\n测试结果摘要:")
            print(f"平均团队奖励: {test_results['mean_reward']:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练和测试智能体")
    
    # 基本选项
    parser.add_argument("--episodes", type=int, default=2000, help="训练回合数")
    parser.add_argument("--render", action="store_true", help="启用环境渲染")
    parser.add_argument("--render_interval", type=int, default=100, help="训练期间渲染的回合间隔")
    parser.add_argument("--eval_interval", type=int, default=1, help="评估间隔（批次数，每批次100回合）")
    parser.add_argument("--test", action="store_true", help="测试模式：加载预训练模型")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志")
    # parser.add_argument("--eval_explo", action="store_true", help="评估团队可利用度")
    parser.add_argument("--eval_episodes", type=int, default=100, help="评估回合数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备类型")
    parser.add_argument("--layers", type=int, default=7, help="神经网络层数")
    parser.add_argument("--max_epi_steps", type=int, default=30, help="每个回合的最大步数")
    parser.add_argument("--force_coop", action="store_true", help="强制合作模式")
    parser.add_argument("--agent_type", type=str, default="nfsp", choices=["nfsp", "ppo"], help="主智能体类型")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="PPO智能体的检查点路径（仅测试模式）")
    
    args = parser.parse_args()
    main(args)

