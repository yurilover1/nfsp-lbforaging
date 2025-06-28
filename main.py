import argparse
import gymnasium as gym
import logging
import lbforaging
from utils import *
from train import train_agents
from test import test_agents
from agents.nfsp_agent import NFSPAgent
import torch


logger = logging.getLogger(__name__)


def main(args):
    """主函数，训练并测试智能体"""
    # 配置日志记录
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # 设置环境和智能体
    render_mode = "human" if args.render else None
    env = gym.make("Foraging-5x5-2p-2f-v3", normalize_reward=False,
                   render_mode=render_mode)
    action_size = 6  # (NONE, NORTH, SOUTH, WEST, EAST, LOAD)

    args.agent_player = env.players[0]

    # 创建主智能体（可选NFSP或PPO）
    main_agent = NFSPAgent(
            player=args.agent_player,
            state_size=calculate_state_size(env), action_size=action_size,
            epsilon_init=0.3, epsilon_decay=10000, epsilon_min=0.05,
            update_freq=1, sl_lr=0.005, rl_lr=1e-5,
            sl_buffer_size=20000, rl_buffer_size=80000,
            rl_start=1000, sl_start=1000, train_freq=100, gamma=0.98, eta=0.0,
            rl_batch_size=128, sl_batch_size=256, hidden_units=256, tau=0.01,
            layers=args.layers, device=args.device, eval_mode='best'
        )
    
    # 训练模式
    if not args.test:
        print(f"\n开始训练智能体（类型: {args.agent_type}）...\n")
        # 使用train_agents函数进行训练
        _ = train_agents(
            env, 
            main_agent, 
            num_episodes=args.episodes, 
            eval_interval=args.eval_interval,
            render=args.render,
            render_interval=args.render_interval,
            layer_num=args.layers
        )
        print("\n训练完成！\n")
    # 测试模式
    else:
        # 加载预训练模型
        pass

        # 打印测试结果摘要
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练和测试智能体")
    
    # 基本选项
    parser.add_argument("--episodes", type=int, default=2000, help="训练回合数")
    parser.add_argument("--render", action="store_true", help="启用环境渲染")
    parser.add_argument("--render_interval", type=int, default=5, help="训练期间渲染的回合间隔")
    parser.add_argument("--eval_interval", type=int, default=1, help="评估间隔（批次数，每批次100回合）")
    parser.add_argument("--test", action="store_true", help="测试模式：加载预训练模型")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志")
    # parser.add_argument("--eval_explo", action="store_true", help="评估团队可利用度")
    parser.add_argument("--eval_episodes", type=int, default=100, help="评估回合数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备类型")
    parser.add_argument("--layers", type=int, default=3, help="神经网络层数")
    parser.add_argument("--agent_type", type=str, default="nfsp", choices=["nfsp", "ppo"], help="主智能体类型")

    args = parser.parse_args()
    main(args)
