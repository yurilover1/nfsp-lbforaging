import os
import argparse
import gymnasium as gym
import logging
import lbforaging  # noqa
from lbforaging.agents import NFSPAgent
from ppo.agent import PPOAgent
from utils import calculate_state_size
from train import train_agents
from test import test_agents
from evaluate import evaluate
from partners.agent import SimpleAgent2
import random
import torch


logger = logging.getLogger(__name__)

def teammate_generate(teammate_num, action_size, id=random.randint(0, 7)):
    teammate_agents = []

    for i in range(teammate_num):
        teammate_agents.append(SimpleAgent2(
            input_dim=12,  # 使用正确的输入维度12而不是state_size
            hidden_dims=[128, 128],
            output_dim=action_size,
            device=args.device
        ))
        model_path = f'./partners/agents_for_5*5/agent_{id}_1.pt'
        teammate_agents[i].load_model(model_path)

    return teammate_agents

def create_agent(args, state_size, action_size):
    """创建指定类型的智能体"""
    agent_dir = f"./models/agent0"
    os.makedirs(agent_dir, exist_ok=True)
        
    if args.agent_type.lower() == "nfsp":
        # 创建NFSP智能体
        agent = NFSPAgent(
            player=args.agent_player, 
            state_size=state_size, action_size=action_size,
            epsilon_init=0.4, epsilon_decay=30000,  epsilon_min=0.05,                
            update_freq=200, sl_lr=0.0005, rl_lr=0.0002,                    
            sl_buffer_size=10000, rl_buffer_size=20000,            
            rl_start=500, sl_start=500, train_freq=2, gamma=0.99, eta=0.1,
            rl_batch_size=128, sl_batch_size=256, hidden_units=256,                
            layers=args.layers, device=args.device
        )
        return agent
    elif args.agent_type.lower() == "ppo":
        # 创建PPO智能体``
        agent = PPOAgent(
            input_dim=12,  # 输入维度为12
            hidden_dims=[128, 128],
            output_dim=action_size,
            device=args.device,
            player=args.agent_player  # 传递player信息
        )
        return agent
    else:
        raise ValueError(f"不支持的智能体类型: {args.agent_type}")

def main(args):
    """主函数，训练并测试智能体"""
    # 配置日志记录
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # 设置环境和智能体
    render_mode = "human" if args.render else None
    env = gym.make("Foraging-5x5-2p-2f-v3", force_coop=args.force_coop, normalize_reward=False,
                   render_mode=render_mode, max_episode_steps=args.max_epi_steps)
    
    # 计算状态空间大小
    state_size = calculate_state_size(env)
    print(f"计算得到的状态空间大小: {state_size}")
    action_size = 6  # (NONE, NORTH, SOUTH, WEST, EAST, LOAD)
    
    # 创建智能体
    agents_list = [[], [], [], [], [], [], [], []]
    args.agent_player = None    
    # 创建主智能体（可选NFSP或PPO）
    for i in range(8):
        agents_list[i].append(create_agent(args, state_size, action_size))
        agents_list[i].extend(teammate_generate(1, action_size, id=i))
    
    # 训练模式
    if not args.test:
        print(f"\n开始训练智能体（类型: {args.agent_type}）...\n")
        for agents in agents_list:
            # # 设置agent0为环境中的Player
            # if hasattr(env, 'players') and len(env.players) > 0:
            #     # 设置智能体控制器
            #     for i, agent in enumerate(agents):
            #         if i < len(env.players):
            #             env.players[i].set_controller(agent)
            #             # 为PPO/NFSP智能体设置正确的player信息
            #             if hasattr(agent, 'player'):
            #                 agent.player = env.players[i]

            # 使用train_agents函数进行训练
            _ = train_agents(
                env, 
                agents, 
                num_episodes=args.episodes, 
                eval_interval=args.eval_interval,
                render=args.render,
                render_interval=args.render_interval,
                teamate_id=agents_list.index(agents)
            )
        print("\n训练完成！\n")
    # 测试模式
    else:
        for agents in agents_list:
            agent0 = agents[0]
            # 加载预训练模型
            if args.agent_type.lower() == "nfsp":
                # 对于NFSP智能体，使用其专有加载方法
                if hasattr(agent0, 'load_models'):
                    agent0.load_models("./models")
            elif args.agent_type.lower() == "ppo":
                # 对于PPO智能体，加载最近的检查点
                checkpoint_path = f"models/ppo_agent_{agents_list.index(agents)}_model.pt"
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
    parser.add_argument("--max_epi_steps", type=int, default=50, help="每个回合的最大步数")
    parser.add_argument("--force_coop", action="store_true", help="强制合作模式")
    parser.add_argument("--agent_type", type=str, default="nfsp", choices=["nfsp", "ppo"], help="主智能体类型")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="PPO智能体的检查点路径（仅测试模式）")
    
    args = parser.parse_args()
    main(args)

