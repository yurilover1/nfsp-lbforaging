import argparse
from test import test_agents

from agents.nfsp_agent import NFSPAgent
from train import train_agents
from utils import *

logger = logging.getLogger(__name__)


def main(args):
    use_visualize = getattr(args, 'use_visualize', False)  # 新增：可视化开关，默认True
    print('rollout_train.py running')
    import lbforaging.foraging.environment
    print('ENV FILE:', lbforaging.foraging.environment.__file__)
    """主函数，训练并测试智能体"""
    # 配置日志记录
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # 设置环境和智能体
    render_mode = "human" if args.render else None
    from lbforaging.foraging.environment import ForagingEnv
    env = ForagingEnv(
        num_agents=2,
        min_player_level=1,
        max_player_level=2,
        min_food_level=2,
        max_food_level=2,
        field_size=(5, 5),
        max_num_food=2,
        sight=5,
        max_episode_steps=30,  # 修改为50步
        force_coop=False,
        observe_agent_levels=True,
        render_mode=render_mode,
        step_reward_factor=0.4,
        attraction_reward_factor=0.2,
        decay_rate=0.01
    )
    action_size = 6  # (NONE, NORTH, SOUTH, WEST, EAST, LOAD)

    env.reset()
    print("玩家位置：", env.agent_positions)
    print("玩家等级：", env.agent_levels)
    print("食物位置和等级：", env.field.food_positions)
    args.agent_player = env.players[0]

    # 创建主智能体（NFSP）
    main_agent = NFSPAgent(
        # 基本参数
        player=args.agent_player,
        state_size=calculate_state_size(env),
        action_size=action_size,
        device=args.device,
        # 网络结构参数
        hidden_units=args.hidden_units,
        layers=args.layers,
        # 训练参数
        gamma=args.gamma,
        eta=args.eta,  # 策略选择概率
        # 学习率
        rl_lr=args.rl_lr,  # PPO学习率
        sl_lr=args.sl_lr,  # 监督学习率
        # 缓冲区参数
        sl_buffer_size=30000,
        # 评估模式
        eval_mode='average',
        # 新增参数
        entropy_coef=args.entropy_coef,
        batch_size=args.batch_size
    )
    teammate_pool = [teammate_generate(6, device=args.device, id=i) for i in range(8)]
    # 训练模式
    if not args.test:
        print(f"\n开始训练NFSP智能体...\n")
        train_agents(env, main_agent, num_episodes=args.episodes, eval_interval=args.eval_interval, render=args.render, render_interval=args.render_interval, layer_num=args.layers, train_freq=50, batch_size=args.batch_size)
        print("\n训练完成！\n")
    # 测试模式
    else:
        print(f"\n开始测试NFSP智能体...\n")
        for i in range(args.eval_episodes):
            agents = [main_agent]
            from train import run_episode
            _, team_reward, steps, _ = run_episode(env, agents, render=args.render)
            print(f"Test Episode {i+1}: reward={team_reward}, steps={steps}")
        print("\n测试完成！\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练和测试NFSP智能体")
    
    # 基本选项
    parser.add_argument("--episodes", type=int, default=100000, help="训练回合数")
    parser.add_argument("--render", action="store_true", help="启用环境渲染")
    parser.add_argument("--render_interval", type=int, default=5, help="训练期间渲染的回合间隔")
    parser.add_argument("--eval_interval", type=int, default=1, help="评估间隔（批次数，每批次100回合）")
    parser.add_argument("--test", action="store_true", help="测试模式：加载预训练模型")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志")
    parser.add_argument("--eval_episodes", type=int, default=50, help="评估回合数")
    parser.add_argument("--device", type=str, default="cpu", help="设备类型")
    
    # NFSP智能体参数
    parser.add_argument("--layers", type=int, default=4, help="神经网络层数")
    parser.add_argument("--hidden_units", type=int, default=512, help="隐藏层单元数")
    parser.add_argument("--eta", type=float, default=0.8, help="策略选择概率（1.0=纯PPO，0.1=NFSP）")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--rl_lr", type=float, default=0.0002, help="PPO学习率")
    parser.add_argument("--sl_lr", type=float, default=0.0005, help="监督学习率")
    parser.add_argument("--entropy_coef", type=float, default=0.25, help="PPO熵系数")
    parser.add_argument("--batch_size", type=int, default=256, help="PPO batch size")

    args = parser.parse_args()
    main(args)
