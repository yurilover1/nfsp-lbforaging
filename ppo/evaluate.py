import os
import gymnasium as gym
import torch
import lbforaging
from ppo.agent import PPOAgent
import argparse
from tqdm import tqdm

def evaluate(args):
    """评估函数"""
    # 创建环境
    env = gym.make(
        "Foraging-5x5-2p-2f-v3",
        render_mode="human" if args.render else None,
        # force_coop=True,
        max_episode_steps=100
    )
    
    # 创建智能体
    agents = [
        PPOAgent(
            input_dim=12,
            hidden_dims=[128, 128],
            output_dim=6,
            device=args.device
        ) for _ in range(2)
    ]
    
    # 加载模型
    for i, agent in enumerate(agents):
        checkpoint_path = os.path.join(args.checkpoint_dir, f"agent_{i}", f"checkpoint_episode_{args.episode}.pt")
        agent.load_model(checkpoint_path)
    
    # 评估循环
    total_rewards = [0, 0]
    
    for episode in tqdm(range(args.num_episodes)):
        obs, _ = env.reset()
        episode_rewards = [0, 0]
        done = False
        truncated = False
        
        while not (done or truncated):
            # 每个智能体选择动作
            actions = []
            for i, agent in enumerate(agents):
                action, _ = agent.select_action(obs[i])
                actions.append(action)
            
            # 执行动作
            obs, rewards, done, truncated, _ = env.step(actions)
            
            # 更新奖励
            for i in range(2):
                episode_rewards[i] += rewards[i]
        
        # 累积总奖励
        for i in range(2):
            total_rewards[i] += episode_rewards[i]
        
        # 打印每个回合的结果
        print(f"回合 {episode + 1}/{args.num_episodes}:")
        print(f"智能体0奖励: {episode_rewards[0]:.2f}")
        print(f"智能体1奖励: {episode_rewards[1]:.2f}")
        print(f"总奖励: {sum(episode_rewards):.2f}")
        print("-" * 50)
    
    # 计算平均奖励
    avg_rewards = [r / args.num_episodes for r in total_rewards]
    
    # 打印最终结果
    print("\n评估结果:")
    print(f"智能体0平均奖励: {avg_rewards[0]:.2f}")
    print(f"智能体1平均奖励: {avg_rewards[1]:.2f}")
    print(f"总平均奖励: {sum(avg_rewards):.2f}")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="检查点目录路径")
    parser.add_argument("--episode", type=int, required=True, help="要加载的检查点回合数")
    parser.add_argument("--num_episodes", type=int, default=100, help="评估回合数")
    args = parser.parse_args()
    
    evaluate(args) 