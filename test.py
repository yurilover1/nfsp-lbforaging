#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import logging

# 导入环境和智能体
import lbforaging
from lbforaging.foraging.environment_3d import ForagingEnv3D
from lbforaging.agents import NFSPAgent
from utils import calculate_state_size, test_agents

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试3D环境中的NFSP智能体")
    parser.add_argument("--episodes", type=int, default=10, help="测试回合数")
    parser.add_argument("--models_dir", type=str, default="./models", help="模型加载目录")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--sleep", type=float, default=0.2, help="每步之间的延迟时间")
    parser.add_argument("--device", type=str, default="cpu:0", help="训练设备")
    args = parser.parse_args()

    # 创建3D环境
    env = ForagingEnv3D(
        n_rows=4,
        n_cols=4,
        n_depth=4,
        num_agents=2,
        num_food=5,
        max_player_level=3,
        min_player_level=2,
        max_food_level=1,
        min_food_level=1,
        sight=4,
        force_coop=False,
        grid_observation=True,
        penalty=0.0,
        step_reward_factor=0.2,
        step_reward_threshold=0.005,
        food_reward_scale=1.0,
        proximity_factor=0.05  # 与训练环境保持一致
    )

    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        env.seed(args.seed)

    # 计算状态空间大小
    state_size = calculate_state_size(env)
    logger.info(f"状态空间大小: {state_size}")
    action_size = 7  # 3D环境有7个动作

    # 创建智能体
    agents = []
    for i in range(env.num_agents):
        agent_dir = os.path.join(args.models_dir, f"agent{i}")
        
        agent = NFSPAgent(
            player=env.players[i],
            state_size=state_size,
            action_size=action_size,
            epsilon_init=0.05,  # 测试时使用较低的探索率
            epsilon_decay=1000,
            epsilon_min=0.05,
            update_freq=20,
            sl_lr=0.01,
            rl_lr=0.01,
            sl_buffer_size=5000,
            rl_buffer_size=10000,
            rl_start=50,
            sl_start=50,
            train_freq=1,
            gamma=0.99,
            eta=0.05,  # 测试时倾向于使用最佳策略
            rl_batch_size=32,
            sl_batch_size=64,
            hidden_units=1024,
            layers=8,
            device=args.device,
            eval_mode='best'  # 使用最佳策略进行评估
        )
        
        # 尝试加载模型
        try:
            agent.load_models(path=agent_dir)
            logger.info(f"已加载智能体{i}的模型从 {agent_dir}")
        except Exception as e:
            logger.warning(f"无法加载智能体{i}的模型: {e}")
            logger.info(f"将使用未训练的智能体{i}")
            
        agents.append(agent)

    # 开始测试
    logger.info("开始测试...")
    
    # 执行测试过程
    test_results = test_agents(
        env=env,
        nfsp_agents=agents,
        eval_episodes=args.episodes,
        eval_explo=False,  # 不评估可利用性
        render=True,  # 始终渲染
        num_demo_episodes=args.episodes,
        render_mode='3d',
        sleep_time=args.sleep
    )
    
    # 打印测试结果
    if 'mean_reward' in test_results:
        logger.info(f"测试结果:")
        logger.info(f"平均团队奖励: {test_results['mean_reward']:.4f}")
        if 'exploitability' in test_results:
            logger.info(f"团队可利用度: {test_results['exploitability']:.4f}")
        
        # 打印每个智能体的奖励
        if 'agent_rewards' in test_results:
            for i, reward in enumerate(test_results['agent_rewards']):
                logger.info(f"智能体{i}平均奖励: {reward:.4f}")
    
    logger.info("测试完成！")

if __name__ == "__main__":
    main() 