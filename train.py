#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import logging
import time
from datetime import datetime

# 导入环境和智能体
import lbforaging
from lbforaging.foraging.environment_3d import ForagingEnv3D
from lbforaging.agents import NFSPAgent
from utils import calculate_state_size, train_agents, save_history, plot_training_curve

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练3D环境中的NFSP智能体")
    parser.add_argument("--episodes", type=int, default=2000, help="训练回合数")
    parser.add_argument("--eval_interval", type=int, default=100, help="评估间隔（回合数）")
    parser.add_argument("--render", action="store_true", help="启用渲染（默认关闭）")
    parser.add_argument("--render_interval", type=int, default=500, help="渲染间隔（回合数）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_dir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--device", type=str, default="cpu:0", help="训练设备")
    parser.add_argument("--proximity_reward", action="store_true", help="启用接近奖励（默认关闭）")
    parser.add_argument("--step_reward", action="store_true", help="启用步进奖励（默认关闭）")
    parser.add_argument("--proximity_factor", type=float, default=0.05, help="接近奖励因子")
    args = parser.parse_args()

    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

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
        proximity_factor=args.proximity_factor,
        enable_proximity_reward=args.proximity_reward,
        enable_step_reward=args.step_reward
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
        agent_dir = os.path.join(args.save_dir, f"agent{i}")
        os.makedirs(agent_dir, exist_ok=True)

        agent = NFSPAgent(
            player=env.players[i],
            state_size=state_size,
            action_size=action_size,
            epsilon_init=0.95,
            epsilon_decay=1000,
            epsilon_min=0.2,
            update_freq=20,
            sl_lr=0.01,
            rl_lr=0.01,
            sl_buffer_size=5000,
            rl_buffer_size=10000,
            rl_start=50,
            sl_start=50,
            train_freq=1,
            gamma=0.99,
            eta=0.2,
            rl_batch_size=32,
            sl_batch_size=64,
            hidden_units=1024,
            layers=8,
            device=args.device
        )
        agents.append(agent)

    # 开始训练
    start_time = time.time()
    logger.info("开始训练...")

    # 执行训练过程
    trained_agents = train_agents(
        env,
        agents,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        render=args.render,
        render_interval=args.render_interval,
        render_mode='3d'
    )

    # 训练结束，打印耗时
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"训练完成！总耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")

    # 保存训练历史和模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, agent in enumerate(trained_agents):
        agent_dir = os.path.join(args.save_dir, f"agent{i}")
        agent.save_models(path=agent_dir)
        logger.info(f"已保存智能体{i}模型到 {agent_dir}")

    # 绘制训练曲线
    plot_training_curve(
        history=save_history(history={}, nfsp_agents=trained_agents),
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        nfsp_agents=trained_agents
    )
    logger.info("已绘制训练曲线")

if __name__ == "__main__":
    main() 