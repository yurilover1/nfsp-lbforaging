#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D LB Foraging环境测试与可视化程序
"""

import os
import sys
import time
import numpy as np
import argparse
from lbforaging.foraging.environment_3d import ForagingEnv3D

def main(args):
    """主函数：创建并运行3D环境"""
    # 创建环境
    env = ForagingEnv3D(
        n_rows=args.rows,
        n_cols=args.cols,
        n_depth=args.depth,
        num_agents=args.agents,
        num_food=args.food,
        sight=args.sight,
        max_player_level=args.max_player_level,
        max_food_level=args.max_food_level,
        min_player_level=args.min_player_level,
        min_food_level=args.min_food_level,
        force_coop=args.force_coop,
        grid_observation=True,
        max_episode_steps=args.max_steps
    )
    
    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        env.seed(args.seed)
    
    # 颜色代码
    class Colors:
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    def print_colorful_slice(env, z, reward=None, action=None):
        """打印单个Z层的彩色切片"""
        print(f"{Colors.BOLD}{Colors.UNDERLINE}Z层 #{z}{Colors.ENDC}")
        
        # 打印列标题
        col_header = '   '
        for col in range(env.n_cols):
            col_header += f" {col:2} "
        print(col_header)
        
        # 打印分隔线
        h_line = '  +'
        for _ in range(env.n_cols):
            h_line += '---+'
        print(h_line)
        
        # 打印每一行
        for row in range(env.n_rows):
            row_str = f"{row:2}|"
            for col in range(env.n_cols):
                cell = ' '
                
                # 标记食物
                for food in env.food_items:
                    if not food.collected and food.position == (row, col, z):
                        cell = f"{Colors.GREEN}F{food.level}{Colors.ENDC}"
                
                # 标记玩家
                for i, player in enumerate(env.players):
                    if player.position == (row, col, z):
                        if cell != ' ':
                            cell += f"{Colors.RED}P{i}{Colors.ENDC}"
                        else:
                            cell = f"{Colors.BLUE}P{i}{Colors.ENDC}"
                
                row_str += f" {cell:3}|"
            
            print(row_str)
            print(h_line)
    
    def print_3d_grid_state(env, rewards=None, actions=None):
        """打印整个3D网格的所有切片"""
        print("\n" + "="*50)
        print(f"{Colors.BOLD}环境状态 | 步数: {env._current_steps}/{env._max_episode_steps}{Colors.ENDC}")
        
        # 打印玩家状态
        print(f"\n玩家状态:")
        for i, player in enumerate(env.players):
            print(f"  玩家 #{i}: 位置={player.position}, 等级={player.level}, 分数={player.score:.2f}")
            if rewards is not None:
                print(f"     奖励: {Colors.YELLOW}{rewards[i]:.3f}{Colors.ENDC}")
            if actions is not None:
                action_name = env.action_set.get(actions[i], "未知")
                print(f"     动作: {Colors.CYAN}{action_name}{Colors.ENDC}")
        
        # 打印食物状态
        print(f"\n食物状态:")
        remaining_food = 0
        for i, food in enumerate(env.food_items):
            status = "可获取" if not food.collected else "已收集"
            status_color = Colors.GREEN if not food.collected else Colors.RED
            print(f"  食物 #{i}: 位置={food.position}, 等级={food.level}, 状态={status_color}{status}{Colors.ENDC}")
            if not food.collected:
                remaining_food += 1
        
        # 打印每个Z层的切片
        print(f"\n网格表示 (剩余食物: {remaining_food}/{len(env.food_items)}):")
        for z in range(env.n_depth):
            print_colorful_slice(env, z, rewards, actions)
        
        print("="*50 + "\n")
    
    # 重置环境
    obs = env.reset()

    # 打印初始状态
    print(f"\n{Colors.BOLD}{Colors.PURPLE}初始状态:{Colors.ENDC}")
    print_3d_grid_state(env)
    
    # 循环执行随机动作
    total_rewards = np.zeros(env.num_agents)
    print(f"\n{Colors.BOLD}开始执行随机动作:{Colors.ENDC}")
    
    step_count = args.steps  # 使用命令行参数指定的步数
    
    for step in range(step_count):
        print(f"\n{Colors.BOLD}{Colors.PURPLE}步骤 #{step+1}/{step_count}{Colors.ENDC}")
        
        # 为每个智能体选择随机动作
        if args.manual:
            # 手动输入动作
            print(f"可用动作: {env.action_set}")
            actions = []
            for i in range(env.num_agents):
                while True:
                    try:
                        action = int(input(f"为智能体 #{i} 输入动作 (0-6): "))
                        if 0 <= action <= 6:
                            break
                        print("无效动作，请重新输入")
                    except ValueError:
                        print("请输入0-6之间的整数")
                actions.append(action)
        else:
            # 随机选择动作
            actions = [np.random.randint(0, 7) for _ in range(env.num_agents)]
            
        action_names = [env.action_set[a] for a in actions]
        print(f"动作: {Colors.CYAN}{action_names}{Colors.ENDC}")

        # 执行环境步进
        obs, rewards, done, truncated, info = env.step(actions)
        total_rewards += rewards

        # 打印更新后的状态
        print(f"\n{Colors.BOLD}更新后的状态:{Colors.ENDC}")
        print_3d_grid_state(env, rewards, actions)
        
        # 打印奖励和总分
        print(f"当前回合奖励: {Colors.YELLOW}{rewards}{Colors.ENDC}")
        print(f"累计奖励总分: {Colors.YELLOW}{total_rewards}{Colors.ENDC}")
        print(f"游戏是否结束: {Colors.RED if done else Colors.GREEN}{done}{Colors.ENDC}")
        
        # 等待时间
        if args.delay > 0:
            time.sleep(args.delay)
        
        # 如果游戏结束，打印结果和原因
        if done:
            print(f"\n{Colors.BOLD}{Colors.PURPLE}游戏结束!{Colors.ENDC}")
            if all(food.collected for food in env.food_items):
                print(f"{Colors.GREEN}原因: 所有食物已收集{Colors.ENDC}")
            elif env._current_steps >= env._max_episode_steps:
                print(f"{Colors.RED}原因: 达到最大步数 ({env._max_episode_steps}){Colors.ENDC}")
            break

    env.close()
    print(f"\n{Colors.BOLD}{Colors.PURPLE}运行结束!{Colors.ENDC}")
    
    # 打印最终分数
    print(f"\n{Colors.BOLD}最终分数:{Colors.ENDC}")
    for i, player in enumerate(env.players):
        print(f"  玩家 #{i}: {Colors.YELLOW}{player.score:.2f}{Colors.ENDC}")
    print(f"  团队总分: {Colors.YELLOW}{sum(player.score for player in env.players):.2f}{Colors.ENDC}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行3D Foraging环境并显示切片视图")
    
    # 环境参数
    parser.add_argument("--rows", type=int, default=3, help="环境行数")
    parser.add_argument("--cols", type=int, default=3, help="环境列数")
    parser.add_argument("--depth", type=int, default=3, help="环境深度")
    parser.add_argument("--agents", type=int, default=2, help="智能体数量")
    parser.add_argument("--food", type=int, default=1, help="食物数量")
    parser.add_argument("--sight", type=int, default=None, help="视野范围")
    parser.add_argument("--max-player-level", type=int, default=1, help="最大玩家等级")
    parser.add_argument("--min-player-level", type=int, default=1, help="最小玩家等级")
    parser.add_argument("--max-food-level", type=int, default=2, help="最大食物等级")
    parser.add_argument("--min-food-level", type=int, default=2, help="最小食物等级")
    parser.add_argument("--force-coop", action="store_true", default=True, help="强制合作模式")
    parser.add_argument("--max-steps", type=int, default=100, help="最大步数")
    
    # 运行参数
    parser.add_argument("--steps", type=int, default=100, help="运行步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--delay", type=float, default=0.2, help="每步之间的延迟时间")
    parser.add_argument("--manual", action="store_true", help="手动输入动作")
    
    args = parser.parse_args()
    main(args)