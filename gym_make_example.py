#!/usr/bin/env python3
"""
使用gym.make()创建和运行LBForaging环境的示例

这个示例展示了如何：
1. 使用gym.make()创建预定义环境
2. 基本的环境交互
3. 访问Field类功能
"""

import gymnasium as gym
import numpy as np
import lbforaging

def main():
    print("=== LBForaging gym.make() 使用示例 ===\n")
    
    # 使用gym.make()创建环境
    print("1. 创建环境:")
    env = gym.make("Foraging-8x8-2p-3f-coop-v3")
    print(f"✓ 环境创建成功: {type(env).__name__}")
    print(f"  智能体数量: {env.num_agents}")
    print(f"  Field尺寸: {env.field.field_size}")
    print(f"  最大食物数量: {env.max_num_food}")
    print(f"  强制协作: {env.force_coop}")
    
    # 重置环境
    print("\n2. 重置环境:")
    observations, info = env.reset(seed=42)
    print(f"✓ 环境重置完成")
    print(f"  观测空间形状: {[obs.shape for obs in observations]}")
    print(f"  动作空间: {env.action_space}")
    
    # 显示初始状态
    print("\n3. 初始环境状态:")
    print(f"智能体位置: {env.agent_positions}")
    print(f"智能体等级: {env.agent_levels}")
    print(f"食物信息: {env.field.get_all_food_infos()}")
    print(f"总食物等级: {env.field.get_total_food_level()}")
    
    print("\nField可视化:")
    print(env.field)
    
    # 运行几步
    print("\n4. 运行环境:")
    for step in range(5):
        # 随机选择动作
        actions = []
        for i in range(env.num_agents):
            # 使用所有可能的动作 (0-5)
            action = np.random.randint(0, 6)
            actions.append(action)
        
        print(f"\n步骤 {step+1}:")
        print(f"  动作: {actions}")
        
        # 执行动作
        next_observations, reward, done, truncated, info = env.step(actions)
        
        print(f"  智能体位置: {env.agent_positions}")
        print(f"  奖励: {reward}")
        print(f"  剩余食物等级: {env.field.get_total_food_level()}")
        print(f"  游戏结束: {done}")
        
        # 显示当前状态
        if step % 2 == 0:  # 每两步显示一次
            print("  Field状态:")
            print("  " + "\n  ".join(str(env.field).split('\n')))
        
        if done:
            print("  🎉 游戏完成!")
            break
        
        observations = next_observations
    
    # Field类功能演示
    print("\n5. Field类功能演示:")
    field = env.field
    
    # 查询功能
    print("查询功能:")
    for agent_id in range(env.num_agents):
        pos = field.get_agent_position(agent_id)
        if pos:
            print(f"  智能体{agent_id} 位置: {pos}")
            # 获取相邻位置
            adjacent = field.get_adjacent_positions(*pos)
            print(f"    相邻位置: {adjacent}")
            
            # 检查相邻食物
            for adj_row, adj_col in adjacent:
                if field.is_food_at(adj_row, adj_col):
                    food_level = field.get_field_point(adj_row, adj_col).get_level()
                    print(f"    相邻食物: ({adj_row}, {adj_col}) 等级{food_level}")
    
    # 区域查询
    center = (4, 4)
    agents_in_area = field.get_agents_in_area(*center, radius=3)
    print(f"中心{center}半径3范围内的智能体: {agents_in_area}")
    
    # 转换为整数数组
    int_array = field.to_int_array()
    print(f"整数数组表示:\n{int_array}")
    
    # 关闭环境
    env.close()
    print("\n✓ 环境已关闭")

def show_available_envs():
    """展示一些可用的环境ID"""
    print("\n=== 可用的环境ID示例 ===")
    
    example_envs = [
        ("Foraging-6x6-2p-2f-v3", "6x6环境，2智能体，2食物"),
        ("Foraging-8x8-3p-4f-coop-v3", "8x8环境，3智能体，4食物，强制协作"),
        ("Foraging-2s-10x10-2p-3f-v3", "10x10环境，视野2，2智能体，3食物"),
        ("Foraging-grid-5x5-2p-2f-coop-v3", "5x5网格环境，协作模式"),
    ]
    
    for env_id, description in example_envs:
        try:
            env = gym.make(env_id)
            print(f"✓ {env_id}")
            print(f"  {description}")
            print(f"  智能体: {env.num_agents}, Field: {env.field.field_size}, 视野: {env.sight}")
            env.close()
        except Exception as e:
            print(f"✗ {env_id}: {e}")
        print()

if __name__ == "__main__":
    main()
    show_available_envs()
    
    print("\n=== 使用说明 ===")
    print("1. 使用 gym.make(env_id) 创建环境")
    print("2. 环境ID格式: Foraging{-2s}-{size}x{size}-{agents}p-{foods}f{-coop}-v3")
    print("3. 通过 env.field 访问Field类功能")
    print("4. 支持所有标准的Gymnasium接口")
    print("\n🌾 Happy Foraging! 🤖") 