#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D LB-Foraging环境演示
使用3D渲染模式可视化环境状态
"""

from lbforaging.foraging import ForagingEnv3D
import numpy as np
import time

def main():
    # 创建3D环境
    env = ForagingEnv3D(
        n_rows=6,       # X轴尺寸
        n_cols=6,       # Y轴尺寸
        n_depth=6,      # Z轴尺寸
        num_agents=3,   # 智能体数量
        num_food=5,     # 食物数量
        max_player_level=3,  # 最大玩家等级
        max_food_level=5,    # 最大食物等级
        grid_observation=True   # 使用网格观察
    )
    
    # 重置环境
    observation = env.reset()
    
    print("控制说明:")
    print("- 方向键: 旋转相机")
    print("- +/- 键: 缩放相机")
    print("- 空格键: 执行随机动作")
    print("- ESC键: 退出")
    print("\n按空格键执行随机动作，按ESC退出")
    
    # 渲染初始状态
    env.render(mode='3d')
    
    # 导入pyglet以便访问按键常量
    import pyglet
    
    # 获取渲染器引用以检查按键
    viewer = env.viewer3d
    
    # 主循环
    done = False
    while not done and viewer.isopen:
        # 渲染状态
        viewer.render(env)
        
        # 处理按键输入
        if viewer.keys[pyglet.window.key.SPACE]:
            # 执行随机动作
            actions = [np.random.randint(0, 7) for _ in range(env.num_agents)]
            obs, rewards, done, info = env.step(actions)
            
            print(f"奖励: {rewards}")
            print(f"信息: {info}")
            
            # 添加短暂延迟避免连续触发
            time.sleep(0.2)
        
        # 退出条件
        if viewer.keys[pyglet.window.key.ESCAPE]:
            break
        
        # 给pyglet一些时间处理事件
        time.sleep(0.01)
    
    env.close()

if __name__ == "__main__":
    main() 