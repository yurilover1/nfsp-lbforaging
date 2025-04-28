#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D LB-Foraging环境渲染测试
使用Viewer3D类来渲染ForagingEnv3D环境
"""

from lbforaging.foraging import ForagingEnv3D
from lbforaging.foraging.rendering3d import Viewer3D
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
        grid_observation=True,   # 使用网格观察
        max_episode_steps=100    # 最大回合步数
    )
    
    # 初始化渲染器
    renderer = Viewer3D(world_size=(env.n_rows, env.n_cols, env.n_depth))
    
    # 重置环境
    env.reset()
    
    print("控制说明:")
    print("- 方向键: 旋转相机")
    print("- +/- 键: 缩放相机")
    print("- 空格键: 执行随机动作")
    print("- ESC键: 退出")
    
    # 主循环
    done = False
    while not done and renderer.isopen:
        # 渲染当前状态
        renderer.render(env)
        
        # 处理键盘输入
        if renderer.keys[pyglet.window.key.SPACE]:
            # 随机选择动作
            actions = [np.random.randint(0, 7) for _ in range(env.num_agents)]
            # 执行动作
            _, rewards, done, _ = env.step(actions)
            print(f"奖励: {rewards}")
            
            # 短暂延迟，避免连续触发
            time.sleep(0.2)
        
        if renderer.keys[pyglet.window.key.ESCAPE]:
            break
        
        # 给pyglet一些时间处理事件
        time.sleep(0.01)
    
    # 关闭环境和渲染器
    env.close()
    renderer.close()

if __name__ == "__main__":
    import pyglet
    main() 