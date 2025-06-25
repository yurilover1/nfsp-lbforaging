"""
2D rendering of the level based foraging domain using Matplotlib
"""

import math
import os
import sys
import numpy as np
# 设置matplotlib后端为TkAgg，更兼容WSL
import matplotlib
matplotlib.use('TkAgg')  # 在导入pyplot之前设置
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image

# Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK

class Viewer(object):
    def __init__(self, world_size):
        self.rows, self.cols = world_size
        self.grid_size = 50
        self.icon_size = 20
        
        self.width = self.cols * self.grid_size
        self.height = self.rows * self.grid_size
        
        # 创建图形和坐标轴
        plt.ion()  # 交互模式
        self.fig, self.ax = plt.subplots(figsize=(self.cols*0.8, self.rows*0.8))
        
        # 加载图标
        script_dir = os.path.dirname(__file__)
        self.img_agent = Image.open(os.path.join(script_dir, "icons/agent.png"))
        self.img_apple = Image.open(os.path.join(script_dir, "icons/apple.png"))
        
        self.isopen = True
        
        # 设置窗口标题
        self.fig.canvas.set_window_title('Level Based Foraging')
        
        # 显式调用一次绘图，确保窗口显示
        self.ax.set_xlim(0, self.cols)
        self.ax.set_ylim(0, self.rows)
        self.ax.grid(True)
        self.fig.canvas.draw()
        plt.pause(0.001)  # 短暂暂停以确保窗口显示

    def close(self):
        plt.close(self.fig)
        self.isopen = False
        
    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def render(self, env, return_rgb_array=False):
        if not self.isopen:
            return None
            
        self.ax.clear()
        
        # 设置坐标轴
        self.ax.set_xlim(0, self.cols)
        self.ax.set_ylim(0, self.rows)
        self.ax.set_xticks(np.arange(0, self.cols+1, 1))
        self.ax.set_yticks(np.arange(0, self.rows+1, 1))
        self.ax.grid(True, color='black', linewidth=1)
        
        # 隐藏坐标轴标签
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # 绘制食物
        self._draw_food(env)
        
        # 绘制智能体
        self._draw_players(env)
        
        # 添加图例
        self._add_legend(env)
        
        # 显示当前步数
        self.ax.set_title(f'Step: {env.current_step}/{env._max_episode_steps}')
        
        # 更新画布
        self.fig.canvas.draw()
        plt.pause(0.001)  # 短暂暂停以确保窗口更新
        
        if return_rgb_array:
            # 从画布中获取RGB数组
            canvas = self.fig.canvas
            width, height = self.fig.get_size_inches() * self.fig.get_dpi()
            buffer = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            img_array = buffer.reshape(int(height), int(width), 3)
            return img_array
        
        return self.isopen

    def _draw_food(self, env):
        # 绘制食物
        idxes = list(zip(*env.field.nonzero()))
        for row, col in idxes:
            if self.img_apple is not None:
                # 使用图像
                self.ax.imshow(self.img_apple, extent=[col, col+1, self.rows-row-1, self.rows-row])
            else:
                # 使用简单矩形
                rect = Rectangle((col, self.rows-row-1), 1, 1, facecolor='red', alpha=0.6)
                self.ax.add_patch(rect)
                
            # 绘制食物等级
            self._draw_badge(row, col, env.field[row, col])

    def _draw_players(self, env):
        # 绘制玩家
        agent_colors = ['blue', 'green', 'purple', 'orange']
        
        for i, player in enumerate(env.players):
            row, col = player.position
            color = agent_colors[i % len(agent_colors)]
            
            # 使用图像，但添加彩色边框区分智能体
            # 先绘制彩色边框
            border = Rectangle((col, self.rows-row-1), 1, 1, 
                              fill=False, edgecolor=color, linewidth=3)
            self.ax.add_patch(border)
            
            # 然后绘制智能体图像
            self.ax.imshow(self.img_agent, extent=[col, col+1, self.rows-row-1, self.rows-row])
            
            # 添加智能体编号标识 - 在智能体中心
            self.ax.text(col + 0.5, self.rows-row-0.5, str(i),
                         color='white', fontsize=14, fontweight='bold',
                         horizontalalignment='center', verticalalignment='center',
                         bbox=dict(facecolor=color, alpha=0.7, boxstyle='circle'))
            
            # 智能体标签位置 - 根据位置调整，避免超出边界
            label_y = self.rows-row-1.2  # 默认在上方
            
            # 如果在顶部边缘，则将标签放在下方
            if row == 0:
                label_y = self.rows-row+0.2
                
            self.ax.text(col + 0.5, label_y, f"Agent {i}",
                         color=color, fontsize=10, fontweight='bold',
                         horizontalalignment='center', verticalalignment='center')
                
            # 绘制玩家等级 - 调整位置确保可见
            self._draw_badge(row, col, player.level, agent_color=color)
    
    def _draw_badge(self, row, col, level, agent_color=None):
        """绘制等级标志，可选择与智能体颜色匹配"""
        # 调整徽章位置，确保在智能体内可见
        x = col + 0.75
        y = self.rows - row - 0.25
        
        # 绘制圆形背景
        circle = plt.Circle((x, y), 0.15, 
                           color='white', ec='black' if agent_color is None else agent_color)
        self.ax.add_patch(circle)
        
        # 显示等级数字
        self.ax.text(x, y, str(int(level)), 
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=10, color='black')

    def _add_legend(self, env):
        """添加图例，标识智能体和食物"""
        agent_colors = ['blue', 'green', 'purple', 'orange']
        legend_elements = []
        
        # 添加智能体图例
        for i, player in enumerate(env.players):
            if i >= len(agent_colors):
                break
                
            color = agent_colors[i]
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color, markersize=10, 
                          label=f'Agent {i} (Level {player.level})')
            )
        
        # 添加食物图例
        if np.any(env.field > 0):
            food_levels = np.unique(env.field[env.field > 0])
            for level in food_levels:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='s', color='w',
                              markerfacecolor='red', markersize=10,
                              label=f'Food (Level {int(level)})')
                )
        
        # 放置图例在图的右上角，不遮挡主要内容
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper right', 
                          bbox_to_anchor=(1.1, 1), fontsize='small')