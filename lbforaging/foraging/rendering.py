"""
2D rendering of the level based foraging domain using Matplotlib
"""

import os

# 设置matplotlib后端为MacOSX，适配macOS动态刷新
import matplotlib
import numpy as np

matplotlib.use('TkAgg')  # 使用跨平台的TkAgg后端
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
        self.fig.canvas.manager.set_window_title('Level Based Foraging')
        
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

    def render(self, render_grid, agent_positions=None, agent_levels=None, food_positions=None, return_rgb_array=False):
        """
        渲染环境状态
        
        Args:
            render_grid: 渲染网格
            agent_positions: 智能体位置列表 [(row, col), ...]
            agent_levels: 智能体等级列表 [level1, level2, ...]
            food_positions: 食物位置和等级字典 {(row, col): level, ...}
            return_rgb_array: 是否返回RGB数组
        """
        if not self.isopen:
            return None
        self.ax.clear()
        self.ax.set_xlim(0, self.cols)
        self.ax.set_ylim(0, self.rows)
        self.ax.set_xticks(np.arange(0, self.cols+1, 1))
        self.ax.set_yticks(np.arange(0, self.rows+1, 1))
        self.ax.grid(True, color='black', linewidth=1)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # 绘制基本网格
        self._draw_food(render_grid)
        
        # 如果提供了新的位置信息，使用新的信息绘制
        if agent_positions is not None and agent_levels is not None:
            self._draw_players_from_positions(agent_positions, agent_levels)
        else:
            self._draw_players(render_grid)
            
        # 如果提供了食物位置信息，重新绘制食物
        if food_positions is not None:
            self._draw_food_from_positions(food_positions)
            
        # 添加图例
        if food_positions is not None:
            food_levels = set(food_positions.values())
            self._add_legend_with_levels(food_levels, len(agent_positions) if agent_positions else 0)
        else:
            self._add_legend(render_grid)
            
        self.ax.set_title('Level Based Foraging')
        self.fig.canvas.draw()
        plt.pause(0.01)  # 短暂暂停以更新显示，不阻塞程序执行
        if return_rgb_array:
            canvas = self.fig.canvas
            width, height = self.fig.get_size_inches() * self.fig.get_dpi()
            buffer = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            img_array = buffer.reshape(int(height), int(width), 3)
            return img_array
        return self.isopen

    def _draw_food(self, render_grid):
        """从渲染网格中绘制食物"""
        for i in range(self.rows):
            for j in range(self.cols):
                val = render_grid[i][j]
                if 1 <= val < 100:  # 食物
                    if self.img_apple is not None:
                        self.ax.imshow(self.img_apple, extent=[j, j+1, self.rows-i-1, self.rows-i])
                    else:
                        rect = Rectangle((j, self.rows-i-1), 1, 1, facecolor='red', alpha=0.6)
                        self.ax.add_patch(rect)
                    self._draw_badge(i, j, val)
                    
    def _draw_food_from_positions(self, food_positions):
        """从位置字典中绘制食物
        
        Args:
            food_positions: 食物位置和等级字典 {(row, col): level, ...}
        """
        for pos, level in food_positions.items():
            i, j = pos  # 行、列坐标
            
            # 绘制食物图标
            if self.img_apple is not None:
                self.ax.imshow(self.img_apple, extent=[j, j+1, self.rows-i-1, self.rows-i])
            else:
                rect = Rectangle((j, self.rows-i-1), 1, 1, facecolor='red', alpha=0.6)
                self.ax.add_patch(rect)
                
            # 绘制食物等级
            self._draw_badge(i, j, level)

    def _draw_players(self, render_grid):
        """从渲染网格中绘制智能体"""
        agent_colors = ['blue', 'green', 'purple', 'orange']
        for i in range(self.rows):
            for j in range(self.cols):
                val = render_grid[i][j]
                if val >= 100:  # 智能体
                    agent_idx = (val - 100) % len(agent_colors)
                    color = agent_colors[agent_idx]
                    border = Rectangle((j, self.rows-i-1), 1, 1, fill=False, edgecolor=color, linewidth=3)
                    self.ax.add_patch(border)
                    self.ax.imshow(self.img_agent, extent=[j, j+1, self.rows-i-1, self.rows-i])
                    self.ax.text(j + 0.5, self.rows-i-0.5, str(agent_idx), color='white', fontsize=14, fontweight='bold', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor=color, alpha=0.7, boxstyle='circle'))
                    label_y = self.rows-i-1.2
                    if i == 0:
                        label_y = self.rows-i+0.2
                    self.ax.text(j + 0.5, label_y, f"Agent {agent_idx}", color=color, fontsize=10, fontweight='bold', horizontalalignment='center', verticalalignment='center')
                    self._draw_badge(i, j, val-100, agent_color=color)
                    
    def _draw_players_from_positions(self, agent_positions, agent_levels):
        """从位置列表中绘制智能体
        
        Args:
            agent_positions: 智能体位置列表 [(row, col), ...]
            agent_levels: 智能体等级列表 [level1, level2, ...]
        """
        agent_colors = ['blue', 'green', 'purple', 'orange']
        
        for idx, (pos, level) in enumerate(zip(agent_positions, agent_levels)):
            if pos is None:  # 跳过无效位置
                continue
                
            i, j = pos  # 行、列坐标
            color = agent_colors[idx % len(agent_colors)]
            
            # 绘制边框
            border = Rectangle((j, self.rows-i-1), 1, 1, fill=False, edgecolor=color, linewidth=3)
            self.ax.add_patch(border)
            
            # 绘制智能体图标
            self.ax.imshow(self.img_agent, extent=[j, j+1, self.rows-i-1, self.rows-i])
            
            # 绘制智能体编号
            self.ax.text(j + 0.5, self.rows-i-0.5, str(idx), color='white', fontsize=14, 
                        fontweight='bold', horizontalalignment='center', verticalalignment='center', 
                        bbox=dict(facecolor=color, alpha=0.7, boxstyle='circle'))
            
            # 绘制标签
            label_y = self.rows-i-1.2
            if i == 0:
                label_y = self.rows-i+0.2
            self.ax.text(j + 0.5, label_y, f"Agent {idx}", color=color, fontsize=10, 
                        fontweight='bold', horizontalalignment='center', verticalalignment='center')
            
            # 绘制等级标志
            self._draw_badge(i, j, level, agent_color=color)

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

    def _add_legend(self, render_grid):
        """从渲染网格生成图例"""
        agent_colors = ['blue', 'green', 'purple', 'orange']
        legend_elements = []
        # 智能体图例
        for idx, color in enumerate(agent_colors):
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Agent {idx}')
            )
        # 食物图例
        food_levels = set()
        for i in range(self.rows):
            for j in range(self.cols):
                val = render_grid[i][j]
                if 1 <= val < 100:
                    food_levels.add(val)
        for level in sorted(food_levels):
            legend_elements.append(
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label=f'Food (Level {int(level)})')
            )
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1), fontsize='small')
            
    def _add_legend_with_levels(self, food_levels, num_agents):
        """直接从等级信息生成图例
        
        Args:
            food_levels: 食物等级集合
            num_agents: 智能体数量
        """
        agent_colors = ['blue', 'green', 'purple', 'orange']
        legend_elements = []
        
        # 智能体图例
        for idx in range(min(num_agents, len(agent_colors))):
            color = agent_colors[idx]
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Agent {idx}')
            )
            
        # 食物图例
        for level in sorted(food_levels):
            legend_elements.append(
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label=f'Food (Level {int(level)})')
            )
            
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1), fontsize='small')