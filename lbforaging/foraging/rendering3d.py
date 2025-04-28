"""
3D rendering of the level based foraging domain
使用PyOpenGL和pyglet实现3D渲染
"""

import math
import os
import sys
import numpy as np
import six
from gymnasium import error

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"

try:
    import pyglet
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        """
    Cannot import pyglet or OpenGL.
    Make sure you have installed both:
    - pip install pyglet
    - pip install PyOpenGL PyOpenGL_accelerate
    """
    )

# 定义一些常用颜色 (R, G, B)
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_BLUE = (0, 0, 255)
_YELLOW = (255, 255, 0)
_PURPLE = (128, 0, 128)
_ORANGE = (255, 165, 0)

_BACKGROUND_COLOR = (240, 240, 240)  # 浅灰色背景
_GRID_COLOR = (180, 180, 180)        # 灰色网格线
_FLOOR_COLOR = (200, 200, 200)       # 地板颜色

def get_display(spec):
    """将显示规格转换为实际的Display对象"""
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )

class Viewer3D(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols, self.depth = world_size

        # 设置网格和元素大小
        self.grid_size = 40  # 每个网格单元的大小
        self.cube_size = 30  # 实体的大小
        
        # 计算窗口大小，加入一些边距
        self.width = int(1024)  # 固定窗口宽度
        self.height = int(768)  # 固定窗口高度
        
        # 创建窗口
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display, caption="3D LB-Foraging"
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        # 启用OpenGL功能
        glEnable(GL_DEPTH_TEST)  # 启用深度测试
        glEnable(GL_BLEND)       # 启用混合
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # 加载资源
        script_dir = os.path.dirname(__file__)
        pyglet.resource.path = [os.path.join(script_dir, "icons")]
        pyglet.resource.reindex()
        
        # 加载纹理
        self.texture_apple = pyglet.resource.texture("apple.png")
        self.texture_agent = pyglet.resource.texture("agent.png")
        
        # 相机设置
        self.camera_distance = max(self.rows, self.cols, self.depth) * 2.5
        self.camera_x = 0
        self.camera_y = 0
        self.camera_z = self.camera_distance
        self.camera_rotation_x = 30  # 初始X轴旋转角度
        self.camera_rotation_y = 45  # 初始Y轴旋转角度
        
        # 设置键盘事件处理
        self.keys = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        
        # 创建批处理对象用于渲染
        self.batch = pyglet.graphics.Batch()
        
        # 初始化帧率显示
        self.fps_display = pyglet.window.FPSDisplay(self.window)

    def close(self):
        """关闭查看器"""
        self.window.close()

    def window_closed_by_user(self):
        """用户关闭窗口时的处理"""
        self.isopen = False
        exit()
    
    def update_camera(self, dt):
        """更新相机位置和方向（可通过键盘控制）"""
        # 相机旋转
        if self.keys[pyglet.window.key.LEFT]:
            self.camera_rotation_y -= 2
        if self.keys[pyglet.window.key.RIGHT]:
            self.camera_rotation_y += 2
        if self.keys[pyglet.window.key.UP]:
            self.camera_rotation_x -= 2
        if self.keys[pyglet.window.key.DOWN]:
            self.camera_rotation_x += 2
            
        # 相机缩放
        if self.keys[pyglet.window.key.PLUS] or self.keys[pyglet.window.key.NUM_ADD]:
            self.camera_distance -= 0.5
        if self.keys[pyglet.window.key.MINUS] or self.keys[pyglet.window.key.NUM_SUBTRACT]:
            self.camera_distance += 0.5
            
        # 限制旋转角度
        self.camera_rotation_x = max(min(self.camera_rotation_x, 89), -89)
        self.camera_distance = max(self.camera_distance, 2)
        
        # 计算相机位置
        rads_y = math.radians(self.camera_rotation_y)
        rads_x = math.radians(self.camera_rotation_x)
        self.camera_x = self.camera_distance * math.sin(rads_y) * math.cos(rads_x)
        self.camera_y = self.camera_distance * math.sin(rads_x)
        self.camera_z = self.camera_distance * math.cos(rads_y) * math.cos(rads_x)

    def setup_3d(self):
        """设置3D渲染环境"""
        # 清除屏幕
        glClearColor(*[c/255.0 for c in _BACKGROUND_COLOR], 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 设置透视投影
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 1000)
        
        # 设置模型视图
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # 设置相机位置和视角
        # 相机位置: (camera_x, camera_y, camera_z)
        # 观察点: (0, 0, 0) - 环境中心
        # 上方向: (0, 1, 0)
        gluLookAt(
            self.camera_x, self.camera_y, self.camera_z,  # 相机位置
            0, 0, 0,                                      # 观察点
            0, 1, 0                                       # 上方向
        )
        
        # 计算环境中心点，使其位于原点
        center_x = (self.rows * self.grid_size) / 2
        center_y = (self.cols * self.grid_size) / 2
        center_z = (self.depth * self.grid_size) / 2
        
        # 平移使得环境中心在原点
        glTranslatef(-center_x, -center_y, -center_z)

    def render(self, env, return_rgb_array=False):
        """渲染环境"""
        pyglet.clock.tick()  # 更新时钟
        self.update_camera(1/60.0)  # 更新相机
        
        self.window.switch_to()
        self.window.dispatch_events()
        
        self.setup_3d()
        
        # 绘制3D网格
        self._draw_grid()
        
        # 绘制食物
        self._draw_food(env)
        
        # 绘制智能体
        self._draw_players(env)
        
        # 绘制帧率
        self.fps_display.draw()
        
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        """绘制3D网格"""
        # 启用线框模式
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(1.0)
        
        # 设置网格颜色
        glColor4f(*[c/255.0 for c in _GRID_COLOR], 0.5)
        
        # 绘制水平网格
        for z in range(self.depth + 1):
            for x in range(self.rows + 1):
                glBegin(GL_LINES)
                glVertex3f(x * self.grid_size, 0, z * self.grid_size)
                glVertex3f(x * self.grid_size, self.cols * self.grid_size, z * self.grid_size)
                glEnd()
                
        for z in range(self.depth + 1):
            for y in range(self.cols + 1):
                glBegin(GL_LINES)
                glVertex3f(0, y * self.grid_size, z * self.grid_size)
                glVertex3f(self.rows * self.grid_size, y * self.grid_size, z * self.grid_size)
                glEnd()
        
        # 绘制垂直网格
        for x in range(self.rows + 1):
            for y in range(self.cols + 1):
                glBegin(GL_LINES)
                glVertex3f(x * self.grid_size, y * self.grid_size, 0)
                glVertex3f(x * self.grid_size, y * self.grid_size, self.depth * self.grid_size)
                glEnd()
        
        # 恢复填充模式
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # 绘制底面
        glColor4f(*[c/255.0 for c in _FLOOR_COLOR], 0.3)
        glBegin(GL_QUADS)
        glVertex3f(0, 0, 0)
        glVertex3f(self.rows * self.grid_size, 0, 0)
        glVertex3f(self.rows * self.grid_size, self.cols * self.grid_size, 0)
        glVertex3f(0, self.cols * self.grid_size, 0)
        glEnd()

    def _draw_food(self, env):
        """绘制食物"""
        for food in env.food_items:
            if not food.collected:
                x, y, z = food.position
                self._draw_cube(
                    x * self.grid_size + self.grid_size/2,  # 中心点x
                    y * self.grid_size + self.grid_size/2,  # 中心点y
                    z * self.grid_size + self.grid_size/2,  # 中心点z
                    self.cube_size * 0.8,  # 稍小一点
                    _RED,  # 食物颜色
                    food.level  # 显示等级
                )

    def _draw_players(self, env):
        """绘制智能体"""
        for i, player in enumerate(env.players):
            x, y, z = player.position
            
            # 选择不同的颜色
            colors = [_BLUE, _GREEN, _YELLOW, _PURPLE, _ORANGE]
            color = colors[i % len(colors)]
            
            self._draw_cube(
                x * self.grid_size + self.grid_size/2,  # 中心点x
                y * self.grid_size + self.grid_size/2,  # 中心点y
                z * self.grid_size + self.grid_size/2,  # 中心点z
                self.cube_size,  # 智能体尺寸
                color,  # 智能体颜色
                player.level  # 显示等级
            )

    def _draw_cube(self, x, y, z, size, color, level=None):
        """绘制一个立方体
        
        参数:
            x, y, z: 立方体中心点的坐标
            size: 立方体边长
            color: 立方体颜色 (R,G,B)
            level: 如果不为None，则在立方体上显示等级
        """
        half = size / 2
        
        # 调整透明度
        r, g, b = color
        glColor4f(r/255.0, g/255.0, b/255.0, 0.8)
        
        # 绘制立方体
        glBegin(GL_QUADS)
        
        # 前面
        glVertex3f(x - half, y - half, z + half)
        glVertex3f(x + half, y - half, z + half)
        glVertex3f(x + half, y + half, z + half)
        glVertex3f(x - half, y + half, z + half)
        
        # 后面
        glVertex3f(x - half, y - half, z - half)
        glVertex3f(x - half, y + half, z - half)
        glVertex3f(x + half, y + half, z - half)
        glVertex3f(x + half, y - half, z - half)
        
        # 顶面
        glVertex3f(x - half, y + half, z - half)
        glVertex3f(x - half, y + half, z + half)
        glVertex3f(x + half, y + half, z + half)
        glVertex3f(x + half, y + half, z - half)
        
        # 底面
        glVertex3f(x - half, y - half, z - half)
        glVertex3f(x + half, y - half, z - half)
        glVertex3f(x + half, y - half, z + half)
        glVertex3f(x - half, y - half, z + half)
        
        # 右面
        glVertex3f(x + half, y - half, z - half)
        glVertex3f(x + half, y + half, z - half)
        glVertex3f(x + half, y + half, z + half)
        glVertex3f(x + half, y - half, z + half)
        
        # 左面
        glVertex3f(x - half, y - half, z - half)
        glVertex3f(x - half, y - half, z + half)
        glVertex3f(x - half, y + half, z + half)
        glVertex3f(x - half, y + half, z - half)
        
        glEnd()
        
        # 绘制立方体的边框
        glColor4f(0.1, 0.1, 0.1, 1.0)  # 黑色边框
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(2.0)
        
        glBegin(GL_QUADS)
        # 前面
        glVertex3f(x - half, y - half, z + half)
        glVertex3f(x + half, y - half, z + half)
        glVertex3f(x + half, y + half, z + half)
        glVertex3f(x - half, y + half, z + half)
        
        # 后面
        glVertex3f(x - half, y - half, z - half)
        glVertex3f(x - half, y + half, z - half)
        glVertex3f(x + half, y + half, z - half)
        glVertex3f(x + half, y - half, z - half)
        
        # 顶面
        glVertex3f(x - half, y + half, z - half)
        glVertex3f(x - half, y + half, z + half)
        glVertex3f(x + half, y + half, z + half)
        glVertex3f(x + half, y + half, z - half)
        
        # 底面
        glVertex3f(x - half, y - half, z - half)
        glVertex3f(x + half, y - half, z - half)
        glVertex3f(x + half, y - half, z + half)
        glVertex3f(x - half, y - half, z + half)
        
        # 右面
        glVertex3f(x + half, y - half, z - half)
        glVertex3f(x + half, y + half, z - half)
        glVertex3f(x + half, y + half, z + half)
        glVertex3f(x + half, y - half, z + half)
        
        # 左面
        glVertex3f(x - half, y - half, z - half)
        glVertex3f(x - half, y - half, z + half)
        glVertex3f(x - half, y + half, z + half)
        glVertex3f(x - half, y + half, z - half)
        
        glEnd()
        
        # 恢复填充模式
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # 如果有等级参数，则显示等级
        if level is not None:
            # 在立方体前面绘制等级数字
            # 使用简单的线条绘制数字
            self._draw_number(x, y, z + half + 0.1, level, half * 1.2)

    def _draw_number(self, x, y, z, number, size):
        """使用线条绘制数字
        
        参数:
            x, y, z: 数字的中心位置
            number: 要绘制的数字
            size: 数字的大小
        """
        number = int(number)  # 确保是整数
        
        # 设置线宽和颜色
        glLineWidth(3.0)
        glColor4f(1.0, 1.0, 1.0, 1.0)  # 白色
        
        # 数字的定义（简化的线段集合）
        segments = {
            1: [(0, -1, 0, 1)],  # 垂直中线
            2: [(-1, 1, 1, 1), (1, 1, 1, 0), (1, 0, -1, 0), (-1, 0, -1, -1), (-1, -1, 1, -1)],  # 2的形状
            3: [(-1, 1, 1, 1), (1, 1, 1, -1), (-1, 0, 1, 0), (-1, -1, 1, -1)],  # 3的形状
            4: [(-1, 1, -1, 0), (-1, 0, 1, 0), (0, 1, 0, -1)],  # 4的形状
            5: [(1, 1, -1, 1), (-1, 1, -1, 0), (-1, 0, 1, 0), (1, 0, 1, -1), (1, -1, -1, -1)],  # 5的形状
            6: [(1, 1, -1, 1), (-1, 1, -1, -1), (-1, -1, 1, -1), (1, -1, 1, 0), (1, 0, -1, 0)],  # 6的形状
            7: [(-1, 1, 1, 1), (1, 1, 0, -1)],  # 7的形状
            8: [(-1, 1, 1, 1), (1, 1, 1, -1), (1, -1, -1, -1), (-1, -1, -1, 1), (-1, 0, 1, 0)],  # 8的形状
            9: [(1, -1, 1, 1), (1, 1, -1, 1), (-1, 1, -1, 0), (-1, 0, 1, 0)]  # 9的形状
        }
        
        # 如果数字在1-9范围内，绘制对应形状
        if 1 <= number <= 9 and number in segments:
            half = size / 2
            scale = size / 2  # 缩放因子
            
            glBegin(GL_LINES)
            for line in segments[number]:
                x1, y1, x2, y2 = line
                glVertex3f(x + x1 * scale, y + y1 * scale, z)
                glVertex3f(x + x2 * scale, y + y2 * scale, z)
            glEnd()
    
    def set_bounds(self, left, right, bottom, top):
        """向后兼容的边界设置函数"""
        pass 