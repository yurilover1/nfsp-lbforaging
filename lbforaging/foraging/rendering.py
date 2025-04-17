"""
2D rendering of the level based foraging domain
"""

import math
import os
import sys
from pyglet import gl
import numpy as np
import six
from gymnasium import error

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
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


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 50
        self.icon_size = 20

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        script_dir = os.path.dirname(__file__)

        pyglet.resource.path = [os.path.join(script_dir, "icons")]
        pyglet.resource.reindex()

        self.img_apple = pyglet.resource.image("apple.png")
        self.img_agent = pyglet.resource.image("agent.png")

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env, return_rgb_array=False):
        glClearColor(*_WHITE, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_food(env)
        self._draw_players(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        # vertical lines
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,  # LEFT X
                        (self.grid_size + 1) * r + 1,  # Y
                        (self.grid_size + 1) * self.cols,  # RIGHT X
                        (self.grid_size + 1) * r + 1,  # Y
                    ),
                ),
                ("c3B", (*_BLACK, *_BLACK)),
            )

        # horizontal lines
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * c + 1,  # X
                        0,  # BOTTOM Y
                        (self.grid_size + 1) * c + 1,  # X
                        (self.grid_size + 1) * self.rows,  # TOP X
                    ),
                ),
                ("c3B", (*_BLACK, *_BLACK)),
            )
        batch.draw()

    def _draw_food(self, env):
        idxes = list(zip(*env.field.nonzero()))
        apples = []
        batch = pyglet.graphics.Batch()

        # print(env.field)
        for row, col in idxes:
            apples.append(
                pyglet.sprite.Sprite(
                    self.img_apple,
                    (self.grid_size + 1) * col,
                    self.height - (self.grid_size + 1) * (row + 1),
                    batch=batch,
                )
            )
        for a in apples:
            a.update(scale=self.grid_size / a.width)
        batch.draw()

        for row, col in idxes:
            self._draw_badge(row, col, env.field[row, col])

    def _draw_players(self, env):
        players = []
        batch = pyglet.graphics.Batch()

        for player in env.players:
            row, col = player.position
            players.append(
                pyglet.sprite.Sprite(
                    self.img_agent,
                    (self.grid_size + 1) * col,
                    self.height - (self.grid_size + 1) * (row + 1),
                    batch=batch,
                )
            )
        for p in players:
            p.update(scale=self.grid_size / p.width)
        batch.draw()
        for p in env.players:
            self._draw_badge(*p.position, p.level)

    def _draw_badge(self, row, col, level):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * (self.grid_size + 1) + (3 / 4) * (self.grid_size + 1)
        badge_y = (
            self.height
            - (self.grid_size + 1) * (row + 1)
            + (1 / 4) * (self.grid_size + 1)
        )

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_WHITE)
        circle.draw(GL_POLYGON)
        glColor3ub(*_BLACK)
        circle.draw(GL_LINE_LOOP)
        
        # 完全移除文本渲染，使用简单的数字符号绘制
        # 使用直线绘制数字符号，绕过文本渲染
        glColor3ub(*_BLACK)
        
        # 绘制表示数字的基本图形，这里仅处理1-9的数字
        digit = int(level)
        if 1 <= digit <= 9:
            # 线段宽度
            glLineWidth(2.0)
            
            # 设置相对大小和位置
            size = radius * 0.8
            x_center = badge_x
            y_center = badge_y
            
            if digit == 1:  # 绘制数字1
                # 垂直线条
                pyglet.graphics.draw(
                    2, GL_LINES,
                    ('v2f', (x_center, y_center - size/2, x_center, y_center + size/2))
                )
            elif digit == 2:  # 绘制数字2
                # 绘制2的简单表示（上横线，右竖线，中横线，左竖线，底横线）
                points = [
                    x_center - size/2, y_center + size/2,  # 左上
                    x_center + size/2, y_center + size/2,  # 右上
                    x_center + size/2, y_center,           # 右中
                    x_center - size/2, y_center,           # 左中
                    x_center - size/2, y_center - size/2,  # 左下
                    x_center + size/2, y_center - size/2   # 右下
                ]
                pyglet.graphics.draw(6, GL_LINE_STRIP, ('v2f', points))
            elif digit == 3:  # 绘制数字3
                # 右侧竖线和三条横线
                h_points = [
                    x_center - size/2, y_center + size/2, x_center + size/2, y_center + size/2,  # 上横
                    x_center - size/2, y_center, x_center + size/2, y_center,                     # 中横
                    x_center - size/2, y_center - size/2, x_center + size/2, y_center - size/2    # 下横
                ]
                pyglet.graphics.draw(6, GL_LINES, ('v2f', h_points))
            elif digit == 4:  # 绘制数字4
                # 左竖线，中横线，右竖线
                points = [
                    x_center - size/2, y_center + size/2,  # 左上
                    x_center - size/2, y_center,           # 左中
                    x_center + size/2, y_center + size/2,  # 右上
                    x_center + size/2, y_center - size/2   # 右下
                ]
                pyglet.graphics.draw(2, GL_LINES, ('v2f', points[0:4]))
                pyglet.graphics.draw(2, GL_LINES, ('v2f', points[2:6]))
                # 中横线
                pyglet.graphics.draw(2, GL_LINES, 
                    ('v2f', (x_center - size/2, y_center, x_center + size/2, y_center))
                )
            else:  # 对于5-9，简单绘制一个点表示有数字
                pyglet.graphics.draw(
                    1, GL_POINTS,
                    ('v2f', (x_center, y_center)),
                    ('c3B', _BLACK)
                )
            
            # 恢复线宽
            glLineWidth(1.0)