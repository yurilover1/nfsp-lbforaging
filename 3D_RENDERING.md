# 3D渲染功能使用指南

## 概述

本项目为Level-Based Foraging环境添加了3D渲染功能，使用PyOpenGL和Pyglet实现，可以更直观地可视化多智能体协作环境。

## 依赖安装

首先，确保安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：
- PyOpenGL
- PyOpenGL_accelerate
- pyglet==1.5.27
- numpy

## 使用方法

### 方法1：直接运行示例脚本

```bash
python 3d_use.py
```

此脚本将创建一个3D环境并使用3D渲染模式显示。

### 方法2：在自定义代码中使用

```python
from lbforaging.foraging import ForagingEnv3D

# 创建3D环境
env = ForagingEnv3D(
    n_rows=6,
    n_cols=6,
    n_depth=6,
    num_agents=3,
    num_food=5
)

# 重置环境
observation = env.reset()

# 开始渲染（使用3D模式）
env.render(mode='3d')

# 获取渲染器引用以便访问键盘控制
viewer = env.viewer3d

# 循环与环境交互
while True:
    # 示例：执行随机动作
    actions = [...]  # 为每个智能体定义动作
    obs, rewards, done, info = env.step(actions)
    
    # 更新渲染
    env.render(mode='3d')
    
    if done:
        break

# 关闭环境
env.close()
```

## 控制说明

3D渲染器支持交互式控制：

- **方向键**：旋转相机视角
  - 上/下：调整相机俯仰角
  - 左/右：调整相机水平旋转角
- **+/-键**：缩放视图（放大/缩小）
- **ESC键**：退出渲染

## 渲染元素说明

- **网格**：表示3D环境的边界
- **彩色立方体**：表示智能体
  - 每种颜色代表不同的智能体
  - 立方体上的数字表示智能体的等级
- **红色小立方体**：表示食物
  - 立方体上的数字表示食物的等级

## 故障排除

如果遇到渲染问题：

1. 确保已正确安装所有依赖
2. 对于MacOS用户，可能需要使用以下命令运行：
   ```bash
   PYTHONPATH=. python 3d_use.py
   ```
3. 如果出现OpenGL相关错误，请检查您的图形驱动程序是否支持OpenGL 3.3+

## 开发说明

3D渲染相关的主要文件：

- `lbforaging/foraging/rendering3d.py`：3D渲染器实现
- `3d_use.py`：使用示例
- `3d_rendering_test.py`：另一个测试示例 