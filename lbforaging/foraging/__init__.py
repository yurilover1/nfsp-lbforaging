from lbforaging.foraging.environment import ForagingEnv  # noqa
from lbforaging.foraging.environment_3d import ForagingEnv3D  # noqa

# 为了确保注册正确执行，务必添加以下导入
from gymnasium.envs.registration import register

# 注册三维环境
register(
    id='Foraging3D-v0',
    entry_point='lbforaging.foraging.environment_3d:ForagingEnv3D',
)

# 如果已有其他环境注册，保留原有注册代码