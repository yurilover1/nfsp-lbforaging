from itertools import product
from gymnasium import register

import sys
import lbforaging.foraging as foraging
sys.modules['foraging'] = foraging

# 导出核心类，便于外部导入
from lbforaging.foraging import ForagingEnv, FieldType, FieldPoint, Field

# 定义环境参数范围
sizes = range(3, 20)
players = range(2, 10)
foods = range(1, 10)
max_food_level = [None]  # [None, 1]
coop = [True, False]
partial_obs = [True, False]
pens = [False]  # [True, False]

# 批量注册环境
for s, p, f, mfl, c, po, pen in product(
    sizes, players, foods, max_food_level, coop, partial_obs, pens
):
    # 环境ID格式
    env_id = "Foraging{4}-{0}x{0}-{1}p-{2}f{3}{5}{6}-v3".format(
        s,
        p,
        f,
        "-coop" if c else "",
        "-2s" if po else "",
        "-ind" if mfl else "",
        "-pen" if pen else "",
    )
    
    # 注册环境
    register(
        id=env_id,
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "num_agents": p,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (s, s),
            "min_food_level": 1,
            "max_food_level": 2 if mfl is None else mfl,
            "max_num_food": f,
            "sight": 2 if po else s,
            "max_episode_steps": 50,
            "force_coop": c,
            "observe_agent_levels": True,
            "render_mode": None,
            "step_reward_factor": 0.1,
            "attraction_reward_factor": 0.5,
            "decay_rate": 0.01,
            "seed": None,
        },
    )

def register_grid_envs():
    """注册网格环境变体"""
    for s, p, f, mfl, c in product(sizes, players, foods, max_food_level, coop):
        for sight in range(1, s + 1):
            env_id = "Foraging-grid{4}-{0}x{0}-{1}p-{2}f{3}{5}-v3".format(
                s,
                p,
                f,
                "-coop" if c else "",
                "" if sight == s else f"-{sight}s",
                "-ind" if mfl else "",
            )
            
            register(
                id=env_id,
                entry_point="lbforaging.foraging:ForagingEnv",
                kwargs={
                    "num_agents": p,
                    "min_player_level": 1,
                    "max_player_level": 2,
                    "field_size": (s, s),
                    "min_food_level": 1,
                    "max_food_level": 2 if mfl is None else mfl,
                    "max_num_food": f,
                    "sight": sight,
                    "max_episode_steps": 50,
                    "force_coop": c,
                    "observe_agent_levels": True,
                    "render_mode": None,
                    "step_reward_factor": 0.1,
                    "attraction_reward_factor": 0.5,
                    "decay_rate": 0.01,
                    "seed": None,
                },
            )

# 自动注册网格环境
register_grid_envs()

# 便捷函数：创建自定义环境
def create_env(num_agents=2, field_size=(8, 8), max_num_food=3, **kwargs):
    """
    直接创建自定义LBForaging环境的便捷函数
    
    Args:
        num_agents: 智能体数量
        field_size: 环境尺寸 (rows, cols)
        max_num_food: 最大食物数量
        **kwargs: 其他环境参数
        
    Returns:
        ForagingEnv: 创建的环境实例
        
    Example:
        >>> import lbforaging
        >>> env = lbforaging.create_env(num_agents=3, field_size=(10, 10), max_num_food=5)
    """
    # 设置默认参数
    default_kwargs = {
        'min_player_level': 1,
        'max_player_level': 2,
        'min_food_level': 1,
        'max_food_level': 2,
        'sight': 2,
        'max_episode_steps': 50,
        'force_coop': False,
        'observe_agent_levels': True,
        'render_mode': None,
        'step_reward_factor': 0.1,
        'attraction_reward_factor': 0.5,
        'decay_rate': 0.01,
        'seed': None,
    }
    
    # 合并参数
    default_kwargs.update(kwargs)
    
    return ForagingEnv(
        num_agents=num_agents,
        field_size=field_size,
        max_num_food=max_num_food,
        **default_kwargs
    )
