from itertools import product

from gymnasium import register

# 移除可能导致循环导入的代码
# import sys
# import lbforaging.foraging as foraging
# sys.modules['foraging'] = foraging

# 延迟导入核心类，避免循环导入
# from lbforaging.foraging import ForagingEnv
# from lbforaging.foraging.types import FieldType, FieldPoint, Field

# 定义环境参数范围 - 减少组合以避免注册过多环境
sizes = [5]  # 只使用5x5的环境
players = [2]  # 只使用2个玩家
foods = [2]  # 只使用2个食物
max_food_level = [None]  # [None, 1]
coop = [False]  # 只使用非合作环境
partial_obs = [False]  # 不使用部分观察
pens = [False]  # 不使用惩罚

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
            "min_player_level": 2,
            "max_player_level": 2,
            "field_size": (s, s),
            "min_food_level": 1,
            "max_food_level": 2 if mfl is None else mfl,
            "max_num_food": f,
            "sight": 2 if po else s,
            "max_episode_steps": 30,
            "force_coop": c,
            "observe_agent_levels": True,
            "render_mode": None,
            "step_reward_factor": 0.4,
            "attraction_reward_factor": 0.2,
            "decay_rate": 0.01,
            "seed": None,
        },
    )

def register_grid_envs():
    """注册网格环境变体 - 只注册一个基本环境"""
    # 只注册一个基本环境
    env_id = "Foraging-grid-5x5-2p-2f-v3"
    register(
        id=env_id,
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "num_agents": 2,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (5, 5),
            "min_food_level": 1,
            "max_food_level": 2,
            "max_num_food": 2,
            "sight": 5,
            "max_episode_steps": 30,
            "force_coop": False,
            "observe_agent_levels": True,
            "render_mode": None,
            "step_reward_factor": 0.1,
            "attraction_reward_factor": 0.5,
            "decay_rate": 0.01,
            "seed": None,
        },
    )

# 自动注册网格环境
try:
    register_grid_envs()
    print("环境注册成功")
except Exception as e:
    print(f"环境注册失败: {e}")


