from itertools import product

from gymnasium import register
from gymnasium.envs.registration import EnvSpec

import sys
import lbforaging.foraging as foraging
sys.modules['foraging'] = foraging

sizes = range(3, 20)
players = range(2, 10)
foods = range(1, 10)
max_food_level = [None]  # [None, 1]
coop = [True, False]
partial_obs = [True, False]
pens = [False]  # [True, False]


for s, p, f, mfl, c, po, pen in product(
    sizes, players, foods, max_food_level, coop, partial_obs, pens
):
    # 环境ID
    env_id = "Foraging{4}-{0}x{0}-{1}p-{2}f{3}{5}{6}-v3".format(
        s,
        p,
        f,
        "-coop" if c else "",
        "-2s" if po else "",
        "-ind" if mfl else "",
        "-pen" if pen else "",
    )
    
    # 创建环境规格
    spec = EnvSpec(
        id=env_id,
        entry_point="lbforaging.foraging:ForagingEnv",
        max_episode_steps=None,  # 不在这里设置最大步数
    )
    
    # 注册环境并提供参数
    register(
        id=env_id,
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": p,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (s, s),
            "min_food_level": 1,
            "max_food_level": mfl,
            "max_num_food": f,
            "sight": 2 if po else s,
            "max_episode_steps": 20,  # 默认值作为环境参数传递
            "force_coop": c,
            "grid_observation": False,
            "penalty": 0.1 if pen else 0.0,
            "three_layer_obs": False,
        },
    )

def register_grid_envs():
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
            
            # 创建环境规格
            spec = EnvSpec(
                id=env_id,
                entry_point="lbforaging.foraging:ForagingEnv",
                max_episode_steps=None,  # 不在这里设置最大步数
            )
            
            register(
                id=env_id,
                entry_point="lbforaging.foraging:ForagingEnv",
                kwargs={
                    "players": p,
                    "min_player_level": 1,
                    "max_player_level": 2,
                    "field_size": (s, s),
                    "min_food_level": 1,
                    "max_food_level": mfl,
                    "max_num_food": f,
                    "sight": sight,
                    "max_episode_steps": 50,  # 默认值作为环境参数传递
                    "force_coop": c,
                    "grid_observation": False,
                    "three_layer_obs": False,
                },
            )
