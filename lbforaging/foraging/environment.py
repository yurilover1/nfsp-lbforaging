from collections import namedtuple, defaultdict, deque
from enum import Enum
from itertools import product
import logging
import time
from typing import Iterable

import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def select_action(self, obs, flag_train=False):
        if hasattr(self.controller, '_step'):
            return self.controller._step(obs, is_train=flag_train)
        else:
            return self.controller.step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(gym.Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        min_player_level,
        max_player_level,
        min_food_level,
        max_food_level,
        field_size,
        max_num_food,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward=True,
        grid_observation=False,
        observe_agent_levels=True,
        penalty=0.0,
        render_mode=None,
        three_layer_obs=False,  # 新增参数：是否使用三层观测模式
        step_reward_factor=0.1,  # 新增：胜利时步数奖励系数
        distance_penalty_factor=0.05,  # 新增：失败时距离惩罚系数
    ):
        self.logger = logging.getLogger(__name__)
        self.render_mode = render_mode
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)

        self.penalty = penalty
        
        # 新增奖励调整参数
        self.step_reward_factor = step_reward_factor  # 胜利时步数奖励系数
        self.distance_penalty_factor = distance_penalty_factor  # 失败时距离惩罚系数

        # 观测模式设置
        self._grid_observation = grid_observation
        self._three_layer_obs = three_layer_obs  # 是否使用三层观测模式

        if isinstance(min_food_level, Iterable):
            assert (
                len(min_food_level) == max_num_food
            ), "min_food_level must be a scalar or a list of length max_num_food"
            self.min_food_level = np.array(min_food_level)
        else:
            self.min_food_level = np.array([min_food_level] * max_num_food)

        if max_food_level is None:
            self.max_food_level = None
        elif isinstance(max_food_level, Iterable):
            assert (
                len(max_food_level) == max_num_food
            ), "max_food_level must be a scalar or a list of length max_num_food"
            self.max_food_level = np.array(max_food_level)
        else:
            self.max_food_level = np.array([max_food_level] * max_num_food)

        if self.max_food_level is not None:
            # check if min_food_level is less than max_food_level
            for min_food_level, max_food_level in zip(
                self.min_food_level, self.max_food_level
            ):
                assert (
                    min_food_level <= max_food_level
                ), "min_food_level must be less than or equal to max_food_level for each food"

        self.max_num_food = max_num_food
        self._food_spawned = 0.0

        if isinstance(min_player_level, Iterable):
            assert (
                len(min_player_level) == players
            ), "min_player_level must be a scalar or a list of length players"
            self.min_player_level = np.array(min_player_level)
        else:
            self.min_player_level = np.array([min_player_level] * players)

        if isinstance(max_player_level, Iterable):
            assert (
                len(max_player_level) == players
            ), "max_player_level must be a scalar or a list of length players"
            self.max_player_level = np.array(max_player_level)
        else:
            self.max_player_level = np.array([max_player_level] * players)

        if self.max_player_level is not None:
            # check if min_player_level is less than max_player_level for each player
            for i, (min_player_level, max_player_level) in enumerate(
                zip(self.min_player_level, self.max_player_level)
            ):
                assert (
                    min_player_level <= max_player_level
                ), f"min_player_level must be less than or equal to max_player_level for each player but was {min_player_level} > {max_player_level} for player {i}"

        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._observe_agent_levels = observe_agent_levels

        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(6)] * len(self.players))
        )
        self.observation_space = gym.spaces.Tuple(
            tuple([self._get_observation_space()] * len(self.players))
        )

        self.viewer = None

        self.n_agents = len(self.players)

    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def _get_observation_space(self):
        """获取每个智能体的观测空间"""
        if self._three_layer_obs:
            return self._get_three_layer_observation_space()
        elif self._grid_observation:
            return self._get_grid_observation_space()
        else:
            return self._get_vector_observation_space()

    def _get_vector_observation_space(self):
        """获取向量观测空间 - 传统的坐标+属性表示"""
        field_x, field_y = self.field.shape[1], self.field.shape[0]
        max_food_level = self._get_max_food_level()
        
        # 食物观测：每个食物3个值(x, y, level)
        food_obs_len = 3 * self.max_num_food
        
        # 智能体观测：每个智能体2或3个值(x, y, [level])
        player_obs_len = (3 if self._observe_agent_levels else 2) * len(self.players)
        
        # 构建最小值和最大值
        min_obs = ([-1, -1, 0] * self.max_num_food + 
                  ([-1, -1, 0] if self._observe_agent_levels else [-1, -1]) * len(self.players))
        
        max_obs = ([field_x - 1, field_y - 1, max_food_level] * self.max_num_food + 
                  ([field_x - 1, field_y - 1, max(self.max_player_level)] 
                   if self._observe_agent_levels else [field_x - 1, field_y - 1]) * len(self.players))
        
        return gym.spaces.Box(
            low=np.array(min_obs, dtype=np.float32),
            high=np.array(max_obs, dtype=np.float32),
            dtype=np.float32
        )

    def _get_grid_observation_space(self):
        """获取网格观测空间 - 四层2D观测(智能体、食物、可访问性、自身标识)"""
        grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)
        max_food_level = self._get_max_food_level()
        max_agent_level = max(self.max_player_level) if self._observe_agent_levels else 1
        
        # 四层：智能体层、食物层、可访问层、自身标识层
        min_obs = np.stack([
            np.zeros(grid_shape, dtype=np.float32),  # 智能体层
            np.zeros(grid_shape, dtype=np.float32),  # 食物层
            np.zeros(grid_shape, dtype=np.float32),  # 可访问层
            np.zeros(grid_shape, dtype=np.float32),  # 自身标识层
        ])
        
        max_obs = np.stack([
            np.ones(grid_shape, dtype=np.float32) * max_agent_level,
            np.ones(grid_shape, dtype=np.float32) * max_food_level,
            np.ones(grid_shape, dtype=np.float32),
            np.ones(grid_shape, dtype=np.float32),
        ])
        
        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _get_three_layer_observation_space(self):
        """获取三层观测空间 - 5x5视野内的三层观测(自身、其他智能体、食物)"""
        grid_shape = (5, 5)
        max_food_level = self._get_max_food_level()
        max_agent_level = max(self.max_player_level) if self._observe_agent_levels else 1
        
        # 三层：自身智能体层、其他智能体层、食物层
        min_obs = np.stack([
            np.zeros(grid_shape, dtype=np.float32),  # 自身智能体层
            np.zeros(grid_shape, dtype=np.float32),  # 其他智能体层
            np.zeros(grid_shape, dtype=np.float32),  # 食物层
        ])
        
        max_obs = np.stack([
            np.ones(grid_shape, dtype=np.float32) * max_agent_level,
            np.ones(grid_shape, dtype=np.float32) * max_agent_level,
            np.ones(grid_shape, dtype=np.float32) * max_food_level,
        ])
        
        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _get_max_food_level(self):
        """获取最大食物等级"""
        if self.max_food_level is not None:
            return max(self.max_food_level)
        else:
            player_levels = sorted(self.max_player_level)
            return sum(player_levels[:3])

    def _make_gym_obs(self):
        """生成Gym格式的观测"""
        if self._three_layer_obs:
            return self._make_three_layer_observations()
        elif self._grid_observation:
            return self._make_grid_observations()
        else:
            return self._make_vector_observations()

    def _make_vector_observations(self):
        """生成向量观测"""
        observations = [self._make_obs(player) for player in self.players]
        nobs = tuple([self._observation_to_vector(obs) for obs in observations])
        self._validate_observations(nobs)
        return nobs

    def _make_grid_observations(self):
        """生成网格观测"""
        global_layers = self._create_global_grid_layers()
        nobs = []
        
        for player in self.players:
            agent_obs = self._extract_agent_grid_view(player, global_layers)
            nobs.append(agent_obs)
        
        nobs = tuple(nobs)
        self._validate_observations(nobs)
        return nobs

    def _make_three_layer_observations(self):
        """生成三层观测"""
        nobs = []
        
        for current_player in self.players:
            agent_obs = self._create_three_layer_view(current_player)
            nobs.append(agent_obs)
        
        nobs = tuple(nobs)
        self._validate_observations(nobs)
        return nobs

    def _observation_to_vector(self, observation):
        """将观测对象转换为向量格式"""
        obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
        
        # 优先显示自身智能体
        seen_players = ([p for p in observation.players if p.is_self] + 
                       [p for p in observation.players if not p.is_self])

        # 填充食物信息
        self._fill_food_vector(obs, observation.field)
        
        # 填充智能体信息
        self._fill_player_vector(obs, seen_players)
        
        return obs

    def _fill_food_vector(self, obs, field):
        """填充食物向量信息"""
        # 初始化所有食物位置为-1
        for i in range(self.max_num_food):
            obs[3 * i:3 * i + 3] = [-1, -1, 0]
        
        # 填充实际存在的食物
        for i, (y, x) in enumerate(zip(*np.nonzero(field))):
            if i < self.max_num_food:
                obs[3 * i:3 * i + 3] = [y, x, field[y, x]]

    def _fill_player_vector(self, obs, seen_players):
        """填充智能体向量信息"""
        player_obs_len = 3 if self._observe_agent_levels else 2
        start_idx = self.max_num_food * 3
        
        # 初始化所有智能体位置为-1
        for i in range(len(self.players)):
            base_idx = start_idx + player_obs_len * i
            obs[base_idx:base_idx + 2] = [-1, -1]
            if self._observe_agent_levels:
                obs[base_idx + 2] = 0
        
        # 填充实际可见的智能体
        for i, player in enumerate(seen_players):
            if i < len(self.players):
                base_idx = start_idx + player_obs_len * i
                obs[base_idx:base_idx + 2] = player.position
                if self._observe_agent_levels:
                    obs[base_idx + 2] = player.level

    def _create_global_grid_layers(self):
        """创建全局网格层"""
        grid_shape_x, grid_shape_y = self.field_size
        grid_shape_x += 2 * self.sight
        grid_shape_y += 2 * self.sight
        grid_shape = (grid_shape_x, grid_shape_y)

        # 智能体层
        agents_layer = self._create_agents_layer(grid_shape)
        
        # 食物层
        foods_layer = self._create_foods_layer(grid_shape)
        
        # 可访问层
        access_layer = self._create_access_layer(grid_shape)

        return np.stack([agents_layer, foods_layer, access_layer])

    def _create_agents_layer(self, grid_shape):
        """创建智能体层"""
        agents_layer = np.zeros(grid_shape, dtype=np.float32)
        for player in self.players:
            player_x, player_y = player.position
            value = player.level if self._observe_agent_levels else 1
            agents_layer[player_x + self.sight, player_y + self.sight] = value
        return agents_layer

    def _create_foods_layer(self, grid_shape):
        """创建食物层"""
        foods_layer = np.zeros(grid_shape, dtype=np.float32)
        foods_layer[self.sight:-self.sight, self.sight:-self.sight] = self.field.copy()
        return foods_layer

    def _create_access_layer(self, grid_shape):
        """创建可访问层"""
        access_layer = np.ones(grid_shape, dtype=np.float32)
        
        # 边界不可访问
        access_layer[:self.sight, :] = 0.0
        access_layer[-self.sight:, :] = 0.0
        access_layer[:, :self.sight] = 0.0
        access_layer[:, -self.sight:] = 0.0
        
        # 智能体位置不可访问
        for player in self.players:
            player_x, player_y = player.position
            access_layer[player_x + self.sight, player_y + self.sight] = 0.0
        
        # 食物位置不可访问
        foods_x, foods_y = self.field.nonzero()
        for x, y in zip(foods_x, foods_y):
            access_layer[x + self.sight, y + self.sight] = 0.0
        
        return access_layer

    def _extract_agent_grid_view(self, player, global_layers):
        """提取智能体的网格视野"""
        agent_x, agent_y = player.position
        start_x = agent_x
        end_x = agent_x + 2 * self.sight + 1
        start_y = agent_y
        end_y = agent_y + 2 * self.sight + 1
        
        # 获取智能体视野范围内的网格
        agent_view = global_layers[:, start_x:end_x, start_y:end_y].copy()
        
        # 创建自身识别层
        self_layer = np.zeros((end_x - start_x, end_y - start_y), dtype=np.float32)
        rel_agent_x = min(self.sight, agent_x)
        rel_agent_y = min(self.sight, agent_y)
        self_layer[rel_agent_x, rel_agent_y] = 1.0
        
        # 将自身识别层加入观测
        return np.concatenate([agent_view, self_layer[np.newaxis, :, :]], axis=0)

    def _create_three_layer_view(self, current_player):
        """为指定智能体创建三层视野观测"""
        grid_size = 5
        grid_shape = (grid_size, grid_size)
        
        # 初始化三个层
        self_layer = np.zeros(grid_shape, dtype=np.float32)
        other_agents_layer = np.zeros(grid_shape, dtype=np.float32)
        food_layer = np.zeros(grid_shape, dtype=np.float32)
        
        center_x, center_y = current_player.position
        
        # 遍历5x5视野区域
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                abs_x = center_x + dx
                abs_y = center_y + dy
                grid_x = dx + 2
                grid_y = dy + 2
                
                # 检查是否在场景内
                if 0 <= abs_x < self.rows and 0 <= abs_y < self.cols:
                    self._fill_three_layer_cell(
                        current_player, abs_x, abs_y, grid_x, grid_y,
                        self_layer, other_agents_layer, food_layer
                    )
        
        return np.stack([self_layer, other_agents_layer, food_layer])

    def _fill_three_layer_cell(self, current_player, abs_x, abs_y, grid_x, grid_y,
                              self_layer, other_agents_layer, food_layer):
        """填充三层观测中的单个格子"""
        # 检查是否为当前智能体
        if abs_x == current_player.position[0] and abs_y == current_player.position[1]:
            value = current_player.level if self._observe_agent_levels else 1
            self_layer[grid_x, grid_y] = value
        
        # 检查是否有其他智能体
        for player in self.players:
            if player != current_player and player.position == (abs_x, abs_y):
                value = player.level if self._observe_agent_levels else 1
                other_agents_layer[grid_x, grid_y] = value
        
        # 检查是否有食物
        if self.field[abs_x, abs_y] > 0:
            food_layer[grid_x, grid_y] = self.field[abs_x, abs_y]

    def _validate_observations(self, nobs):
        """验证观测是否符合观测空间"""
        for i, obs in enumerate(nobs):
            assert self.observation_space[i].contains(obs), \
                f"观测空间错误: obs: {obs.shape}, obs_space: {self.observation_space[i]}"

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def _distance_to_nearest_food(self, player_position):
        """
        计算智能体到最近食物的曼哈顿距离
        
        参数:
            player_position: 智能体位置 (row, col)
            
        返回:
            int: 到最近食物的曼哈顿距离，如果没有食物返回场地对角线长度
        """
        food_positions = np.argwhere(self.field > 0)
        
        if len(food_positions) == 0:
            # 如果没有食物，返回场地对角线长度作为最大距离
            return self.rows + self.cols
        
        # 计算到所有食物的曼哈顿距离
        distances = []
        for food_pos in food_positions:
            distance = abs(player_position[0] - food_pos[0]) + abs(player_position[1] - food_pos[1])
            distances.append(distance)
        
        return min(distances)

    def spawn_food(self, max_num_food, min_levels, max_levels):
        food_count = 0
        attempts = 0
        min_levels = max_levels if self.force_coop else min_levels

        # permute food levels
        food_permutation = self.np_random.permutation(max_num_food)
        min_levels = min_levels[food_permutation]
        max_levels = max_levels[food_permutation]

        while food_count < max_num_food and attempts < 1000:
            attempts += 1
            row = self.np_random.integers(1, self.rows - 1)
            col = self.np_random.integers(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = (
                min_levels[food_count]
                if min_levels[food_count] == max_levels[food_count]
                else self.np_random.integers(
                    min_levels[food_count], max_levels[food_count] + 1
                )
            )
            food_count += 1
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self, min_player_levels, max_player_levels):
        # permute player levels
        player_permutation = self.np_random.permutation(len(self.players))
        min_player_levels = min_player_levels[player_permutation]
        max_player_levels = max_player_levels[player_permutation]
        for player, min_player_level, max_player_level in zip(
            self.players, min_player_levels, max_player_levels
        ):
            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.integers(0, self.rows)
                col = self.np_random.integers(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.np_random.integers(min_player_level, max_player_level + 1),
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            # setting seed
            super().reset(seed=seed, options=options)

        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.min_player_level, self.max_player_level)
        player_levels = sorted([player.level for player in self.players])

        self.spawn_food(
            self.max_num_food,
            min_levels=self.min_food_level,
            max_levels=self.max_food_level
            if self.max_food_level is not None
            else np.array([sum(player_levels[:3])] * self.max_num_food),
        )
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        nobs = self._make_gym_obs()
        return nobs, self._get_info()

    def step(self, actions):

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players
        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])
            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                for a in adj_players:
                    a.reward -= self.penalty
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                a.reward = float(a.level * food)
                if self._normalize_reward:
                    a.reward = a.reward / float(
                        adj_player_level * self._food_spawned
                    )  # normalize reward
            # and the food is removed
            self.field[frow, fcol] = 0

        self.current_step += 2

        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        # 游戏结束时对不做归一化的奖励作调整
        if self._game_over and not self._normalize_reward:
            self._apply_final_reward_adjustments()

        for p in self.players:
            p.score += p.reward

        rewards = [p.reward for p in self.players]
        done = self._game_over
        truncated = False
        info = self._get_info()

        return self._make_gym_obs(), rewards, done, truncated, info

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=self.render_mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()

    def run(self, agents=None, is_training=False, render=False, sleep_time=0.5):
        """
        运行完整的回合，由agents控制行动选择
        
        参数:
            agents: 控制玩家的智能体列表，若为None则使用self.players中设置的控制器
            is_training: 是否处于训练模式
            render: 是否渲染游戏界面
            sleep_time: 渲染时每步之间的间隔时间(秒)
            
        返回:
            trajectories: 每个玩家的轨迹列表
            payoffs: 每个玩家的总奖励
        """
        if agents is not None:
            # 临时设置控制器
            for i, agent in enumerate(agents):
                if i < len(self.players):
                    self.players[i].set_controller(agent)
                    
        # 重置环境
        obss, _ = self.reset()
        done = False
        trajectories = [[] for _ in range(len(self.players))]
        payoffs = np.zeros(len(self.players))

        # 初始化动作缓冲区
        actions_buffs = [deque(maxlen=50) for _ in range(len(self.players))]  # 每个智能体的动作历史
        
        # 渲染初始状态
        if render:
            self.render()
            time.sleep(sleep_time)
        
        # 逐步执行，直到回合结束
        steps = 0
        while not done:
            steps += 2
            # 收集动作
            actions = []
            for i, player in enumerate(self.players):
                # 提取有效动作
                valid_actions = [action.value for action in self._valid_actions[player]]
                
                # 构建observation字典 - 支持网格观察和非网格观察模式
                obs_dict = {
                    'obs': obss[i], 'actions': valid_actions  # 使用实际有效的动作，不是所有动作
                }
                
                # 让智能体选择动作
                action = player.select_action(obs_dict, flag_train=is_training)

                
                # 检测重复动作模式
                if self._repeated_actions_detected(action, actions_buffs[i]):
                    # 如果检测到重复，选择一个不同的随机动作
                    other_valid_actions = \
                        [a for a in valid_actions if a != action]
                    if other_valid_actions:  # 确保有其他有效动作可选
                        action = np.random.choice(other_valid_actions)
                # 记录动作到缓冲区
                actions_buffs[i].append(action)
                actions.append(action)
            
            # 使用环境的step函数执行动作
            next_obss, rewards, done, _, _ = self.step(actions)
            
            # 更新累计奖励
            payoffs += rewards
            
            # 记录每个智能体的轨迹
            for i in range(len(self.players)):
                next_valid_actions = [action.value for action in
                                      self._valid_actions[self.players[i]]]
                
                # 轨迹格式：[obs_dict, action, reward, next_obs_dict, done]
                trajectory_segment = [
                    {'obs': obss[i], 'actions': valid_actions},  # 当前观察和有效动作
                    actions[i],
                    rewards[i],
                    {'obs': next_obss[i], 'actions': next_valid_actions},  # 下一观察和有效动作
                    done
                ]
                trajectories[i].append(trajectory_segment)

            for ts in trajectories[0]:  #ts为局中智能体每次行动的轨迹
                if len(ts) > 0:
                    # ts结构：[obs_dict, action, reward, next_obs_dict, done]
                    # 添加轨迹(s_t, a_t, r_t, s_t+1)
                    self.add_traj([ts[0], ts[1], ts[2],
                                    ts[0] if ts[4] else ts[3], ts[4]])
            # 更新观察
            obss = next_obss
            
            if render:
                self.render()
                time.sleep(sleep_time)
        
        return trajectories, payoffs, self.current_step

    def test_make_gym_obs(self):
        """Test wrapper to test the current observation in a public manner."""
        return self._make_gym_obs()

    def test_gen_valid_moves(self):
        """Wrapper around a private method to test if the generated moves are valid."""
        try:
            self._gen_valid_moves()
        except Exception as _:
            return False
        return True
    def _repeated_actions_detected(self, action, actions_buff):
        """
        检测动作是否出现重复，防止智能体陷入动作循环
        
        参数:
            action: 当前选择的动作
            actions_buff: 最近收集的动作序列
            
        返回:
            bool: 如果检测到重复动作模式返回True，否则返回False
        """
        # 如果动作缓冲区为空，不可能有重复
        if not actions_buff:
            return False
        
        # 环境状态分析：计算当前环境中食物数量和智能体数量    
        food_count = np.count_nonzero(self.field)
        agent_count = len(self.players)
        
        # 调整策略：根据环境中的食物和智能体数量动态调整
        
        # 单目标情况下使用较严格的检测
        if food_count <= 1:
            # 检测循环模式：如果缓冲区足够长，检查是否有2步或3步的循环
            if len(actions_buff) >= 3 and action != 5:
                # 超过3次连续相同动作视为重复
                if action == actions_buff[-1] == actions_buff[-2] == actions_buff[-3]:
                    return True
                    
                if len(actions_buff) >= 4:
                    # 检测2步循环 (A-B-A-B模式)
                    if action == actions_buff[-2] and actions_buff[-1] == actions_buff[-3]:
                        return True
                    
                    # 检测3步循环 (A-B-C-A-B-C模式)
                    if len(actions_buff) >= 6:
                        if (action == actions_buff[-3] and 
                            actions_buff[-1] == actions_buff[-4] and 
                            actions_buff[-2] == actions_buff[-5]):
                            return True
        # 多目标多智能体情况下，仅检测长时间重复的加载动作
        else:
            # 检查是否为加载动作(Action.LOAD.value = 5)
            if action == 5:
                # 只有当连续加载次数超过阈值时才判定为重复
                load_threshold = min(5, food_count + 1)  # 根据食物数量调整阈值
                
                # 统计连续加载次数
                consecutive_loads = 0
                for a in reversed(actions_buff):
                    if a == 5:
                        consecutive_loads += 1
                    else:
                        break
                        
                # 连续加载次数超过阈值返回True
                if consecutive_loads >= load_threshold:
                    # 检查是否有其他智能体在当前智能体附近
                    # 由于无法直接获取当前智能体的位置，我们通过其他方式检测周围的智能体
                    
                    # 寻找执行LOAD动作的食物位置
                    loaded_food_locations = []
                    for row in range(self.rows):
                        for col in range(self.cols):
                            if self.field[row, col] > 0 and self.adjacent_players(row, col):
                                loaded_food_locations.append((row, col))
                    
                    # 如果有正在被多个智能体尝试加载的食物，允许继续等待
                    for loc in loaded_food_locations:
                        if len(self.adjacent_players(*loc)) > 1:
                            return False
                    
                    return True
            # 对于非加载动作，只检测极长的重复序列
            elif len(actions_buff) >= 8:
                # 检测8次完全相同的动作
                if all(a == action for a in list(actions_buff)[-7:]):
                    return True
                    
        return False

    def _apply_final_reward_adjustments(self):
        """
        在游戏结束时应用最终的奖励调整
        
        包含两种调整机制：
        1. 胜利时（所有食物被收集）：根据步数给予奖励加成，步数越少奖励越高
        2. 失败时（超时）：根据智能体到最近食物的距离给予惩罚，距离越远惩罚越大
        """
        # 判断游戏结束的原因
        is_victory = self.field.sum() == 0  # 所有食物被收集完毕
        is_timeout = (self.current_step >= self._max_episode_steps)
        
        if is_victory:
            # 胜利情况：根据步数给予奖励加成
            self._apply_step_reward_bonus()
        elif is_timeout:
            # 失败情况：根据到食物距离给予惩罚
            self._apply_distance_penalty()
    
    def _apply_step_reward_bonus(self):
        """
        胜利时根据步数给予奖励加成
        步数越少，奖励加成越高
        """
        # 计算步数效率系数：(最大步数 - 当前步数) / 最大步数
        # 这样步数越少，系数越大
        step_efficiency = max(0, (self._max_episode_steps - self.current_step) / self._max_episode_steps)
        
        player = self.players[0]    #仅为主智能体添加奖惩
        # 只对有正奖励的玩家给予步数加成
        if player.reward > 0:
            step_bonus = player.reward * self.step_reward_factor * step_efficiency
            player.reward += step_bonus

            self.logger.debug(f"玩家 {player.name} 获得步数奖励加成: {step_bonus:.4f} "
                            f"(效率系数: {step_efficiency:.4f}, 当前步数: {self.current_step})")
    
    def _apply_distance_penalty(self):
        """
        失败时根据智能体到最近食物的距离给予惩罚
        距离越远，惩罚越大
        """
        # 计算场地的最大可能距离（对角线距离）
        max_distance = self.rows + self.cols - 2
        
        player = self.players[0]    #仅为主智能体添加奖惩
        # 计算到最近食物的距离
        distance_to_food = self._distance_to_nearest_food(player.position)

        # 计算距离惩罚：距离越远惩罚越大
        # 使用归一化的距离比例
        distance_ratio = min(1.0, distance_to_food / max_distance)
        distance_penalty = self.distance_penalty_factor * distance_ratio
            
        # 应用惩罚（减少奖励）
        player.reward -= distance_penalty

        self.logger.debug(f"玩家 {player.name} 获得距离惩罚: -{distance_penalty:.4f} "
                        f"(距离: {distance_to_food}, 距离比例: {distance_ratio:.4f})")

    def valid_actions(self, player_id):
        return self._valid_actions[self.players[player_id]]