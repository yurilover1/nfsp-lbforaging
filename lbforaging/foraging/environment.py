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

    def step(self, obs):
        if hasattr(self.controller, '_step'):
            return self.controller._step(obs)
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
        step_reward_factor=0.5,  # 步数奖励因子
        step_reward_threshold=0.1,  # 步数奖励阈值，表示最大步数的比例
    ):
        self.logger = logging.getLogger(__name__)
        self.render_mode = render_mode
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)

        self.penalty = penalty
        
        # 步数奖励相关参数
        self.step_reward_factor = step_reward_factor
        self.step_reward_threshold = step_reward_threshold

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
        self._grid_observation = grid_observation
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
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        player_levels = sorted(self.max_player_level)
        max_food_level = (
            max(self.max_food_level)
            if self.max_food_level is not None
            else sum(player_levels[:3])
        )
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_num_food = self.max_num_food

            if self._observe_agent_levels:
                min_obs = [-1, -1, 0] * max_num_food + [-1, -1, 0] * len(self.players)
                max_obs = [field_x - 1, field_y - 1, max_food_level] * max_num_food + [
                    field_x - 1,
                    field_y - 1,
                    max(self.max_player_level),
                ] * len(self.players)
            else:
                min_obs = [-1, -1, 0] * max_num_food + [-1, -1] * len(self.players)
                max_obs = [field_x - 1, field_y - 1, max_food_level] * max_num_food + [
                    field_x - 1,
                    field_y - 1,
                ] * len(self.players)
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            if self._observe_agent_levels:
                agents_max = np.ones(grid_shape, dtype=np.float32) * max(
                    self.max_player_level
                )
            else:
                agents_max = np.ones(grid_shape, dtype=np.float32)

            # foods layer: foods level
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min])
            max_obs = np.stack([agents_max, foods_max, access_max])

        low_obs = np.array(min_obs)
        high_obs = np.array(max_obs)
        assert low_obs.shape == high_obs.shape
        return gym.spaces.Box(
            low=low_obs, high=high_obs, shape=[len(low_obs)], dtype=np.float32
        )

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(
            players,
            min_player_level=1,
            max_player_level=2,
            min_food_level=1,
            max_food_level=None,
            field_size=None,
            max_num_food=None,
            sight=None,
            max_episode_steps=50,
            force_coop=False,
        )

        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

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

    def _make_gym_obs(self):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.max_num_food):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            player_obs_len = 3 if self._observe_agent_levels else 2
            for i in range(len(self.players)):
                obs[self.max_num_food * 3 + player_obs_len * i] = -1
                obs[self.max_num_food * 3 + player_obs_len * i + 1] = -1
                if self._observe_agent_levels:
                    obs[self.max_num_food * 3 + player_obs_len * i + 2] = 0

            for i, p in enumerate(seen_players):
                obs[self.max_num_food * 3 + player_obs_len * i] = p.position[0]
                obs[self.max_num_food * 3 + player_obs_len * i + 1] = p.position[1]
                if self._observe_agent_levels:
                    obs[self.max_num_food * 3 + player_obs_len * i + 2] = p.level

            return obs

        def make_global_grid_arrays():
            """
            Create global arrays for grid observation space
            """
            grid_shape_x, grid_shape_y = self.field_size
            grid_shape_x += 2 * self.sight
            grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)

            agents_layer = np.zeros(grid_shape, dtype=np.float32)
            for player in self.players:
                player_x, player_y = player.position
                if self._observe_agent_levels:
                    agents_layer[player_x + self.sight, player_y + self.sight] = (
                        player.level
                    )
                else:
                    agents_layer[player_x + self.sight, player_y + self.sight] = 1

            foods_layer = np.zeros(grid_shape, dtype=np.float32)
            foods_layer[self.sight : -self.sight, self.sight : -self.sight] = (
                self.field.copy()
            )

            access_layer = np.ones(grid_shape, dtype=np.float32)
            # out of bounds not accessible
            access_layer[: self.sight, :] = 0.0
            access_layer[-self.sight :, :] = 0.0
            access_layer[:, : self.sight] = 0.0
            access_layer[:, -self.sight :] = 0.0
            # agent locations are not accessible
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self.sight, player_y + self.sight] = 0.0
            # food locations are not accessible
            foods_x, foods_y = self.field.nonzero()
            for x, y in zip(foods_x, foods_y):
                access_layer[x + self.sight, y + self.sight] = 0.0

            return np.stack([agents_layer, foods_layer, access_layer])

        def get_agent_grid_bounds(agent_x, agent_y):
            return (
                agent_x,
                agent_x + 2 * self.sight + 1,
                agent_y,
                agent_y + 2 * self.sight + 1,
            )

        observations = [self._make_obs(player) for player in self.players]
        if self._grid_observation:
            layers = make_global_grid_arrays()
            agents_bounds = [
                get_agent_grid_bounds(*player.position) for player in self.players
            ]
            nobs = tuple(
                [
                    layers[:, start_x:end_x, start_y:end_y]
                    for start_x, end_x, start_y, end_y in agents_bounds
                ]
            )
        else:
            nobs = tuple([make_obs_array(obs) for obs in observations])

        # check the space of obs
        for i, obs in enumerate(nobs):
            assert self.observation_space[i].contains(
                obs
            ), f"obs space error: obs: {obs}, obs_space: {self.observation_space[i]}"

        return nobs

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
        self.current_step += 1

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

        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

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

        # 初始化轨迹历史队列和动作缓冲区
        actions_buffs = [deque(maxlen=50) for _ in range(len(self.players))]  # 每个智能体的动作历史
        
        # 渲染初始状态
        if render:
            self.render()
            time.sleep(sleep_time)
        
        # 逐步执行，直到回合结束
        while not done:
            # 收集动作
            actions = []
            for i, player in enumerate(self.players):
                # 提取有效动作
                valid_actions = [action.value for action in self._valid_actions[player]]
                
                # 构建observation字典
                obs_dict = {
                    'obs': obss[i],
                    'actions': valid_actions  # 使用实际有效的动作，不是所有动作
                }
                
                # 让智能体选择动作
                if player.controller:
                    action = player.step(obs_dict)
                else:
                    # 如果没有控制器，随机选择动作
                    action = np.random.choice(valid_actions)
                # 检测重复动作模式
                if self._repeated_actions_detected(action, actions_buffs[i]):
                    # 如果检测到重复，选择一个不同的随机动作
                    other_valid_actions = [a for a in valid_actions if a != action]
                    if other_valid_actions:  # 确保有其他有效动作可选
                        action = np.random.choice(other_valid_actions)
                # if len(actions_buffs[i]) >= 3 and actions_buffs[i][-1] == 5 and self.adjacent_food(*player.position):
                #     action = 5
                # 记录动作到缓冲区
                actions_buffs[i].append(action)
                actions.append(action)
            
            # 使用环境的step函数执行动作
            next_obss, rewards, done, _, _ = self.step(actions)
            
            # 更新累计奖励
            payoffs += rewards
            
            # 记录每个智能体的轨迹
            for i in range(len(self.players)):
                next_valid_actions = [action.value for action in self._valid_actions[self.players[i]]]
                
                # 轨迹格式：[obs_dict, action, reward, next_obs_dict, done]
                trajectory_segment = [
                    {'obs': obss[i], 'actions': valid_actions},  # 当前观察和有效动作
                    actions[i],
                    rewards[i],
                    {'obs': next_obss[i], 'actions': next_valid_actions},  # 下一观察和有效动作
                    done
                ]
                trajectories[i].append(trajectory_segment)


            # 更新观察
            obss = next_obss
            
            if render:
                self.render()
                time.sleep(sleep_time)
        
        return trajectories, payoffs

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
        