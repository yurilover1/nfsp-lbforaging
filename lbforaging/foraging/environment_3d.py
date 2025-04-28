# 3D LB Foraging Environment

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import namedtuple, defaultdict, deque
import random
import time
import logging


class Player:
    def __init__(self, player_id, position, level, history_len=None, field_size=(6, 6, 6)):
        self.id = player_id
        self.position = position  # (x, y, z)
        self.level = level
        self._field_size = field_size
        self.history_len = history_len
        if history_len is not None:
            self.history = []
        self.controller = None
        self.score = 0
        self.reward = 0

    @property
    def field_size(self):
        """返回场地大小"""
        return self._field_size

    @field_size.setter
    def field_size(self, value):
        """设置场地大小"""
        self._field_size = value

    def step(self, action):
        # 0 - STAY, 1 - UP, 2 - DOWN, 3 - LEFT, 4 - RIGHT, 5 - FORWARD, 6 - BACKWARD
        # Actions are now in 3D
        x, y, z = self.position

        if action == 1:  # UP
            y = min(self._field_size[1] - 1, y + 1)
        elif action == 2:  # DOWN
            y = max(0, y - 1)
        elif action == 3:  # LEFT
            x = max(0, x - 1)
        elif action == 4:  # RIGHT
            x = min(self._field_size[0] - 1, x + 1)
        elif action == 5:  # FORWARD (Z+)
            z = min(self._field_size[2] - 1, z + 1)
        elif action == 6:  # BACKWARD (Z-)
            z = max(0, z - 1)

        self.position = (x, y, z)

        if self.history_len is not None:
            self.history.append(action)
            if len(self.history) > self.history_len:
                self.history.pop(0)

    def get_state(self):
        return {
            "position": self.position,
            "level": self.level,
            "history": self.history if self.history_len is not None else None
        }

    def set_controller(self, controller):
        """设置玩家的控制器"""
        self.controller = controller

    @property
    def name(self):
        """获取玩家名称"""
        if self.controller:
            return self.controller.name
        else:
            return f"Player_{self.id}"


class Food:
    def __init__(self, position, level):
        self.position = position  # (x, y, z)
        self.level = level
        self.collected = False


class ForagingEnv3D(gym.Env):
    """
    A 3D version of the Level-Based Foraging Environment.

    Environment Dynamics:
    - A 3D grid world of size n_rows x n_cols x n_depth
    - Players and food items are randomly placed on the grid
    - Players have a level
    - Food items have a level
    - Players can move in 6 directions (up, down, left, right, forward, backward)
    - A group of players can collect a food item if their combined level is greater than or equal to the food's level
    - Players receive a reward equal to their level divided by the sum of player levels that collected the food item
    """

    metadata = {'render.modes': ['human', '3d']}
    action_set = {0: 'STAY', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'FORWARD', 6: 'BACKWARD'}
    Observation = namedtuple('Observation', ['field', 'actions', 'players', 'game_over', 'sight', 'current_step'])
    PlayerObservation = namedtuple('PlayerObservation', ['position', 'level', 'history', 'is_self'])

    def __init__(self, n_rows=6, n_cols=6, n_depth=6, num_agents=2, num_food=1, max_episode_steps=100,
                 sight=None, max_player_level=3, max_food_level=3, force_coop=False, grid_observation=False,
                 penalty=0.0, history_len=None, step_reward_factor=0.5, step_reward_threshold=0.1,
                 min_player_level=1, min_food_level=None, food_reward_scale=0.0, proximity_factor=0.02, 
                 enable_proximity_reward=True, enable_step_reward=True):
        self.logger = logging.getLogger(__name__)

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_depth = n_depth
        self.num_agents = num_agents
        self.num_food = num_food
        self.sight = sight
        self.max_player_level = max_player_level
        self.min_player_level = min_player_level
        self.food_reward_scale = food_reward_scale  # 保存食物奖励比例
        self.proximity_factor = proximity_factor  # 接近奖励因子（越接近食物奖励越大）
        self.enable_proximity_reward = enable_proximity_reward  # 是否启用接近奖励
        self.enable_step_reward = enable_step_reward  # 是否启用步进奖励

        # 如果没有提供min_food_level，那么设置为1
        if min_food_level is None:
            self.min_food_level = np.array([1] * num_food)
        elif isinstance(min_food_level, (int, float)):
            self.min_food_level = np.array([min_food_level] * num_food)
        else:
            self.min_food_level = np.array(min_food_level)

        if max_food_level is None:
            self.max_food_level = None
        elif isinstance(max_food_level, (int, float)):
            self.max_food_level = np.array([max_food_level] * num_food)
        else:
            self.max_food_level = np.array(max_food_level)

        self.penalty = penalty
        self.grid_observation = grid_observation
        self.history_len = history_len
        self.step_reward_factor = step_reward_factor
        self.step_reward_threshold = step_reward_threshold

        self._max_episode_steps = max_episode_steps
        self._current_steps = 0

        self._field_size = (n_rows, n_cols, n_depth)
        self.force_coop = force_coop
        self._normalize_reward = True  # 是否归一化奖励
        self._game_over = False
        self._np_random = None
        self.current_step = 0

        self.action_space = spaces.Tuple([spaces.Discrete(7) for _ in range(self.num_agents)])

        self.players = []
        self.food_items = []
        self._food_spawned = 0.0
        self._valid_actions = None

        if not grid_observation:
            # Define spaces for player and food observations
            players_obs_space = spaces.Box(
                low=np.array([0, 0, 0, 1, 0, 0] * self.num_agents),
                high=np.array([n_rows - 1, n_cols - 1, n_depth - 1, max_player_level, 1, 1] * self.num_agents),
                dtype=np.float32
            )

            food_obs_space = spaces.Box(
                low=np.array([0, 0, 0, 1, 0] * self.num_food),
                high=np.array([n_rows - 1, n_cols - 1, n_depth - 1, max_food_level, 1] * self.num_food),
                dtype=np.float32
            )

            self.observation_space = spaces.Tuple([players_obs_space, food_obs_space])
        else:
            # 3D grid observation
            # Channel 0: Binary matrix denoting player positions
            # Channel 1: Matrix denoting player levels
            # Channel 2: Binary matrix denoting food positions
            # Channel 3: Matrix denoting food levels
            if isinstance(max_food_level, (int, float)):
                max_level = max(max_player_level, max_food_level)
            else:
                # 如果max_food_level是数组，取其最大值
                max_level = max(max_player_level,
                                np.max(max_food_level) if max_food_level is not None else max_player_level)

            # 现在使用标量进行比较
            self.observation_space = spaces.Box(
                low=0,
                high=max_level,
                shape=(4, n_rows, n_cols, n_depth),
                dtype=np.float32
            )

        self.reset()

    def seed(self, seed=None):
        """设置随机种子"""
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
        else:
            self._np_random = np.random.RandomState()
        return [seed]

    @property
    def np_random(self):
        """获取随机数生成器"""
        if self._np_random is None:
            self._np_random = np.random.RandomState()
        return self._np_random

    def reset(self, seed=None, options=None):
        """重置环境并返回初始观测"""
        if seed is not None:
            self.seed(seed)

        self._current_steps = 0
        self.current_step = 0
        self.players = []
        self.food_items = []
        self._game_over = False

        # Place players randomly
        player_positions = self._get_empty_positions(self.num_agents)
        for i, pos in enumerate(player_positions):
            level = self.np_random.randint(self.min_player_level, self.max_player_level + 1)
            player = Player(i, pos, level, self.history_len, self.field_size)
            player.score = 0
            player.reward = 0
            self.players.append(player)

        # Place food randomly
        food_positions = self._get_empty_positions(self.num_food)
        for i, pos in enumerate(food_positions):
            if self.force_coop:
                # 如果启用强制合作,食物等级需要大于单个智能体的等级
                min_level = max(self.min_food_level[i], max(player.level for player in self.players) + 1)
                if isinstance(self.max_food_level, np.ndarray):
                    max_level = self.max_food_level[i]
                else:
                    max_level = self.max_food_level
                level = self.np_random.randint(min_level, max_level + 1)
            else:
                # 否则使用正常的食物等级范围
                if isinstance(self.max_food_level, np.ndarray):
                    max_level = self.max_food_level[i]
                else:
                    max_level = self.max_food_level
                level = self.np_random.randint(self.min_food_level[i], max_level + 1)

            self.food_items.append(Food(pos, level))

        # 计算总食物等级
        self._food_spawned = sum(food.level for food in self.food_items)

        # 生成有效移动
        self._gen_valid_moves()

        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info

    def _get_empty_positions(self, num_positions):
        """获取空位置"""
        positions = []
        empty_positions = [(x, y, z) for x in range(self.n_rows) for y in range(self.n_cols) for z in
                           range(self.n_depth)]

        # Remove positions that are already occupied
        for player in self.players:
            if player.position in empty_positions:
                empty_positions.remove(player.position)

        for food in self.food_items:
            if food.position in empty_positions and not food.collected:
                empty_positions.remove(food.position)

        # Sample positions without replacement
        sampled_positions = random.sample(empty_positions, num_positions)
        positions.extend(sampled_positions)

        return positions

    def step(self, actions):
        """执行环境步进"""
        self._current_steps += 1
        self.current_step += 1

        # 初始化玩家奖励
        for p in self.players:
            p.reward = 0

        # Execute player actions
        for i, action in enumerate(actions):
            self.players[i].step(action)

        # Check for food collection
        rewards = np.zeros(self.num_agents)
        food_collection = defaultdict(list)

        # Group players by position to check for food collection
        for player in self.players:
            for food_idx, food in enumerate(self.food_items):
                if not food.collected and player.position == food.position:
                    food_collection[food_idx].append(player)

        # Check if collected food and distribute rewards
        for food_idx, players_list in food_collection.items():
            food = self.food_items[food_idx]
            combined_level = sum(player.level for player in players_list)

            if combined_level >= food.level:
                food.collected = True
                # Distribute rewards based on player levels
                for player in players_list:
                    # 食物收集奖励固定为1.0
                    player.reward = self.food_reward_scale
                    player.score += player.reward
                    rewards[player.id] = player.reward

        # 记录上一步每个智能体与食物的距离，用于计算是否接近了食物
        player_prev_distances = []
        
        # 添加基于接近食物的额外奖励
        for i, player in enumerate(self.players):
            if rewards[i] == 0:  # 如果该玩家没有获得食物收集奖励
                # 找到距离最近的食物
                min_distance = float('inf')
                nearest_food = None
                
                for food in self.food_items:
                    if not food.collected:
                        # 计算曼哈顿距离
                        distance = abs(player.position[0] - food.position[0]) + \
                                  abs(player.position[1] - food.position[1]) + \
                                  abs(player.position[2] - food.position[2])
                        
                        if distance < min_distance:
                            min_distance = distance
                            nearest_food = food
                
                # 如果找到了食物，计算接近奖励
                if min_distance < float('inf') and self.enable_proximity_reward:
                    # 接近奖励随距离减小而增加，最大值不超过食物收集奖励的30%
                    max_proximity_reward = self.food_reward_scale * 0.1
                    
                    # 使用反比例函数，但在距离为0时有上限
                    # 大幅降低接近奖励
                    proximity_reward = min(
                        max_proximity_reward,
                        self.proximity_factor * 0.1 * (1.0 / (min_distance + 1))
                    )
                    
                    rewards[i] += proximity_reward
                    player.reward += proximity_reward
                    player.score += proximity_reward
                
                # 添加小的步进奖励，即使没有接近食物（降低至几乎为0）
                if self.enable_step_reward:
                    step_reward = 0.001
                    rewards[i] += step_reward
                    player.reward += step_reward
                    player.score += step_reward

        # Apply penalty for unsuccessful actions (if penalty is set)
        if self.penalty > 0:
            for i, action in enumerate(actions):
                if action != 0 and rewards[i] == 0:  # If the agent moved but got no reward
                    rewards[i] -= self.penalty
                    self.players[i].reward -= self.penalty
                    self.players[i].score -= self.penalty

        # Check if game is over (all food collected or max steps reached)
        self._game_over = (
            all(food.collected for food in self.food_items) or
            self._current_steps >= self._max_episode_steps
        )

        # 生成有效移动
        self._gen_valid_moves()

        observation = self._get_observation()
        info = self._get_info()
        truncated = False

        return observation, rewards, self._game_over, truncated, info

    def _get_observation(self):
        """生成观测"""
        if not self.grid_observation:
            # Vector representation
            player_obs = []
            for player in self.players:
                x, y, z = player.position
                normalized_pos = [x, y, z]
                normalized_level = player.level
                history = player.history if self.history_len is not None else []

                player_info = normalized_pos + [normalized_level] + history
                player_obs.extend(player_info)

            food_obs = []
            for food in self.food_items:
                if not food.collected:
                    x, y, z = food.position
                    normalized_pos = [x, y, z]
                    normalized_level = food.level

                    food_info = normalized_pos + [normalized_level, 1]  # 1 indicates not collected
                    food_obs.extend(food_info)
                else:
                    # If food is collected, use zeros
                    food_obs.extend([0, 0, 0, 0, 0])

            return (np.array(player_obs, dtype=np.float32), np.array(food_obs, dtype=np.float32))
        else:
            # Grid representation
            obs = np.zeros((4, self.n_rows, self.n_cols, self.n_depth), dtype=np.float32)

            # Channel 0 & 1: Players
            for player in self.players:
                x, y, z = player.position
                obs[0, x, y, z] = 1  # Binary player position
                obs[1, x, y, z] = player.level  # Player level

            # Channel 2 & 3: Food
            for food in self.food_items:
                if not food.collected:
                    x, y, z = food.position
                    obs[2, x, y, z] = 1  # Binary food position
                    obs[3, x, y, z] = food.level  # Food level

            return obs

    def get_agent_obs(self):
        """返回单个智能体的观测"""
        # Returns a list of individual agent observations
        observations = []

        for agent_id in range(self.num_agents):
            if not self.grid_observation:
                # Vector observation for individual agent
                agent_obs = []

                # Information about self first
                for i, player in enumerate(self.players):
                    if i == agent_id:
                        x, y, z = player.position
                        agent_obs.extend([x, y, z, player.level, 1, 1])  # Last two: is_self, active

                # Information about other players
                for i, player in enumerate(self.players):
                    if i != agent_id:
                        x, y, z = player.position
                        agent_obs.extend([x, y, z, player.level, 0, 1])  # Not self, active

                # Information about food
                for food in self.food_items:
                    if not food.collected:
                        x, y, z = food.position
                        agent_obs.extend([x, y, z, food.level, 1])  # 1 indicates not collected
                    else:
                        agent_obs.extend([0, 0, 0, 0, 0])

                observations.append(np.array(agent_obs, dtype=np.float32))
            else:
                # Grid observation for individual agent
                agent_grid = np.zeros((4, self.n_rows, self.n_cols, self.n_depth), dtype=np.float32)

                # Channel 0: Self position
                x, y, z = self.players[agent_id].position
                agent_grid[0, x, y, z] = 1

                # Channel 1: Other players
                for i, player in enumerate(self.players):
                    if i != agent_id:
                        x, y, z = player.position
                        agent_grid[1, x, y, z] = 1  # Binary position
                        agent_grid[2, x, y, z] = player.level  # Level

                # Channel 3: Food
                for food in self.food_items:
                    if not food.collected:
                        x, y, z = food.position
                        agent_grid[3, x, y, z] = food.level

                observations.append(agent_grid)

        return observations

    def render(self, mode='human'):
        """渲染当前环境状态"""
        if mode == '3d':
            # 如果没有3D渲染器，则创建一个
            if not hasattr(self, 'viewer3d') or self.viewer3d is None:
                from lbforaging.foraging.rendering3d import Viewer3D
                self.viewer3d = Viewer3D(world_size=(self.n_rows, self.n_cols, self.n_depth))
            
            return self.viewer3d.render(self)
        else:
            # 使用原始的文本渲染方式
            for z in range(self.n_depth):
                print(f"Level Z={z}:")
                grid = np.zeros((self.n_rows, self.n_cols), dtype=object)
                grid[:] = ' '

                # Mark the food
                for food in self.food_items:
                    if not food.collected and food.position[2] == z:
                        x, y, _ = food.position
                        grid[x, y] = f'F{food.level}'

                # Mark the players
                for player in self.players:
                    if player.position[2] == z:
                        x, y, _ = player.position
                        if grid[x, y] == ' ':
                            grid[x, y] = f'P{player.id}'
                        else:
                            grid[x, y] = grid[x, y] + f',P{player.id}'

                # Print the grid
                horizontal_line = '-' * (self.n_cols * 4 + 1)
                print(horizontal_line)
                for row in range(self.n_rows):
                    row_str = '|'
                    for col in range(self.n_cols):
                        cell = grid[row, col]
                        row_str += f' {cell:2} |'
                    print(row_str)
                    print(horizontal_line)
                print()

    def close(self):
        """关闭环境"""
        if hasattr(self, 'viewer3d') and self.viewer3d is not None:
            self.viewer3d.close()
            self.viewer3d = None

    def _gen_valid_moves(self):
        """生成每个玩家的有效动作"""
        self._valid_actions = {}
        for player in self.players:
            valid_actions = []
            # 0 - STAY always valid
            valid_actions.append(0)

            x, y, z = player.position

            # 检查每个方向是否有效
            # 1 - UP
            if y < self.n_cols - 1 and self._is_empty_location(x, y + 1, z):
                valid_actions.append(1)

            # 2 - DOWN
            if y > 0 and self._is_empty_location(x, y - 1, z):
                valid_actions.append(2)

            # 3 - LEFT
            if x > 0 and self._is_empty_location(x - 1, y, z):
                valid_actions.append(3)

            # 4 - RIGHT
            if x < self.n_rows - 1 and self._is_empty_location(x + 1, y, z):
                valid_actions.append(4)

            # 5 - FORWARD (Z+)
            if z < self.n_depth - 1 and self._is_empty_location(x, y, z + 1):
                valid_actions.append(5)

            # 6 - BACKWARD (Z-)
            if z > 0 and self._is_empty_location(x, y, z - 1):
                valid_actions.append(6)

            # 检查是否有可收集的食物
            if self.adjacent_food(*player.position) > 0:
                # 在3D环境中我们不使用LOAD动作，而是通过玩家位置来判断食物收集
                pass

            self._valid_actions[player] = valid_actions

    def _is_empty_location(self, x, y, z):
        """检查位置是否为空"""
        # 检查是否有食物
        for food in self.food_items:
            if not food.collected and food.position == (x, y, z):
                return False

        # 检查是否有玩家
        for player in self.players:
            if player.position == (x, y, z):
                return False

        return True

    def adjacent_food(self, x, y, z):
        """检查相邻位置是否有食物"""
        food_count = 0

        # 检查6个相邻位置
        directions = [
            (x + 1, y, z), (x - 1, y, z),  # 左右
            (x, y + 1, z), (x, y - 1, z),  # 上下
            (x, y, z + 1), (x, y, z - 1)  # 前后
        ]

        for pos in directions:
            if 0 <= pos[0] < self.n_rows and 0 <= pos[1] < self.n_cols and 0 <= pos[2] < self.n_depth:
                for food in self.food_items:
                    if not food.collected and food.position == pos:
                        food_count += 1

        return food_count

    def adjacent_food_location(self, x, y, z):
        """获取相邻的食物位置"""
        # 检查6个相邻位置
        directions = [
            (x + 1, y, z), (x - 1, y, z),  # 左右
            (x, y + 1, z), (x, y - 1, z),  # 上下
            (x, y, z + 1), (x, y, z - 1)  # 前后
        ]

        for pos in directions:
            if 0 <= pos[0] < self.n_rows and 0 <= pos[1] < self.n_cols and 0 <= pos[2] < self.n_depth:
                for food in self.food_items:
                    if not food.collected and food.position == pos:
                        return pos

        return None

    def adjacent_players(self, x, y, z):
        """获取相邻的玩家"""
        adj_players = []

        # 检查6个相邻位置
        directions = [
            (x + 1, y, z), (x - 1, y, z),  # 左右
            (x, y + 1, z), (x, y - 1, z),  # 上下
            (x, y, z + 1), (x, y, z - 1)  # 前后
        ]

        for pos in directions:
            for player in self.players:
                if player.position == pos:
                    adj_players.append(player)

        return adj_players

    def _get_info(self):
        """返回环境信息"""
        return {
            "food_collected": sum(1 for food in self.food_items if food.collected),
            "total_food": len(self.food_items),
            "player_scores": [player.score for player in self.players]
        }

    def run(self, agents=None, is_training=False, render=False, sleep_time=0.5, render_mode='human'):
        """
        运行完整的回合，由agents控制行动选择

        参数:
            agents: 控制玩家的智能体列表，若为None则使用self.players中设置的控制器
            is_training: 是否处于训练模式
            render: 是否渲染游戏界面
            sleep_time: 渲染时每步之间的间隔时间(秒)
            render_mode: 渲染模式，'human'为文本渲染，'3d'为3D渲染

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
        obss = self.reset()
        # 确保观察是一个列表，每个智能体一个观察
        if not isinstance(obss, list):
            obss = self.get_agent_obs()
            
        done = False
        trajectories = [[] for _ in range(len(self.players))]
        payoffs = np.zeros(len(self.players))

        # 初始化轨迹历史队列和动作缓冲区
        actions_buffs = [deque(maxlen=200) for _ in range(len(self.players))]  # 每个智能体的动作历史

        # 渲染初始状态
        if render:
            self.render(mode=render_mode)
            time.sleep(sleep_time)

        # 逐步执行，直到回合结束
        while not done:
            # 收集动作
            actions = []
            for i, player in enumerate(self.players):
                # 提取有效动作
                valid_actions = self._valid_actions[player] if player in self._valid_actions else list(range(7))

                # 构建observation字典
                obs_dict = {
                    'obs': obss[i] if isinstance(obss, list) else obss,
                    'actions': valid_actions  # 使用实际有效的动作，不是所有动作
                }

                # 让智能体选择动作
                if player.controller:
                    try:
                        action = player.controller.step(obs_dict)
                    except Exception as e:
                        print(f"智能体选择动作时出错: {e}")
                        # 出错时随机选择动作
                        action = np.random.choice(valid_actions)
                else:
                    # 如果没有控制器，随机选择动作
                    action = np.random.choice(valid_actions)

                # 检测重复动作模式
                if self._repeated_actions_detected(action, actions_buffs[i]):
                    # 如果检测到重复，选择一个不同的随机动作
                    other_valid_actions = [a for a in valid_actions if a != action]
                    if other_valid_actions:  # 确保有其他有效动作可选
                        action = np.random.choice(other_valid_actions)

                # 记录动作到缓冲区
                actions_buffs[i].append(action)
                actions.append(action)

            # 使用环境的step函数执行动作
            next_obss, rewards, done, truncated, info = self.step(actions)
            
            # 确保next_obss是一个列表，每个智能体一个观察
            if not isinstance(next_obss, list):
                next_obss = self.get_agent_obs()

            # 更新累计奖励
            payoffs += rewards

            # 记录每个智能体的轨迹
            for i in range(len(self.players)):
                next_valid_actions = self._valid_actions[self.players[i]] if self.players[i] in self._valid_actions else list(range(7))

                # 轨迹格式：[obs_dict, action, reward, next_obs_dict, done]
                trajectory_segment = [
                    {'obs': obss[i] if isinstance(obss, list) else obss, 'actions': valid_actions},  # 当前观察和有效动作
                    actions[i],
                    rewards[i],
                    {'obs': next_obss[i] if isinstance(next_obss, list) else next_obss, 'actions': next_valid_actions},  # 下一观察和有效动作
                    done
                ]
                trajectories[i].append(trajectory_segment)

            # 更新观察
            obss = next_obss

            if render:
                self.render(mode=render_mode)
                time.sleep(sleep_time)

        return trajectories, payoffs

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
        food_count = sum(1 for food in self.food_items if not food.collected)
        agent_count = len(self.players)
        
        # 检测3D空间中的循环模式
        # 三维动作映射: 0-STAY, 1-UP, 2-DOWN, 3-LEFT, 4-RIGHT, 5-FORWARD, 6-BACKWARD
        
        # 检测轴向循环（在同一轴上来回移动）
        if len(actions_buff) >= 4:
            # 垂直方向循环 (UP-DOWN-UP-DOWN)
            if action in [1, 2] and all(a in [1, 2] for a in list(actions_buff)[-3:]):
                # 检查是否形成上下交替模式
                pattern = [action] + list(actions_buff)[-3:]
                if (pattern[0] == 1 and pattern[1] == 2 and pattern[2] == 1 and pattern[3] == 2) or \
                   (pattern[0] == 2 and pattern[1] == 1 and pattern[2] == 2 and pattern[3] == 1):
                    return True
            
            # 水平方向循环 (LEFT-RIGHT-LEFT-RIGHT)
            if action in [3, 4] and all(a in [3, 4] for a in list(actions_buff)[-3:]):
                # 检查是否形成左右交替模式
                pattern = [action] + list(actions_buff)[-3:]
                if (pattern[0] == 3 and pattern[1] == 4 and pattern[2] == 3 and pattern[3] == 4) or \
                   (pattern[0] == 4 and pattern[1] == 3 and pattern[2] == 4 and pattern[3] == 3):
                    return True
            
            # 前后方向循环 (FORWARD-BACKWARD-FORWARD-BACKWARD)
            if action in [5, 6] and all(a in [5, 6] for a in list(actions_buff)[-3:]):
                # 检查是否形成前后交替模式
                pattern = [action] + list(actions_buff)[-3:]
                if (pattern[0] == 5 and pattern[1] == 6 and pattern[2] == 5 and pattern[3] == 6) or \
                   (pattern[0] == 6 and pattern[1] == 5 and pattern[2] == 6 and pattern[3] == 5):
                    return True
        
        # 检测多维循环（在三个维度上形成循环）
        if len(actions_buff) >= 6:
            # 检测是否形成六步循环：例如UP-LEFT-FORWARD-DOWN-RIGHT-BACKWARD
            axis_groups = {
                'vertical': [1, 2],      # UP/DOWN
                'horizontal': [3, 4],    # LEFT/RIGHT
                'depth': [5, 6]          # FORWARD/BACKWARD
            }
            
            # 提取最近的6个动作（包括当前动作）
            recent_actions = [action] + list(actions_buff)[-5:]
            
            # 检查是否每个维度都使用了，并且形成了循环
            dimensions_used = set()
            for a in recent_actions:
                if a in axis_groups['vertical']:
                    dimensions_used.add('vertical')
                elif a in axis_groups['horizontal']:
                    dimensions_used.add('horizontal')
                elif a in axis_groups['depth']:
                    dimensions_used.add('depth')
            
            # 如果使用了全部三个维度，检查是否形成循环
            if len(dimensions_used) == 3:
                # 检查是否最近6步和更早的6步形成相同模式
                if len(actions_buff) >= 12:
                    if recent_actions == list(actions_buff)[-11:-5]:
                        return True

        # 单目标情况下使用较严格的检测
        if food_count <= 1:
            # 检测循环模式：如果缓冲区足够长，检查是否有2步或3步的循环
            if len(actions_buff) >= 3 and action != 0:  # 非STAY动作
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
                            
                    # 检测4步循环 (A-B-C-D-A-B-C-D模式)
                    if len(actions_buff) >= 8:
                        if (action == actions_buff[-4] and
                            actions_buff[-1] == actions_buff[-5] and
                            actions_buff[-2] == actions_buff[-6] and
                            actions_buff[-3] == actions_buff[-7]):
                            return True

        # 多目标多智能体情况下，仅检测长时间重复的相同动作
        else:
            # 检查是否有过多的重复动作
            if len(actions_buff) >= 100:  # 将阈值从6改为100，大幅增加允许的重复动作次数
                # 检测100次完全相同的非STAY动作
                if action != 0 and all(a == action for a in list(actions_buff)[-99:]):
                    return True
            
            # 检测在多个轴上的往复运动
            if len(actions_buff) >= 100:  # 将阈值从12改为100
                # 创建动作对 (垂直、水平、深度方向的动作对)
                action_pairs = [(1, 2), (3, 4), (5, 6)]
                for a1, a2 in action_pairs:
                    # 检查是否在某一对方向上有超过50次往复运动
                    pair_count = sum(1 for a in list(actions_buff)[-99:] + [action] if a in [a1, a2])
                    if pair_count >= 50:  # 将阈值从8改为50
                        # 进一步检查是否真的在往复运动（例如不是一直向一个方向）
                        directions = [a for a in list(actions_buff)[-99:] + [action] if a in [a1, a2]]
                        changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
                        if changes >= 20:  # 从3次增加到20次方向变化
                            return True

        return False

    def get_valid_actions(self):
        """返回所有玩家的有效动作组合"""
        from itertools import product
        return list(product(*[self._valid_actions[player] for player in self.players]))

    # 兼容性方法，适配原ForagingEnv的接口
    @property
    def rows(self):
        """返回行数"""
        return self.n_rows

    @property
    def cols(self):
        """返回列数"""
        return self.n_cols

    @property
    def depth(self):
        """返回深度"""
        return self.n_depth

    @property
    def field_size(self):
        """返回场地大小"""
        return self._field_size

    @field_size.setter
    def field_size(self, value):
        """设置场地大小"""
        self._field_size = value

    @property
    def game_over(self):
        """返回游戏是否结束"""
        return self._game_over

    def update_sight(self, new_sight):
        """更新视野范围"""
        old_sight = self.sight
        self.sight = new_sight
        self.logger.info(f"已更新视野范围: {old_sight} -> {new_sight}")
        return self.sight


# Example usage:
if __name__ == "__main__":
    # Create a 3D Foraging environment
    env = ForagingEnv3D(
        n_rows=6,
        n_cols=6,
        n_depth=6,  # 3D dimension
        num_agents=2,
        num_food=1,
        sight=None,
        max_player_level=3,
        max_food_level=3,
        grid_observation=True,
        max_episode_steps=100  # 明确设置最大步数为100
    )

    # 颜色代码
    class Colors:
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    def print_colorful_slice(env, z, reward=None, action=None):
        """打印单个Z层的彩色切片"""
        print(f"{Colors.BOLD}{Colors.UNDERLINE}Z层 #{z}{Colors.ENDC}")
        
        # 打印列标题
        col_header = '   '
        for col in range(env.n_cols):
            col_header += f" {col:2} "
        print(col_header)
        
        # 打印分隔线
        h_line = '  +'
        for _ in range(env.n_cols):
            h_line += '---+'
        print(h_line)
        
        # 打印每一行
        for row in range(env.n_rows):
            row_str = f"{row:2}|"
            for col in range(env.n_cols):
                cell = ' '
                
                # 标记食物
                for food in env.food_items:
                    if not food.collected and food.position == (row, col, z):
                        cell = f"{Colors.GREEN}F{food.level}{Colors.ENDC}"
                
                # 标记玩家
                for i, player in enumerate(env.players):
                    if player.position == (row, col, z):
                        if cell != ' ':
                            cell += f"{Colors.RED}P{i}{Colors.ENDC}"
                        else:
                            cell = f"{Colors.BLUE}P{i}{Colors.ENDC}"
                
                row_str += f" {cell:3}|"
            
            print(row_str)
            print(h_line)
    
    def print_3d_grid_state(env, rewards=None, actions=None):
        """打印整个3D网格的所有切片"""
        print("\n" + "="*50)
        print(f"{Colors.BOLD}环境状态 | 步数: {env._current_steps}/{env._max_episode_steps}{Colors.ENDC}")
        
        # 打印玩家状态
        print(f"\n玩家状态:")
        for i, player in enumerate(env.players):
            print(f"  玩家 #{i}: 位置={player.position}, 等级={player.level}, 分数={player.score:.2f}")
            if rewards is not None:
                print(f"     奖励: {Colors.YELLOW}{rewards[i]:.3f}{Colors.ENDC}")
            if actions is not None:
                action_name = env.action_set.get(actions[i], "未知")
                print(f"     动作: {Colors.CYAN}{action_name}{Colors.ENDC}")
        
        # 打印食物状态
        print(f"\n食物状态:")
        remaining_food = 0
        for i, food in enumerate(env.food_items):
            status = "可获取" if not food.collected else "已收集"
            status_color = Colors.GREEN if not food.collected else Colors.RED
            print(f"  食物 #{i}: 位置={food.position}, 等级={food.level}, 状态={status_color}{status}{Colors.ENDC}")
            if not food.collected:
                remaining_food += 1
        
        # 打印每个Z层的切片
        print(f"\n网格表示 (剩余食物: {remaining_food}/{len(env.food_items)}):")
        for z in range(env.n_depth):
            print_colorful_slice(env, z, rewards, actions)
        
        print("="*50 + "\n")
    
    # Reset the environment
    obs = env.reset()

    # 打印初始状态
    print(f"\n{Colors.BOLD}{Colors.PURPLE}初始状态:{Colors.ENDC}")
    print_3d_grid_state(env)
    
    # 循环执行随机动作
    total_rewards = np.zeros(env.num_agents)
    print(f"\n{Colors.BOLD}开始执行随机动作:{Colors.ENDC}")
    
    step_count = 20  # 增加步数，以便观察更长时间
    for step in range(step_count):
        print(f"\n{Colors.BOLD}{Colors.PURPLE}步骤 #{step+1}/{step_count}{Colors.ENDC}")
        
        # 为每个智能体选择随机动作
        actions = [np.random.randint(0, 7) for _ in range(env.num_agents)]
        action_names = [env.action_set[a] for a in actions]
        print(f"动作: {Colors.CYAN}{action_names}{Colors.ENDC}")

        # 执行环境步进
        obs, rewards, done, truncated, info = env.step(actions)
        total_rewards += rewards

        # 打印更新后的状态
        print(f"\n{Colors.BOLD}更新后的状态:{Colors.ENDC}")
        print_3d_grid_state(env, rewards, actions)
        
        # 打印奖励和总分
        print(f"当前回合奖励: {Colors.YELLOW}{rewards}{Colors.ENDC}")
        print(f"累计奖励总分: {Colors.YELLOW}{total_rewards}{Colors.ENDC}")
        print(f"游戏是否结束: {Colors.RED if done else Colors.GREEN}{done}{Colors.ENDC}")
        
        # 如果游戏结束，打印结果和原因
        if done:
            print(f"\n{Colors.BOLD}{Colors.PURPLE}游戏结束!{Colors.ENDC}")
            if all(food.collected for food in env.food_items):
                print(f"{Colors.GREEN}原因: 所有食物已收集{Colors.ENDC}")
            elif env._current_steps >= env._max_episode_steps:
                print(f"{Colors.RED}原因: 达到最大步数 ({env._max_episode_steps}){Colors.ENDC}")
            break

    env.close()
    print(f"\n{Colors.BOLD}{Colors.PURPLE}运行结束!{Colors.ENDC}")