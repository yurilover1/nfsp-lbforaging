import time
from collections import deque
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding

from .types import Field, Player


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5

# 动作到位置偏移的映射
ACTION_TO_OFFSET = {
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.WEST: (0, -1),
    Action.EAST: (0, 1),
}

class ForagingEnv(gym.Env):
    """
    使用Field类管理的LBForaging环境
    
    主要改进：
    - 使用Field类集中管理环境状态
    - 智能体动作对环境的影响集中展示
    - 更好的性能和可维护性
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(
        self,
        num_agents,
        min_player_level,
        max_player_level,
        min_food_level,
        max_food_level,
        field_size,
        max_num_food,
        sight,
        max_episode_steps,
        force_coop,
        observe_agent_levels=True,
        render_mode=None,
        step_reward_factor=0.1,
        attraction_reward_factor=0.1,
        decay_rate=0.01,
        seed=None,
        verbose=False,
    ):
        self.render_mode = render_mode
        self.num_agents = num_agents
        self.verbose = verbose
        
        # 使用Field类集中管理环境状态
        self.field = Field(field_size)
        
        # 玩家对象列表
        self.players = []
        
        # 奖励参数
        self.step_reward_factor = step_reward_factor
        self.attraction_reward_factor = attraction_reward_factor
        self.decay_rate = decay_rate
        
        # 位置历史记录
        self.agent_positions_history = []
        self.food_positions_history = []
        self.step_attraction_rewards = []  # 每步吸引力奖励记录

        # 食物等级配置
        self.min_food_level = self._ensure_array(min_food_level, max_num_food)
        self.max_food_level = self._ensure_array(max_food_level, max_num_food) if max_food_level is not None else None
        self.max_num_food = max_num_food
        self._food_spawned = 0.0

        # 智能体等级配置
        self.min_player_level = self._ensure_array(min_player_level, num_agents)
        self.max_player_level = self._ensure_array(max_player_level, num_agents)

        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps
        self._observe_agent_levels = observe_agent_levels

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(6)] * self.num_agents))
        
        # 观测空间计算
        if self.max_food_level is not None:
            max_food_level = max(self.max_food_level) if hasattr(self.max_food_level, '__iter__') else self.max_food_level
        else:
            max_food_level = sum(sorted(self.max_player_level)[:3])
        min_obs = ([-1, -1, 0] * self.max_num_food + 
                  ([-1, -1, 0] if self._observe_agent_levels else [-1, -1]) * self.num_agents)
        max_obs = ([field_size[1] - 1, field_size[0] - 1, max_food_level] * self.max_num_food + 
                  ([field_size[1] - 1, field_size[0] - 1, max(self.max_player_level)] 
                   if self._observe_agent_levels else [field_size[1] - 1, field_size[0] - 1]) * self.num_agents)
        
        obs_space = gym.spaces.Box(low=np.array(min_obs, dtype=np.float32), high=np.array(max_obs, dtype=np.float32), dtype=np.float32)
        self.observation_space = gym.spaces.Tuple(tuple([obs_space] * self.num_agents))

        self.viewer = None
        self.reward_events = []
        
        # 设置随机种子
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        else:
            self._np_random, _ = seeding.np_random(None)

        # 确保环境初始化时生成食物
        self._spawn_food()

        # 有效/无效动作统计
        self.valid_action_count = 0
        self.invalid_action_count = 0

    def _get_field_int_representation(self):
        """获取field的整数表示，用于兼容性"""
        return self.field.to_int_array()

    def set_player_agents(self, agents):
        """为玩家设置智能体对象"""
        for i, agent in enumerate(agents):
            if i < len(self.players):
                self.players[i].agent = agent
    
    def _get_ego_player(self):
        """获取ego玩家（第一个玩家）"""
        return self.players[0] if len(self.players) > 0 else None

    def distribute_rewards(self, rewards):
        """为ego玩家分发奖励（仅给主玩家奖励）"""
        ego_player = self._get_ego_player()
        if ego_player:
            if isinstance(rewards, (int, float)):
                ego_player.add_reward(rewards)
            elif isinstance(rewards, (list, tuple)) and len(rewards) > 0:
                ego_player.add_reward(rewards[0])
    
    def reset_player_rewards(self):
        """重置ego玩家的奖励"""
        ego_player = self._get_ego_player()
        if ego_player:
            ego_player.reset_reward()

    @staticmethod
    def _ensure_array(value, size):
        """确保值是指定大小的数组"""
        if hasattr(value, '__iter__') and not isinstance(value, str):
            return np.array(value)
        return np.array([value] * size)

    @staticmethod
    def _calculate_new_position(current_pos, action):
        """计算动作后的新位置"""
        # 将整数动作转换为Action枚举
        if isinstance(action, int):
            try:
                action_enum = Action(action)
            except ValueError:
                return current_pos
        else:
            action_enum = action
        
        if action_enum in ACTION_TO_OFFSET:
            offset = ACTION_TO_OFFSET[action_enum]
            new_pos = (current_pos[0] + offset[0], current_pos[1] + offset[1])
            return new_pos
        else:
            return current_pos

    #==============================
    # 玩家访问属性（委托给Field类和Player对象）
    #==============================
    
    @property
    def agent_positions(self):
        """动态生成玩家位置列表（保持向后兼容）"""
        positions = []
        for player_id in range(self.num_agents):
            pos = self.field.get_player_position(player_id)
            positions.append(pos if pos is not None else (0, 0))
        return positions
    
    @property
    def agent_levels(self):
        """动态生成玩家等级列表（保持向后兼容）"""
        levels = []
        for player_id in range(self.num_agents):
            level = self.field.get_player_level(player_id)
            levels.append(level if level > 0 else 1)
        return levels
    
    def get_agent_position_from_field(self, player_id):
        """根据玩家ID获取其位置（保持向后兼容）"""
        return self.field.get_player_position(player_id)
    
    def get_agent_level_from_field(self, player_id):
        """根据玩家ID获取其等级（保持向后兼容）"""
        return self.field.get_player_level(player_id)
    
    def get_player_object(self, player_id):
        """根据玩家ID获取Player对象"""
        return self.field.get_player_object(player_id)
    
    def get_agent_at_position(self, position):
        """获取指定位置的玩家ID，如果没有玩家返回None"""
        row, col = position
        try:
            player_id = self.field.get_player_id(row, col)
            return player_id if player_id >= 0 else None
        except IndexError:
            return None
    
    def is_agent_at_position(self, position):
        """检查指定位置是否有玩家"""
        return self.get_agent_at_position(position) is not None
    
    def get_agents_in_area(self, center_position, radius):
        """获取指定区域内的所有玩家ID列表"""
        center_row, center_col = center_position
        return self.field.get_players_in_area(center_row, center_col, radius)

    #==============================
    # 观测生成方法
    #==============================
    
    def _make_observations(self):
        """生成观测"""
        observations = []
        
        for agent_idx in range(self.num_agents):
            # 获取智能体位置
            agent_pos = self.agent_positions[agent_idx]
            
            # 如果智能体位置无效，生成空观测
            if agent_pos is None:
                obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
                observations.append(obs)
                continue
                
            # 创建观测向量
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            
            # 获取智能体视野内的食物信息
            local_foods = self._get_local_field(agent_pos)
            
            # 填充食物信息
            self._fill_food_info(obs, local_foods)
            
            # 填充智能体信息（已修改为将观察者放在第一位）
            self._fill_agent_info(obs, agent_pos)
            
            # 添加调试信息
            if hasattr(self, 'verbose') and self.verbose:
                print(f"智能体{agent_idx}的观测: 形状={obs.shape}, 内容={obs}")
            
            observations.append(obs)
            
        return tuple(observations)

    def _get_local_field(self, agent_pos):
        """获取智能体视野内的field信息"""
        row, col = agent_pos
        start_row = max(row - self.sight, 0)
        end_row = min(row + self.sight + 1, self.field.rows)
        start_col = max(col - self.sight, 0)
        end_col = min(col + self.sight + 1, self.field.cols)
        
        # 返回视野范围内的食物位置信息
        local_foods = []
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                if self.field.is_food_at(i, j):
                    level = self.field.get_field_point(i, j).get_level()
                    # 转换为相对坐标
                    rel_i = i - start_row
                    rel_j = j - start_col
                    local_foods.append((rel_i, rel_j, level))
        
        return local_foods

    def _fill_food_info(self, obs, local_foods):
        """填充食物信息到观测向量（一维数组，每个食物3个元素）"""
        for i in range(self.max_num_food):
            base_idx = 3 * i
            if i < len(local_foods):
                rel_i, rel_j, level = local_foods[i]
                obs[base_idx:base_idx + 3] = [rel_i, rel_j, level]
            else:
                obs[base_idx:base_idx + 3] = [-1, -1, 0]

    def _fill_agent_info(self, obs, observer_pos):
        """填充玩家信息到观测向量"""
        start_idx = self.max_num_food * 3
        agent_obs_len = 3 if self._observe_agent_levels else 2
        
        # 获取观察者ID
        observer_id = None
        for i in range(self.num_agents):
            if self.get_agent_position_from_field(i) == observer_pos:
                observer_id = i
                break
        
        # 如果找不到观察者ID，使用默认顺序
        if observer_id is None:
            agent_order = list(range(self.num_agents))
        else:
            # 将观察者放在第一位
            agent_order = [observer_id] + [i for i in range(self.num_agents) if i != observer_id]
        
        # 按照重新排序后的顺序填充智能体信息
        for idx, i in enumerate(agent_order):
            base_idx = start_idx + agent_obs_len * idx
            player_pos = self.get_agent_position_from_field(i)
            
            # 检查是否在视野内
            if player_pos and self._in_sight(observer_pos, player_pos):
                # 转换为相对坐标
                rel_pos = self._to_relative_pos(observer_pos, player_pos)
                obs[base_idx:base_idx + 2] = rel_pos
                if self._observe_agent_levels:
                    obs[base_idx + 2] = self.get_agent_level_from_field(i)
            else:
                obs[base_idx:base_idx + 2] = [-1, -1]
                if self._observe_agent_levels:
                    obs[base_idx + 2] = 0

    def _in_sight(self, observer_pos, target_pos):
        """检查目标是否在观察者视野内"""
        dx = abs(observer_pos[0] - target_pos[0])
        dy = abs(observer_pos[1] - target_pos[1])
        return dx <= self.sight and dy <= self.sight

    def _to_relative_pos(self, observer_pos, target_pos):
        """将目标位置转换为相对于观察者的位置"""
        return (
            target_pos[0] - observer_pos[0] + min(self.sight, observer_pos[0]),
            target_pos[1] - observer_pos[1] + min(self.sight, observer_pos[1])
        )

    #==============================
    # 游戏状态访问
    #==============================

    @property
    def game_over(self):
        return self._game_over

    #==============================
    # 位置和动作验证
    #==============================
    
    def _gen_valid_actions(self):
        """计算每个智能体的有效动作"""
        self._valid_actions = []
        movement_actions = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
        
        for agent_idx in range(self.num_agents):
            valid_actions = [Action.NONE]  # NONE总是有效
            agent_pos = self.get_agent_position_from_field(agent_idx)
            
            # 检查移动动作
            valid_actions.extend(action for action in movement_actions 
                               if self._is_valid_move_action(agent_idx, action))
                    
            # 检查加载动作
            if self._can_load(agent_pos):
                valid_actions.append(Action.LOAD)
                
            self._valid_actions.append(valid_actions)

    def _is_valid_move_action(self, agent_idx, action):
        """检查移动动作是否有效"""
        pos = self.get_agent_position_from_field(agent_idx)
        if pos is None:
            return False
        new_pos = self._calculate_new_position(pos, action)
            
        return self._is_valid_position(new_pos)

    def _is_valid_position(self, pos):
        """检查位置是否有效"""
        row, col = pos
        
        # 边界检查
        if not self.field._is_valid_position(row, col):
            return False
            
        # 占用检查 - 检查是否为空
        if not self.field.is_empty(row, col):
            return False
                
        return True

    def _can_load(self, pos):
        """检查是否可以加载相邻食物"""
        row, col = pos
        adjacent_positions = self.field.get_adjacent_positions(row, col)
        for adj_row, adj_col in adjacent_positions:
            if self.field.is_food_at(adj_row, adj_col):
                return True
        return False

    #==============================
    # 智能体移动和field更新
    #==============================

    def _process_agent_movement(self, player_idx, action):
        """处理玩家移动"""
        # 统计动作有效性
        if action == Action.LOAD.value:
            self.valid_action_count += 1
            return True  # 表示动作已处理，奖励逻辑在后续步骤处理
        elif action == Action.NONE.value:
            # NONE动作总是有效的，表示智能体选择不移动
            self.valid_action_count += 1
            return False
        elif action in [Action.NORTH.value, Action.SOUTH.value, Action.WEST.value, Action.EAST.value]:
            pos = self.field.get_player_position(player_idx)
            if pos is None:
                self.invalid_action_count += 1
                return False
            new_pos = self._calculate_new_position(pos, action)
            can_move = self._is_valid_position(new_pos)
            if can_move:
                self.field.move_player(player_idx, new_pos[0], new_pos[1])
                self.valid_action_count += 1
            else:
                self.invalid_action_count += 1
        else:
            self.invalid_action_count += 1
        return False  # 返回False表示不是加载动作

    #==============================
    # 环境重置和初始化
    #==============================

    def reset(self, seed=None, options=None):
        """重置环境"""
        self.current_step = 0
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # 重置field - Field类会自动处理清空和初始化
        self.field = Field(self.field.field_size)
        
        # 重置玩家列表
        self.players = []
        
        # 重置食物相关变量
        self._food_spawned = 0.0
        
        self._spawn_agents()
        self._spawn_food()
        
        self._game_over = False
        self._gen_valid_actions()

        # 重置玩家奖励
        self.reset_player_rewards()

        observations = self._make_observations()
        self.reward_events = []
        if not hasattr(self, 'current_episode'):
            self.current_episode = 0
        else:
            self.current_episode += 1
        if self.current_episode == 0:
            import os
            os.makedirs('logs', exist_ok=True)
            with open('logs/attraction_step_reward.csv', 'w') as f:
                f.write('episode,step,attraction_reward,agent_pos,food_pos\n')
        
        # 重置统计变量
        self.valid_action_count = 0
        self.invalid_action_count = 0
        return observations, {}

    def _spawn_agents(self):
        """生成玩家"""
        for i in range(self.num_agents):
            # 创建Player对象
            player_level = (
                self._np_random.integers(self.min_player_level[i], self.max_player_level[i] + 1) 
                if i == 0 else 1
            )
            player = Player(
                player_id=i,
                level=player_level,
                is_self=(i == 0)  # 第一个玩家为主玩家
            )
            self.players.append(player)
            
            # 为玩家找位置
            attempts = 0
            while attempts < 1000:
                row = self._np_random.integers(0, self.field.rows)
                col = self._np_random.integers(0, self.field.cols)
                if self.field.is_empty(row, col):
                    if self.field.place_player(player, row, col):
                        break
                attempts += 1

    def _spawn_food(self):
        """生成食物（基于food_positions，带详细调试）"""
        food_count = 0
        attempts = 0
        min_levels = self.max_food_level if self.force_coop else self.min_food_level
        max_levels = self.max_food_level if self.max_food_level is not None else np.array([sum(sorted(self.agent_levels)[:2])] * self.max_num_food)
        while food_count < self.max_num_food and attempts < 1000:
            attempts += 1
            row = self._np_random.integers(0, self.field.rows)
            col = self._np_random.integers(0, self.field.cols)
            can_place = self.field.is_empty(row, col)
            if can_place:
                food_level = (
                    min_levels[food_count] if min_levels[food_count] == max_levels[food_count]
                    else self._np_random.integers(min_levels[food_count], max_levels[food_count] + 1)
                )
                placed = self.field.place_food(row, col, food_level)
                if placed:
                    food_count += 1
        self._food_spawned = self.field.get_total_food_level()

    def _process_loading(self, loading_agents):
        """处理食物加载（基于food_positions）"""
        agents_to_process = set(loading_agents)
        loaded_foods = []
        while agents_to_process:
            agent_idx = agents_to_process.pop()
            agent_pos = self.get_agent_position_from_field(agent_idx)
            if agent_pos is None:
                continue
            food_pos = self._get_adjacent_food(agent_pos)
            if food_pos is None:
                continue
            food_level = self.field.food_positions.get(food_pos, 0)
            if food_level == 0:
                continue
            adj_agents = [a for a in self._get_adjacent_agents(food_pos) if a in agents_to_process]
            total_level = sum(self.field.get_player_level(a) for a in adj_agents) + self.field.get_player_level(agent_idx)
            agents_to_process -= set(adj_agents)
            success = total_level >= food_level
            for a in adj_agents + [agent_idx]:
                loaded_foods.append({'success': success, 'agent': a, 'food_level': food_level})
            if success:
                self.field.remove_food(*food_pos)
        return loaded_foods

    def _get_adjacent_food(self, pos):
        """获取相邻食物位置（基于food_positions）"""
        row, col = pos
        adjacent_positions = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        for adj_row, adj_col in adjacent_positions:
            if (adj_row, adj_col) in self.field.food_positions:
                return (adj_row, adj_col)
        return None

    def _get_adjacent_agents(self, pos):
        """获取相邻玩家（基于player_positions）"""
        row, col = pos
        adjacent = []
        adjacent_positions = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        for adj_row, adj_col in adjacent_positions:
            for player_id, player_pos in self.field.player_positions.items():
                if player_pos == (adj_row, adj_col):
                    adjacent.append(player_id)
                    break
        return adjacent

    #==============================
    # 环境主要交互方法
    #==============================
    
    def __step(self, ego_action, teammate_actions):
        self.current_step += 1
        # ====== 处理 agent 动作 ======
        all_actions = [ego_action] + teammate_actions
        loading_agents = []
        
        # 处理每个 agent 的动作
        for agent_idx, action in enumerate(all_actions):
            if action == Action.LOAD.value:
                loading_agents.append(agent_idx)
            else:
                # 处理移动动作
                self._process_agent_movement(agent_idx, action)
        
        # 处理加载动作
        if loading_agents:
            self._process_loading(loading_agents)
        # ====== END ======
        
        # 记录位置历史
        self.agent_positions_history.append(list(self.agent_positions))
        food_positions = [(i, j) for i, j, level in self.field.get_all_food_infos()]
        self.food_positions_history.append(food_positions)
        
        # ====== 使用Field类计算吸引力奖励 ======
        if not hasattr(self, 'step_attraction_rewards'):
            self.step_attraction_rewards = []
        
        # 获取主玩家（ego player）
        ego_player = self.players[0] if len(self.players) > 0 else None
        
        # 使用Field类计算吸引力奖励
        if ego_player and self.current_step > 1:
            # 设置Field的吸引力奖励因子
            self.field.set_attraction_reward_factor(self.attraction_reward_factor)
            # 计算吸引力奖励（包含日志记录）
            norm_reward = self.field.update_attraction_reward(
                ego_player, 
                self.current_step, 
                getattr(self, 'current_episode', 0)
            )
            self.step_attraction_rewards.append(norm_reward)
        else:
            norm_reward = 0.0
        
        # ====== END ======
        
        # 检查游戏是否结束
        food_sum = self.field.get_total_food_level()
        self._game_over = (food_sum == 0 or self._max_episode_steps <= self.current_step)
        
        # 更新有效动作
        self._gen_valid_actions()
        
        # 计算终局奖励
        if self._game_over:
            reward_detail = self._calculate_final_reward()
            final_reward = reward_detail['total']
            self._last_reward_detail = reward_detail  # 保存详细reward组成
            self.distribute_rewards(final_reward)
        else:
            final_reward = 0.0
            self._last_reward_detail = {'total': 0.0, 'base': 0.0, 'attraction': 0.0, 'step': 0.0}
        
        # 修改：每步reward都包含吸引力奖励，终局时再加终局奖励
        reward = norm_reward
        if self._game_over:
            reward += final_reward
        
        # 准备返回值
        done = self._game_over
        truncated = False
        
        # 重新生成观测
        observations = self._make_observations()
        
        return observations, reward, done, truncated, {}

    def step(self, actions):
        """执行环境中的一个时间步骤"""
        if isinstance(actions, (list, tuple, np.ndarray)):
            ego_action = actions[0]
            teammate_actions = actions[1:] if len(actions) > 1 else []
        else:
            ego_action = actions
            teammate_actions = []
        return self.__step(ego_action, teammate_actions)
    
    def run(self, agents, is_training=False, render=False, sleep_time=0.5):
        """运行完整的回合"""
        # 重置环境
        obss, _ = self.reset()
        done = False
        trajectories = []
        final_reward = 0

        # 将智能体关联到玩家对象
        for i, agent in enumerate(agents):
            if i < len(self.players):
                self.players[i].agent = agent

        # 初始化玩家设置
        ego_player = self.players[0] if len(self.players) > 0 else None
        teammate_players = self.players[1:] if len(self.players) > 1 else []
        
        # 初始化动作缓冲区
        ego_actions_buff = deque(maxlen=50)
        teammate_actions_buffs = [deque(maxlen=50) for _ in range(len(teammate_players))]

        # 渲染初始状态
        if render:
            self.render()
            time.sleep(sleep_time)
        
        # 逐步执行，直到回合结束
        while not done:
            # 获取ego玩家的动作
            ego_action = self._get_ego_action(ego_player, obss, ego_actions_buff, is_training)
            
            # 获取teammate玩家的动作列表
            teammate_actions = self._get_teammate_actions(teammate_players, obss, teammate_actions_buffs, is_training)
            
            # 执行动作并获取结果
            next_obss, reward, done, _, _ = self.__step(ego_action, teammate_actions)
            
            # 记录轨迹
            if is_training:
                trajectories.append([obss[0], ego_action, next_obss[0], done])
            
            # 记录终局奖励和详细组成
            if done:
                final_reward = reward
                final_reward_detail = self._last_reward_detail.copy() if hasattr(self, '_last_reward_detail') else None
            
            # 更新观察
            obss = next_obss

            if render:
                self.render()
                time.sleep(sleep_time)

        # 添加轨迹到经验缓冲区
        if is_training and ego_player and ego_player.agent and hasattr(ego_player.agent, 'add_traj2buffer'):
            step_attraction_rewards = self.step_attraction_rewards if hasattr(self, 'step_attraction_rewards') else []
            for idx, ts in enumerate(trajectories):
                # 每步奖励只包含该步吸引力奖励，只有最后一步加终局奖励
                step_reward = 0.0
                if idx < len(step_attraction_rewards):
                    step_reward += step_attraction_rewards[idx]
                if idx == len(trajectories) - 1:
                    step_reward += final_reward
                ego_player.agent.add_traj2buffer([
                    ts[0],
                    ts[1],
                    step_reward,
                    ts[2],
                    ts[3]
                ])
            self.step_attraction_rewards = []  # 清空
        
        return trajectories, final_reward, self.current_step, final_reward_detail

    def _get_ego_action(self, ego_player, obss, ego_actions_buff, is_training):
        if not ego_player:
            return self._validate_single_action(0, Action.NONE.value)
            
        # 获取ego的有效动作
        valid_actions = [action.value for action in self._valid_actions[0]]
        
        # 使用Player对象的select_action接口
        action = ego_player.select_action({'obs': obss[0], 'actions': valid_actions}, is_training)
        
        # 检测重复动作
        if self._repeated_actions_detected(action, ego_actions_buff):
            other_valid_actions = [a for a in valid_actions if a != action]
            if other_valid_actions:
                action = np.random.choice(other_valid_actions)
        
        ego_actions_buff.append(action)
        
        # 验证ego动作
        validated_action = self._validate_single_action(0, action)
        return validated_action

    def _get_teammate_actions(self, teammate_players, obss, teammate_actions_buffs, is_training):
        teammate_actions = []
        for i in range(1, self.num_agents):
            teammate_idx = i - 1
            if teammate_idx < len(teammate_players):
                teammate_player = teammate_players[teammate_idx]
                valid_actions = [action.value for action in self._valid_actions[i]]
                obs_input = obss[i]
                # 只传obs和actions，避免is_training传错位置
                action = teammate_player.select_action({'obs': obs_input, 'actions': valid_actions})
                
                # 检测重复动作
                if self._repeated_actions_detected(action, teammate_actions_buffs[teammate_idx]):
                    other_valid_actions = [a for a in valid_actions if a != action]
                    if other_valid_actions:
                        action = np.random.choice(other_valid_actions)
                
                teammate_actions_buffs[teammate_idx].append(action)
                
                # 验证teammate动作
                teammate_action = self._validate_single_action(i, action)
            else:
                # 没有对应的teammate玩家，使用默认动作
                teammate_action = self._validate_single_action(i, Action.NONE.value)
            
            teammate_actions.append(teammate_action)
        
        return teammate_actions

    def _validate_single_action(self, agent_idx, action):
        """验证单个智能体的动作"""
        try:
            action_enum = Action(action)
            if action_enum in self._valid_actions[agent_idx]:
                return action_enum.value
            else:
                return Action.NONE.value
        except ValueError:
            return Action.NONE.value

    def _repeated_actions_detected(self, action, actions_buff):
        """增强循环检测灵活性"""
        buff = list(actions_buff)
        if not buff or action == 5:  # LOAD动作不检测重复
            return False
        buff_len = len(buff)
        # 检测连续相同动作（至少4次）
        if buff_len >= 3 and all(action == buff[-i-1] for i in range(3)):
            return True
        # 检测2~5步循环模式
        for cycle in range(2, 6):
            if buff_len >= cycle * 2:
                pattern = buff[-cycle:]
                if all(buff[-cycle*(j+1):-cycle*j] == pattern for j in range(1, 2)):
                    return True
        return False

    #==============================
    # 终局奖励计算
    #==============================
    
    def _calculate_final_reward(self):
        """计算终局奖励，返回详细组成"""
        reward = 0.0
        base_reward = 0.0
        step_reward = 0.0

        # 基础奖励
        food_sum = self.field.get_total_food_level()
        if food_sum == 0:
            base_reward = 1.0
        else:
            success_rate = (self._food_spawned - food_sum) / self._food_spawned if self._food_spawned > 0 else 0
            base_reward = success_rate * 0.5 if success_rate > 0 else -1.0
            # 限制基础奖励的最大值为1.0
            base_reward = min(base_reward, 1.0)
        reward += base_reward

        # 步数奖励
        if reward > 0:
            step_efficiency = np.exp(-self.decay_rate * self.current_step)
            step_reward = reward * self.step_reward_factor * step_efficiency
            reward += step_reward

        return {
            'total': reward,
            'base': base_reward,
            'attraction': 0.0,  # 保留字段但设为0，保持兼容性
            'step': step_reward
        }



    #==============================
    # 渲染方法
    #==============================
    
    def render(self):
        """渲染环境"""
        if not self._rendering_initialized:
            from .rendering import Viewer
            self.viewer = Viewer(self.field.field_size)
            self._rendering_initialized = True
            
        # 获取field的二维数组表示
        field_array = self._get_field_int_representation()
        
        # 获取智能体位置和等级
        agent_positions = self.agent_positions
        agent_levels = self.agent_levels
        
        # 获取食物位置和等级
        food_positions = self.field.food_positions
        
        # 使用新的渲染方法，传递智能体和食物的位置信息
        return self.viewer.render(
            field_array, 
            agent_positions=agent_positions,
            agent_levels=agent_levels,
            food_positions=food_positions,
            return_rgb_array=self.render_mode == "rgb_array"
        )

    def close(self):
        """关闭环境"""
        if self.viewer:
            self.viewer.close() 