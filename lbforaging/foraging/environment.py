import time
from collections import deque
from enum import Enum
import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding

from .types import FieldType, FieldPoint, Field


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


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
        attraction_reward_factor=0.5,
        decay_rate=0.01,
        seed=None,
    ):
        self.render_mode = render_mode
        self.num_agents = num_agents
        
        # 使用Field类集中管理环境状态
        self.field = Field(field_size)
        
        # 奖励参数
        self.step_reward_factor = step_reward_factor
        self.attraction_reward_factor = attraction_reward_factor
        self.decay_rate = decay_rate
        
        # 位置历史记录
        self.agent_positions_history = []
        self.food_positions_history = []

        # 食物等级配置
        self.min_food_level = np.array(min_food_level) if hasattr(min_food_level, '__iter__') else np.array([min_food_level] * max_num_food)
        if max_food_level is not None:
            self.max_food_level = np.array(max_food_level) if hasattr(max_food_level, '__iter__') else np.array([max_food_level] * max_num_food)
        else:
            self.max_food_level = None
        self.max_num_food = max_num_food
        self._food_spawned = 0.0

        # 智能体等级配置
        self.min_player_level = np.array(min_player_level) if hasattr(min_player_level, '__iter__') else np.array([min_player_level] * num_agents)
        self.max_player_level = np.array(max_player_level) if hasattr(max_player_level, '__iter__') else np.array([max_player_level] * num_agents)

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

    def _get_field_int_representation(self):
        """获取field的整数表示，用于兼容性"""
        return self.field.to_int_array()

    #==============================
    # 智能体访问属性（委托给Field类）
    #==============================
    
    @property
    def agent_positions(self):
        """动态生成智能体位置列表"""
        positions = []
        for agent_id in range(self.num_agents):
            pos = self.field.get_agent_position(agent_id)
            positions.append(pos if pos is not None else (0, 0))
        return positions
    
    @property
    def agent_levels(self):
        """动态生成智能体等级列表"""
        levels = []
        for agent_id in range(self.num_agents):
            level = self.field.get_agent_level(agent_id)
            levels.append(level if level > 0 else 1)
        return levels
    
    def get_agent_position_from_field(self, agent_id):
        """根据智能体ID获取其位置"""
        return self.field.get_agent_position(agent_id)
    
    def get_agent_level_from_field(self, agent_id):
        """根据智能体ID获取其等级"""
        return self.field.get_agent_level(agent_id)
    
    def get_agent_at_position(self, position):
        """获取指定位置的智能体ID，如果没有智能体返回None"""
        row, col = position
        try:
            agent_id = self.field.get_agent_id(row, col)
            return agent_id if agent_id >= 0 else None
        except IndexError:
            return None
    
    def is_agent_at_position(self, position):
        """检查指定位置是否有智能体"""
        return self.get_agent_at_position(position) is not None
    
    def get_agents_in_area(self, center_position, radius):
        """获取指定区域内的所有智能体ID列表"""
        center_row, center_col = center_position
        return self.field.get_agents_in_area(center_row, center_col, radius)

    #==============================
    # 观测生成方法
    #==============================
    
    def _make_observations(self):
        """生成观测"""
        observations = []
        
        for agent_idx in range(self.num_agents):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            agent_pos = self.agent_positions[agent_idx]
            
            # 获取智能体视野内的食物信息
            local_foods = self._get_local_field(agent_pos)
            
            # 填充食物信息
            self._fill_food_info(obs, local_foods)
            
            # 填充智能体信息
            self._fill_agent_info(obs, agent_idx, agent_pos)
            
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
        """填充食物信息到观测向量"""
        for i in range(self.max_num_food):
            base_idx = 3 * i
            if i < len(local_foods):
                y, x, level = local_foods[i]
                obs[base_idx:base_idx + 3] = [y, x, level]
            else:
                obs[base_idx:base_idx + 3] = [-1, -1, 0]

    def _fill_agent_info(self, obs, observer_idx, observer_pos):
        """填充智能体信息到观测向量"""
        start_idx = self.max_num_food * 3
        agent_obs_len = 3 if self._observe_agent_levels else 2
        
        for i in range(self.num_agents):
            base_idx = start_idx + agent_obs_len * i
            agent_pos = self.get_agent_position_from_field(i)
            
            # 检查是否在视野内
            if agent_pos and self._in_sight(observer_pos, agent_pos):
                # 转换为相对坐标
                rel_pos = self._to_relative_pos(observer_pos, agent_pos)
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
        for agent_idx in range(self.num_agents):
            valid_actions = []
            agent_pos = self.get_agent_position_from_field(agent_idx)
            
            # NONE总是有效
            valid_actions.append(Action.NONE)
            
            # 检查移动动作
            for action in [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]:
                if self._is_valid_move_action(agent_idx, action):
                    valid_actions.append(action)
                    
            # 检查加载动作
            if self._can_load(agent_pos):
                valid_actions.append(Action.LOAD)
                
            self._valid_actions.append(valid_actions)

    def _is_valid_move_action(self, agent_idx, action):
        """检查移动动作是否有效"""
        pos = self.get_agent_position_from_field(agent_idx)
        if pos is None:
            return False
        if action == Action.NORTH:
            new_pos = (pos[0] - 1, pos[1])
        elif action == Action.SOUTH:
            new_pos = (pos[0] + 1, pos[1])
        elif action == Action.WEST:
            new_pos = (pos[0], pos[1] - 1)
        elif action == Action.EAST:
            new_pos = (pos[0], pos[1] + 1)
        else:
            return False
            
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

    def _process_agent_movement(self, agent_idx, action):
        """处理智能体移动"""
        if action == Action.LOAD:
            return True  # 返回True表示是加载动作
        elif action in [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]:
            # 计算新位置
            pos = self.field.get_agent_position(agent_idx)
            if pos is None:
                return False
                
            if action == Action.NORTH:
                new_pos = (pos[0] - 1, pos[1])
            elif action == Action.SOUTH:
                new_pos = (pos[0] + 1, pos[1])
            elif action == Action.WEST:
                new_pos = (pos[0], pos[1] - 1)
            elif action == Action.EAST:
                new_pos = (pos[0], pos[1] + 1)
            
            if self._is_valid_position(new_pos):
                # 使用Field类的移动方法
                self.field.move_agent(agent_idx, new_pos[0], new_pos[1])
                
        return False  # 返回False表示不是加载动作

    def _update_field(self):
        """更新field以反映智能体位置"""
        # Field类自动维护状态，此方法保留以保持兼容性
        pass

    #==============================
    # 环境重置和初始化
    #==============================

    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # 重置field - Field类会自动处理清空和初始化
        self.field = Field(self.field.field_size)
        
        self._spawn_agents()
        self._update_field()
        self._spawn_food()
        
        self.current_step = 0
        self._game_over = False
        self._gen_valid_actions()

        observations = self._make_observations()
        self.reward_events = []
        return observations, {}

    def _spawn_agents(self):
        """生成智能体"""
        for i in range(self.num_agents):
            attempts = 0
            while attempts < 1000:
                row = self.np_random.integers(0, self.field.rows)
                col = self.np_random.integers(0, self.field.cols)
                if self.field.is_empty(row, col):
                    agent_level = (
                        self.np_random.integers(self.min_player_level[i], self.max_player_level[i] + 1) 
                        if i == 0 else 1
                    )
                    if self.field.place_agent(i, row, col, agent_level):
                        break
                attempts += 1

    def _spawn_food(self):
        """生成食物"""
        food_count = 0
        attempts = 0
        min_levels = self.max_food_level if self.force_coop else self.min_food_level
        max_levels = self.max_food_level if self.max_food_level is not None else np.array([sum(sorted(self.agent_levels)[:2])] * self.max_num_food)

        while food_count < self.max_num_food and attempts < 1000:
            attempts += 1
            row = self.np_random.integers(1, self.field.rows - 1)
            col = self.np_random.integers(1, self.field.cols - 1)

            if self.field.is_empty(row, col):
                food_level = (
                    min_levels[food_count] if min_levels[food_count] == max_levels[food_count]
                    else self.np_random.integers(min_levels[food_count], max_levels[food_count] + 1)
                )
                if self.field.place_food(row, col, food_level):
                    food_count += 1
                
        # 计算总食物量
        self._food_spawned = self.field.get_total_food_level()

    #==============================
    # 食物加载处理
    #==============================

    def _process_loading(self, loading_agents):
        """处理食物加载"""
        agents_to_process = set(loading_agents)
        loaded_foods = []
        
        while agents_to_process:
            agent_idx = agents_to_process.pop()
            agent_pos = self.get_agent_position_from_field(agent_idx)
            if agent_pos is None:
                continue
            
            # 查找相邻食物
            food_pos = self._get_adjacent_food(agent_pos)
            if food_pos is None:
                continue
                
            food_level = self.field.get_field_point(*food_pos).get_level()
            
            # 查找协作智能体
            adj_agents = [a for a in self._get_adjacent_agents(food_pos) if a in agents_to_process]
            total_level = sum(self.field.get_agent_level(a) for a in adj_agents) + self.field.get_agent_level(agent_idx)
            
            agents_to_process -= set(adj_agents)
            success = total_level >= food_level
            
            # 记录结果
            for a in adj_agents + [agent_idx]:
                loaded_foods.append({'success': success, 'agent': a, 'food_level': food_level})
            
            if success:
                self.field.remove_food(*food_pos)
         
        return loaded_foods

    def _get_adjacent_food(self, pos):
        """获取相邻食物位置"""
        row, col = pos
        adjacent_positions = self.field.get_adjacent_positions(row, col)
        for adj_row, adj_col in adjacent_positions:
            if self.field.is_food_at(adj_row, adj_col):
                return (adj_row, adj_col)
        return None

    def _get_adjacent_agents(self, pos):
        """获取相邻智能体"""
        row, col = pos
        adjacent = []
        adjacent_positions = self.field.get_adjacent_positions(row, col)
        for adj_row, adj_col in adjacent_positions:
            agent_id = self.field.get_agent_id(adj_row, adj_col)
            if agent_id >= 0:
                adjacent.append(agent_id)
        return adjacent

    #==============================
    # 环境主要交互方法
    #==============================
    
    def __step(self, ego_action, teammate_actions):
        """执行环境中的一个时间步骤"""
        # 记录位置历史
        self.agent_positions_history.append(list(self.agent_positions))
        food_positions = [(i, j) for i, j, level in self.field.get_all_food_infos()]
        self.food_positions_history.append(food_positions)
       
        # 处理移动和加载
        loading_agents = set()
        
        # 转换ego动作为Action枚举
        try:
            ego_action_enum = Action(ego_action)
        except ValueError:
            ego_action_enum = Action.NONE
        
        # 处理主智能体动作
        if self._process_agent_movement(0, ego_action_enum):
            loading_agents.add(0)
        
        # 处理队友智能体动作
        for teammate_idx, teammate_action in enumerate(teammate_actions):
            agent_idx = teammate_idx + 1
            
            try:
                teammate_action_enum = Action(teammate_action)
            except ValueError:
                teammate_action_enum = Action.NONE
                
            if self._process_agent_movement(agent_idx, teammate_action_enum):
                loading_agents.add(agent_idx)
                
        # 更新field以反映新的智能体位置
        self._update_field()
                
        # 处理食物加载行为
        loaded_foods = self._process_loading(loading_agents)
        self.reward_events.extend(loaded_foods)
            
        # 更新环境状态
        self.current_step += 1
        
        # 检查游戏是否结束
        food_sum = self.field.get_total_food_level()
        self._game_over = (food_sum == 0 or self._max_episode_steps <= self.current_step)
        
        # 更新有效动作
        self._gen_valid_actions()
        
        # 计算终局奖励
        if self._game_over:
            reward = self._calculate_final_reward()
        else:
            reward = 0.0
            
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

        # 初始化智能体设置
        ego = agents[0] if len(agents) > 0 else None
        teammates = agents[1:] if len(agents) > 1 else []
        
        # 初始化动作缓冲区
        ego_actions_buff = deque(maxlen=50)
        teammate_actions_buffs = [deque(maxlen=50) for _ in range(len(teammates))]

        # 渲染初始状态
        if render:
            self.render()
            time.sleep(sleep_time)
        
        # 逐步执行，直到回合结束
        while not done:
            # 获取ego智能体的动作
            ego_action = self._get_ego_action(ego, obss, ego_actions_buff, is_training)
            
            # 获取teammate智能体的动作列表
            teammate_actions = self._get_teammate_actions(teammates, obss, teammate_actions_buffs, is_training)
            
            # 执行动作并获取结果
            next_obss, reward, done, _, _ = self.__step(ego_action, teammate_actions)
            
            # 记录轨迹
            if is_training:
                trajectories.append([obss[0], ego_action, next_obss[0], done])
            
            # 记录终局奖励
            final_reward = reward if done else 0
            
            # 更新观察
            obss = next_obss

            if render:
                self.render()
                time.sleep(sleep_time)

        # 添加轨迹到经验缓冲区
        if is_training and ego and hasattr(ego, 'add_traj2buffer'):
            for ts in trajectories:
                ego.add_traj2buffer([
                    ts[0],
                    ts[1],
                    final_reward,
                    ts[2],
                    ts[3]
                ])
        
        return trajectories, final_reward, self.current_step

    def _get_ego_action(self, ego, obss, ego_actions_buff, is_training):
        """获取ego智能体的动作"""
        if not ego:
            return self._validate_single_action(0, Action.NONE.value)
            
        # 获取ego的有效动作
        valid_actions = [action.value for action in self._valid_actions[0]]
        
        # ego智能体选择动作
        action = ego.select_action({'obs': obss[0], 'actions': valid_actions}, is_training)
        
        # 检测重复动作
        if self._repeated_actions_detected(action, ego_actions_buff):
            other_valid_actions = [a for a in valid_actions if a != action]
            if other_valid_actions:
                action = np.random.choice(other_valid_actions)
        
        ego_actions_buff.append(action)
        
        # 验证ego动作
        return self._validate_single_action(0, action)

    def _get_teammate_actions(self, teammates, obss, teammate_actions_buffs, is_training):
        """获取所有teammate智能体的动作"""
        teammate_actions = []
        
        for i in range(1, self.num_agents):
            teammate_idx = i - 1
            
            if teammate_idx < len(teammates):
                teammate = teammates[teammate_idx]
                
                # 获取teammate的有效动作
                valid_actions = [action.value for action in self._valid_actions[i]]
                
                # teammate智能体选择动作
                action = teammate.select_action({'obs': obss[i], 'actions': valid_actions}, is_training)
                
                # 检测重复动作
                if self._repeated_actions_detected(action, teammate_actions_buffs[teammate_idx]):
                    other_valid_actions = [a for a in valid_actions if a != action]
                    if other_valid_actions:
                        action = np.random.choice(other_valid_actions)
                
                teammate_actions_buffs[teammate_idx].append(action)
                
                # 验证teammate动作
                teammate_action = self._validate_single_action(i, action)
            else:
                # 没有对应的teammate智能体，使用默认动作
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
        """检测动作是否出现重复"""
        if not actions_buff:
            return False
        
        if len(actions_buff) >= 3 and action != 5:
            # 超过3次连续相同动作视为重复
            if action == actions_buff[-1] == actions_buff[-2] == actions_buff[-3]:
                return True
                
            if len(actions_buff) >= 4:
                # 检测2步循环
                if action == actions_buff[-2] and actions_buff[-1] == actions_buff[-3]:
                    return True
                
                # 检测3步循环
                if len(actions_buff) >= 6:
                    if (action == actions_buff[-3] and 
                        actions_buff[-1] == actions_buff[-4] and 
                        actions_buff[-2] == actions_buff[-5]):
                        return True
                    
        return False

    #==============================
    # 终局奖励计算
    #==============================
    
    def _calculate_final_reward(self):
        """计算终局奖励"""
        reward = 0.0
        
        # 基础奖励
        food_sum = self.field.get_total_food_level()
        if food_sum == 0:
            reward = 1.0
        else:
            success_rate = (self._food_spawned - food_sum) / self._food_spawned if self._food_spawned > 0 else 0
            reward = success_rate * 0.5 if success_rate > 0 else -1.0
            
        # 吸引力奖励
        if self.agent_positions_history and self.food_positions_history:
            attraction_reward = self._calculate_attraction_reward()
            reward += attraction_reward
        
        # 步数奖励
        if reward > 0:
            step_efficiency = np.exp(-self.decay_rate * self.current_step)
            reward += reward * self.step_reward_factor * step_efficiency
            
        return reward

    def _calculate_attraction_reward(self):
        """计算吸引力奖励"""
        total_distance_change = 0
        valid_steps = 0
        
        for i in range(1, len(self.agent_positions_history)):
            current_foods = self.food_positions_history[i]
            prev_foods = self.food_positions_history[i-1]
            
            if len(current_foods) == 0 or len(current_foods) != len(prev_foods):
                continue
                
            agent_pos = self.agent_positions_history[i][0]
            prev_agent_pos = self.agent_positions_history[i-1][0]
            
            if len(current_foods) > 0:
                current_min_dist = min(abs(agent_pos[0] - f[0]) + abs(agent_pos[1] - f[1]) for f in current_foods)
                prev_min_dist = min(abs(prev_agent_pos[0] - f[0]) + abs(prev_agent_pos[1] - f[1]) for f in prev_foods)
                total_distance_change += prev_min_dist - current_min_dist
                valid_steps += 1
        
        if valid_steps == 0:
            return 0.0
            
        avg_distance_change = total_distance_change / valid_steps
        attraction_reward = 2 / (1 + np.exp(-avg_distance_change)) - 1
        
        self.agent_positions_history.clear()
        self.food_positions_history.clear()
        
        return attraction_reward * self.attraction_reward_factor

    #==============================
    # 渲染方法
    #==============================
    
    def render(self):
        """渲染环境"""
        if not self._rendering_initialized:
            from .rendering import Viewer
            self.viewer = Viewer(self.field.shape)
            self._rendering_initialized = True
        
        # 创建兼容的渲染数据
        render_env = type('RenderEnv', (), {
            'field': self._get_field_int_representation(),
            'agent_positions': self.agent_positions,
            'agent_levels': self.agent_levels
        })()
        
        return self.viewer.render(render_env, return_rgb_array=self.render_mode == "rgb_array")

    def close(self):
        """关闭环境"""
        if self.viewer:
            self.viewer.close() 