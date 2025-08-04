from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class FieldType(Enum):
    EMPTY = 0
    AGENT = 1
    FOOD = 2


@dataclass
class Player:
    """
    玩家类，作为智能体和环境交互的中介
    
    Attributes:
        player_id: 玩家ID
        level: 玩家等级
        position: 玩家位置 (row, col)
        score: 玩家得分
        reward: 玩家奖励
        is_self: 是否为主玩家
        agent: 关联的智能体对象
        prev_position: 前一步位置，用于吸引力奖励计算
    """
    player_id: int
    level: int = 1
    position: Tuple[int, int] = (0, 0)
    score: float = 0.0
    reward: float = 0.0
    is_self: bool = False
    agent: Optional[object] = None
    prev_position: Optional[Tuple[int, int]] = None
    
    def __post_init__(self):
        """初始化前一步位置"""
        if self.prev_position is None:
            self.prev_position = self.position
    
    def select_action(self, observation, is_training=False):
        """
        选择动作的接口，委托给关联的智能体
        
        Args:
            observation: 观察信息，可以是obs或包含actions的字典
            is_training: 是否为训练模式
            
        Returns:
            选择的动作
        """
        # 确保observation包含有效动作列表
        if isinstance(observation, dict) and 'actions' in observation:
            valid_actions = observation['actions']
            obs_data = observation['obs']
        else:
            # 如果没有提供有效动作，使用默认的0-5范围
            valid_actions = list(range(6))  # 0-5对应NONE到LOAD
            obs_data = observation
        
        # 传递包含有效动作的完整observation
        action = self.agent.select_action({'obs': obs_data, 'actions': valid_actions}, is_training)
        return action
    
    def add_reward(self, reward):
        """添加奖励"""
        self.reward += reward
        self.score += reward
    
    def reset_reward(self):
        """重置奖励"""
        self.reward = 0.0
    
    def update_position(self, new_position):
        """更新位置，同时保存前一步位置"""
        self.prev_position = self.position
        self.position = new_position
    
    def get_position(self):
        """获取当前位置"""
        return self.position
    
    def get_prev_position(self):
        """获取前一步位置"""
        return self.prev_position
    
    def set_prev_position(self, prev_pos):
        """设置前一步位置"""
        self.prev_position = prev_pos
    
    def get_level(self):
        """获取等级"""
        return self.level
    
    def get_score(self):
        """获取得分"""
        return self.score
    
    def get_reward(self):
        """获取奖励"""
        return self.reward
    
    def __str__(self):
        return f"Player(id={self.player_id}, level={self.level}, pos={self.position}, score={self.score})"
    
    def __repr__(self):
        return self.__str__()


@dataclass
class FieldPoint:
    """
    表示游戏field中的一个点
    
    Attributes:
        type: 点的类型 (EMPTY, AGENT, FOOD)
        level: 等级信息 (None表示无等级，int表示具体等级)
        player: 玩家对象 (仅当type为AGENT时有效，None表示非智能体)
        position: 位置信息 (row, col)
    """
    type: FieldType
    level: Optional[int] = None
    player: Optional[Player] = None
    position: Optional[tuple] = None
    
    def __post_init__(self):
        """确保level和player的一致性"""
        if self.type == FieldType.EMPTY:
            self.level = None
            self.player = None
        elif self.type == FieldType.AGENT:
            if self.level is None:
                self.level = 1  # 默认等级
            # player 应由外部设置，这里不覆盖
        elif self.type == FieldType.FOOD:
            if self.level is None:
                self.level = 1  # 默认等级
            self.player = None
    
    def is_empty(self) -> bool:
        """检查是否为空"""
        return self.type == FieldType.EMPTY
    
    def is_agent(self) -> bool:
        """检查是否为智能体"""
        return self.type == FieldType.AGENT
    
    def is_food(self) -> bool:
        """检查是否为食物"""
        return self.type == FieldType.FOOD
    
    def get_level(self) -> int:
        """获取等级，空位置返回0"""
        return self.level if self.level is not None else 0
    
    def get_player(self) -> Optional[Player]:
        """获取玩家对象，仅当为智能体时返回有效值"""
        return self.player if self.is_agent() else None
    
    def get_position(self) -> Optional[tuple]:
        """获取位置信息"""
        return self.position
    
    def to_int(self) -> int:
        """转换为整数表示，用于兼容性"""
        if self.type == FieldType.EMPTY:
            return 0
        elif self.type == FieldType.AGENT:
            return -self.level
        elif self.type == FieldType.FOOD:
            return self.level
        return 0


class Field:
    """
    游戏field的集中管理类
    
    该类维护三个核心数据结构：
    1. field_points: FieldPoint对象的二维数组，存储详细的位置信息
    2. player_ids: 整数二维数组，用于快速查询玩家ID（-1表示空位，>=0表示玩家ID）
    3. food_positions: 食物位置映射，用于快速查找食物信息
    
    这种设计使得智能体动作对环境的影响能够集中展示和管理。
    """
    
    def __init__(self, field_size: Tuple[int, int]):
        """
        初始化Field
        
        Args:
            field_size: field的尺寸 (行数, 列数)
        """
        self.field_size = field_size
        self.rows, self.cols = field_size
        
        # FieldPoint对象的二维数组
        self.field_points = np.empty(field_size, dtype=object)
        
        # 玩家ID的整数二维数组（-1表示空位，>=0表示玩家ID）
        self.player_ids = np.full(field_size, -1, dtype=np.int32)
        
        # 初始化所有位置为空
        self._initialize_empty_field()
        
        # 玩家位置映射，用于快速查找
        self.player_positions = {}  # {player_id: (row, col)}
        
        # 食物位置映射，用于快速查找食物信息
        self.food_positions = {}  # {(row, col): level}
        
        # 食物拾取记录，用于吸引力奖励计算
        self.food_pickup_history = []  # [(step, food_pos, food_level), ...]
        
        # 吸引力奖励计算相关
        self._prev_food_positions = set()  # 前一步的食物位置
        self._attraction_reward_factor = 0.1  # 吸引力奖励因子
    
    def _initialize_empty_field(self):
        """初始化空field"""
        for i in range(self.rows):
            for j in range(self.cols):
                self.field_points[i, j] = FieldPoint(
                    type=FieldType.EMPTY, 
                    position=(i, j)
                )
    
    def get_field_point(self, row: int, col: int) -> FieldPoint:
        """获取指定位置的FieldPoint对象"""
        if not self._is_valid_position(row, col):
            raise IndexError(f"位置 ({row}, {col}) 超出field范围")
        return self.field_points[row, col]
    
    def get_player_id(self, row: int, col: int) -> int:
        """获取指定位置的玩家ID（-1表示空位）"""
        if not self._is_valid_position(row, col):
            raise IndexError(f"位置 ({row}, {col}) 超出field范围")
        return self.player_ids[row, col]
    
    def is_empty(self, row: int, col: int) -> bool:
        """检查指定位置是否为空"""
        return self.get_player_id(row, col) == -1 and self.get_field_point(row, col).is_empty()
    
    def is_player_at(self, row: int, col: int) -> bool:
        """检查指定位置是否有玩家"""
        return self.get_player_id(row, col) >= 0
    
    def is_food_at(self, row: int, col: int) -> bool:
        """检查指定位置是否有食物"""
        return self.get_field_point(row, col).is_food()
    
    def get_player_position(self, player_id: int) -> Optional[Tuple[int, int]]:
        """获取玩家的位置"""
        return self.player_positions.get(player_id, None)
    
    def get_player_level(self, player_id: int) -> int:
        """获取玩家的等级"""
        pos = self.get_player_position(player_id)
        if pos is None:
            return 0
        return self.get_field_point(*pos).get_level()
    
    def get_player_object(self, player_id: int) -> Optional[Player]:
        """获取玩家对象"""
        pos = self.get_player_position(player_id)
        if pos is None:
            return None
        return self.get_field_point(*pos).get_player()
    
    def place_player(self, player: Player, row: int, col: int) -> bool:
        """
        在指定位置放置玩家
        
        Args:
            player: 玩家对象
            row, col: 位置坐标
            
        Returns:
            bool: 是否成功放置
        """
        if not self._is_valid_position(row, col):
            return False
        
        if not self.is_empty(row, col):
            return False
        
        # 如果玩家已存在其他位置，先清除
        if player.player_id in self.player_positions:
            old_pos = self.player_positions[player.player_id]
            self.clear_position(*old_pos)
        
        # 更新玩家位置
        player.update_position((row, col))
        
        # 放置玩家
        self.field_points[row, col] = FieldPoint(
            type=FieldType.AGENT,
            level=player.level,
            player=player,
            position=(row, col)
        )
        self.player_ids[row, col] = player.player_id
        self.player_positions[player.player_id] = (row, col)
        
        return True
    
    def place_food(self, row: int, col: int, level: int = 1) -> bool:
        """
        在指定位置放置食物
        
        Args:
            row, col: 位置坐标
            level: 食物等级
            
        Returns:
            bool: 是否成功放置
        """
        if not self._is_valid_position(row, col):
            return False
        
        if not self.is_empty(row, col):
            return False
        
        # 放置食物
        self.field_points[row, col] = FieldPoint(
            type=FieldType.FOOD,
            level=level,
            position=(row, col)
        )
        
        # 更新食物位置映射
        self.food_positions[(row, col)] = level
        
        return True
    
    def move_player(self, player_id: int, new_row: int, new_col: int) -> bool:
        """
        移动玩家到新位置
        
        Args:
            player_id: 玩家ID
            new_row, new_col: 新位置坐标
            
        Returns:
            bool: 是否成功移动
        """
        if not self._is_valid_position(new_row, new_col):
            return False
        
        if not self.is_empty(new_row, new_col):
            return False
        
        old_pos = self.get_player_position(player_id)
        if old_pos is None:
            return False
        
        # 获取玩家对象
        old_row, old_col = old_pos
        player = self.get_field_point(old_row, old_col).get_player()
        if player is None:
            return False
        
        # 清除旧位置
        self.clear_position(old_row, old_col)
        
        # 更新玩家位置（这会自动保存前一步位置）
        player.update_position((new_row, new_col))
        
        # 放置到新位置
        return self.place_player(player, new_row, new_col)
    
    def clear_position(self, row: int, col: int):
        """清空指定位置"""
        if not self._is_valid_position(row, col):
            return
        
        # 如果是玩家，从位置映射中移除
        player_id = self.get_player_id(row, col)
        if player_id >= 0:
            if player_id in self.player_positions:
                del self.player_positions[player_id]
        
        # 如果是食物，从食物映射中移除
        if self.is_food_at(row, col):
            if (row, col) in self.food_positions:
                del self.food_positions[(row, col)]
        
        # 重置为空
        self.field_points[row, col] = FieldPoint(
            type=FieldType.EMPTY,
            position=(row, col)
        )
        self.player_ids[row, col] = -1
    
    def remove_food(self, row: int, col: int, step: int = 0) -> bool:
        """
        移除指定位置的食物
        
        Args:
            row, col: 位置坐标
            step: 当前步数，用于记录拾取历史
            
        Returns:
            bool: 是否成功移除（该位置确实有食物）
        """
        if not self._is_valid_position(row, col):
            return False
        
        if not self.is_food_at(row, col):
            return False
        
        # 记录食物拾取历史
        food_level = self.get_food_level(row, col)
        self.food_pickup_history.append((step, (row, col), food_level))
        
        self.clear_position(row, col)
        return True
    
    def get_food_pickup_history(self):
        """获取食物拾取历史"""
        return self.food_pickup_history.copy()
    
    def clear_food_pickup_history(self):
        """清空食物拾取历史"""
        self.food_pickup_history.clear()
    
    def update_attraction_reward(self, player: Player, step: int, episode: int = 0) -> float:
        """
        更新并计算当前步的吸引力奖励
        
        Args:
            player: 玩家对象
            step: 当前步数
            episode: 当前回合数，用于日志记录
            
        Returns:
            float: 当前步的吸引力奖励
        """
        if not player or player.prev_position is None:
            return 0.0
        
        # 获取当前食物位置
        current_food_positions = set(self.food_positions.keys())
        
        # 获取玩家当前位置和前一步位置
        current_pos = player.position
        prev_pos = player.prev_position
        
        # 计算吸引力奖励
        step_reward = 0.0
        
        # 检查食物数量是否保持不变（与原来环境逻辑一致）
        if (self._prev_food_positions and current_food_positions and 
            len(self._prev_food_positions) > 0 and 
            len(current_food_positions) == len(self._prev_food_positions)):
            
            # 计算到最近食物的距离变化
            current_min_dist = min(abs(current_pos[0] - f[0]) + abs(current_pos[1] - f[1]) 
                                 for f in current_food_positions)
            prev_min_dist = min(abs(prev_pos[0] - f[0]) + abs(prev_pos[1] - f[1]) 
                              for f in self._prev_food_positions)
            
            # 计算距离变化
            distance_change = prev_min_dist - current_min_dist
            
            # 使用sigmoid函数计算吸引力奖励
            if distance_change != 0:
                step_reward = 2 / (1 + np.exp(-distance_change)) - 1
                step_reward *= self._attraction_reward_factor
        
        # 更新前一步的食物位置
        self._prev_food_positions = current_food_positions.copy()
        
        # 记录到csv日志
        import os
        os.makedirs('logs', exist_ok=True)
        with open('logs/attraction_step_reward.csv', 'a') as f:
            f.write(f"{episode},{step},{step_reward},{current_pos},{list(current_food_positions)}\n")
        
        return step_reward
    
    def reset_attraction_state(self):
        """重置吸引力奖励相关状态"""
        self._prev_food_positions.clear()
    
    def set_attraction_reward_factor(self, factor: float):
        """设置吸引力奖励因子"""
        self._attraction_reward_factor = factor
    
    def get_adjacent_positions(self, row: int, col: int) -> list:
        """获取相邻位置列表"""
        adjacent = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_position(new_row, new_col):
                adjacent.append((new_row, new_col))
        return adjacent
    
    def get_players_in_area(self, center_row: int, center_col: int, radius: int) -> list:
        """获取指定区域内的所有玩家ID列表"""
        players = []
        for row in range(max(0, center_row - radius), 
                        min(self.rows, center_row + radius + 1)):
            for col in range(max(0, center_col - radius), 
                           min(self.cols, center_col + radius + 1)):
                player_id = self.get_player_id(row, col)
                if player_id >= 0:
                    players.append(player_id)
        return players
    
    def get_all_player_positions(self) -> dict:
        """获取所有玩家的位置"""
        return self.player_positions.copy()
    
    def get_all_players(self) -> list:
        """获取所有玩家对象"""
        players = []
        for player_id in self.player_positions:
            player = self.get_player_object(player_id)
            if player:
                players.append(player)
        return players
    
    def get_all_food_infos(self) -> list:
        """获取所有食物的位置和等级"""
        # 使用食物映射提高效率
        return [(row, col, level) for (row, col), level in self.food_positions.items()]
    
    def get_food_level(self, row: int, col: int) -> int:
        """
        获取指定位置食物的等级
        
        Args:
            row, col: 位置坐标
            
        Returns:
            int: 食物等级，如果不是食物返回0
        """
        if not self._is_valid_position(row, col):
            return 0
        if not self.is_food_at(row, col):
            return 0
        return self.get_field_point(row, col).get_level()
    
    def get_food_position(self, food_id: int) -> Optional[Tuple[int, int]]:
        """
        根据食物ID获取食物位置（保持向后兼容）
        
        Args:
            food_id: 食物ID（在食物映射中，使用位置作为键）
            
        Returns:
            Optional[Tuple[int, int]]: 食物位置，如果不存在返回None
        """
        # 由于食物映射使用位置作为键，这里需要特殊处理
        # 暂时返回None，因为食物没有ID概念
        return None
    
    def get_food_at_position(self, row: int, col: int) -> Optional[Tuple[int, int, int]]:
        """
        获取指定位置的食物信息
        
        Args:
            row, col: 位置坐标
            
        Returns:
            Optional[Tuple[int, int, int]]: (row, col, level) 或 None
        """
        if not self._is_valid_position(row, col):
            return None
        if not self.is_food_at(row, col):
            return None
        level = self.get_field_point(row, col).get_level()
        return (row, col, level)
    
    def get_all_food_positions(self) -> dict:
        """
        获取所有食物的位置映射
        
        Returns:
            dict: {(row, col): level} 格式的食物位置映射
        """
        return self.food_positions.copy()
    
    def get_food_count(self) -> int:
        """
        获取当前食物数量
        
        Returns:
            int: 食物总数
        """
        return len(self.food_positions)
    
    def is_food_available(self, row: int, col: int) -> bool:
        """
        检查指定位置是否有可用的食物
        
        Args:
            row, col: 位置坐标
            
        Returns:
            bool: 是否有食物
        """
        return (row, col) in self.food_positions
    
    def get_total_food_level(self) -> int:
        """获取总食物等级"""
        # 使用食物映射提高效率
        return sum(self.food_positions.values())
    
    def to_int_array(self) -> np.ndarray:
        """转换为整数数组表示（用于兼容性）"""
        int_field = np.zeros(self.field_size, dtype=np.int32)
        for i in range(self.rows):
            for j in range(self.cols):
                int_field[i, j] = self.field_points[i, j].to_int()
        return int_field
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """检查位置是否有效"""
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def __str__(self) -> str:
        """字符串表示"""
        result = []
        for i in range(self.rows):
            row_str = []
            for j in range(self.cols):
                player_id = self.player_ids[i, j]
                if player_id >= 0:
                    row_str.append(f"P{player_id}")
                elif self.is_food_at(i, j):
                    level = self.get_field_point(i, j).get_level()
                    row_str.append(f"F{level}")
                else:
                    row_str.append("--")
            result.append(" ".join(row_str))
        return "\n".join(result) 