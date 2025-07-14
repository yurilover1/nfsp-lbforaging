from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


class FieldType(Enum):
    EMPTY = 0
    AGENT = 1
    FOOD = 2


@dataclass
class FieldPoint:
    """
    表示游戏field中的一个点
    
    Attributes:
        type: 点的类型 (EMPTY, AGENT, FOOD)
        level: 等级信息 (None表示无等级，int表示具体等级)
        agent_id: 智能体索引 (仅当type为AGENT时有效，None表示非智能体)
        position: 位置信息 (row, col)
    """
    type: FieldType
    level: Optional[int] = None
    agent_id: Optional[int] = None
    position: Optional[tuple] = None
    
    def __post_init__(self):
        """确保level和agent_id的一致性"""
        if self.type == FieldType.EMPTY:
            self.level = None
            self.agent_id = None
        elif self.type == FieldType.AGENT:
            if self.level is None:
                self.level = 1  # 默认等级
            # agent_id应该由外部设置
        elif self.type == FieldType.FOOD:
            if self.level is None:
                self.level = 1  # 默认等级
            self.agent_id = None
    
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
    
    def get_agent_id(self) -> Optional[int]:
        """获取智能体ID，仅当为智能体时返回有效值"""
        return self.agent_id if self.is_agent() else None
    
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
    
    该类维护两个核心数据结构：
    1. field_points: FieldPoint对象的二维数组，存储详细的位置信息
    2. agent_ids: 整数二维数组，用于快速查询智能体ID（-1表示空位，>=0表示智能体ID）
    
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
        
        # 智能体ID的整数二维数组（-1表示空位，>=0表示智能体ID）
        self.agent_ids = np.full(field_size, -1, dtype=np.int32)
        
        # 初始化所有位置为空
        self._initialize_empty_field()
        
        # 智能体位置映射，用于快速查找
        self.agent_positions = {}  # {agent_id: (row, col)}
    
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
    
    def get_agent_id(self, row: int, col: int) -> int:
        """获取指定位置的智能体ID（-1表示空位）"""
        if not self._is_valid_position(row, col):
            raise IndexError(f"位置 ({row}, {col}) 超出field范围")
        return self.agent_ids[row, col]
    
    def is_empty(self, row: int, col: int) -> bool:
        """检查指定位置是否为空"""
        return self.get_agent_id(row, col) == -1 and self.get_field_point(row, col).is_empty()
    
    def is_agent_at(self, row: int, col: int) -> bool:
        """检查指定位置是否有智能体"""
        return self.get_agent_id(row, col) >= 0
    
    def is_food_at(self, row: int, col: int) -> bool:
        """检查指定位置是否有食物"""
        return self.get_field_point(row, col).is_food()
    
    def get_agent_position(self, agent_id: int) -> Optional[Tuple[int, int]]:
        """获取智能体的位置"""
        return self.agent_positions.get(agent_id, None)
    
    def get_agent_level(self, agent_id: int) -> int:
        """获取智能体的等级"""
        pos = self.get_agent_position(agent_id)
        if pos is None:
            return 0
        return self.get_field_point(*pos).get_level()
    
    def place_agent(self, agent_id: int, row: int, col: int, level: int = 1) -> bool:
        """
        在指定位置放置智能体
        
        Args:
            agent_id: 智能体ID
            row, col: 位置坐标
            level: 智能体等级
            
        Returns:
            bool: 是否成功放置
        """
        if not self._is_valid_position(row, col):
            return False
        
        if not self.is_empty(row, col):
            return False
        
        # 如果智能体已存在其他位置，先清除
        if agent_id in self.agent_positions:
            old_pos = self.agent_positions[agent_id]
            self.clear_position(*old_pos)
        
        # 放置智能体
        self.field_points[row, col] = FieldPoint(
            type=FieldType.AGENT,
            level=level,
            agent_id=agent_id,
            position=(row, col)
        )
        self.agent_ids[row, col] = agent_id
        self.agent_positions[agent_id] = (row, col)
        
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
        
        return True
    
    def move_agent(self, agent_id: int, new_row: int, new_col: int) -> bool:
        """
        移动智能体到新位置
        
        Args:
            agent_id: 智能体ID
            new_row, new_col: 新位置坐标
            
        Returns:
            bool: 是否成功移动
        """
        if not self._is_valid_position(new_row, new_col):
            return False
        
        if not self.is_empty(new_row, new_col):
            return False
        
        old_pos = self.get_agent_position(agent_id)
        if old_pos is None:
            return False
        
        # 获取智能体信息
        old_row, old_col = old_pos
        agent_level = self.get_field_point(old_row, old_col).get_level()
        
        # 清除旧位置
        self.clear_position(old_row, old_col)
        
        # 放置到新位置
        return self.place_agent(agent_id, new_row, new_col, agent_level)
    
    def clear_position(self, row: int, col: int):
        """清空指定位置"""
        if not self._is_valid_position(row, col):
            return
        
        # 如果是智能体，从位置映射中移除
        agent_id = self.get_agent_id(row, col)
        if agent_id >= 0:
            if agent_id in self.agent_positions:
                del self.agent_positions[agent_id]
        
        # 重置为空
        self.field_points[row, col] = FieldPoint(
            type=FieldType.EMPTY,
            position=(row, col)
        )
        self.agent_ids[row, col] = -1
    
    def remove_food(self, row: int, col: int) -> bool:
        """
        移除指定位置的食物
        
        Returns:
            bool: 是否成功移除（该位置确实有食物）
        """
        if not self._is_valid_position(row, col):
            return False
        
        if not self.is_food_at(row, col):
            return False
        
        self.clear_position(row, col)
        return True
    
    def get_adjacent_positions(self, row: int, col: int) -> list:
        """获取相邻位置列表"""
        adjacent = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_position(new_row, new_col):
                adjacent.append((new_row, new_col))
        return adjacent
    
    def get_agents_in_area(self, center_row: int, center_col: int, radius: int) -> list:
        """获取指定区域内的所有智能体ID列表"""
        agents = []
        for row in range(max(0, center_row - radius), 
                        min(self.rows, center_row + radius + 1)):
            for col in range(max(0, center_col - radius), 
                           min(self.cols, center_col + radius + 1)):
                agent_id = self.get_agent_id(row, col)
                if agent_id >= 0:
                    agents.append(agent_id)
        return agents
    
    def get_all_agent_positions(self) -> dict:
        """获取所有智能体的位置"""
        return self.agent_positions.copy()
    
    def get_all_food_infos(self) -> list:
        """获取所有食物的位置和等级"""
        foods = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.is_food_at(i, j):
                    level = self.get_field_point(i, j).get_level()
                    foods.append((i, j, level))
        return foods
    
    def get_total_food_level(self) -> int:
        """获取总食物等级"""
        total = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if self.is_food_at(i, j):
                    total += self.get_field_point(i, j).get_level()
        return total
    
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
                agent_id = self.agent_ids[i, j]
                if agent_id >= 0:
                    row_str.append(f"A{agent_id}")
                elif self.is_food_at(i, j):
                    level = self.get_field_point(i, j).get_level()
                    row_str.append(f"F{level}")
                else:
                    row_str.append("--")
            result.append(" ".join(row_str))
        return "\n".join(result) 