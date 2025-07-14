# LBForaging - Level-Based Foraging Environment

一个基于Gymnasium的多智能体觅食环境，支持协作和竞争策略。本项目使用先进的Field类架构来集中管理环境状态，提供高效的智能体-环境交互。

## 🌟 主要特性

- **多智能体协作**: 支持2-10个智能体的协作或竞争
- **等级机制**: 智能体和食物都有等级，需要合适的等级组合才能获取食物  
- **Field类架构**: 集中的环境状态管理，高效的位置查询和状态更新
- **灵活配置**: 丰富的参数配置选项，适应不同研究需求
- **兼容Gymnasium**: 完全兼容Gymnasium接口，易于集成

## 📦 安装

```bash
git clone <repository-url>
cd nfsp-lbforaging
pip install -e .
```

## 🚀 快速开始

### 方法1: 使用gym.make()创建预定义环境（推荐）

```python
import gymnasium as gym
import lbforaging  # 导入以注册环境

# 使用标准的gym.make()创建环境
env = gym.make("Foraging-8x8-2p-3f-coop-v3")

# 重置环境
observations, info = env.reset()
print(f"智能体数量: {env.num_agents}")
print(f"Field尺寸: {env.field.field_size}")

# 运行环境
done = False
while not done:
    actions = [env.action_space[i].sample() for i in range(env.num_agents)]
    observations, reward, done, truncated, info = env.step(actions)

env.close()
```

### 方法2: 直接创建自定义环境

```python
import lbforaging

# 创建自定义环境
env = lbforaging.create_env(
    num_agents=3,
    field_size=(10, 10),
    max_num_food=5,
    force_coop=True,
    max_episode_steps=100
)
```

### 方法3: 使用ForagingEnv类

```python
from lbforaging import ForagingEnv

env = ForagingEnv(
    num_agents=2,
    min_player_level=1,
    max_player_level=3,
    min_food_level=1,
    max_food_level=3,
    field_size=(8, 8),
    max_num_food=3,
    sight=2,
    max_episode_steps=50,
    force_coop=False
)
```

## 🎛️ 环境参数详解

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_agents` | int | 2 | 智能体数量 (2-10) |
| `field_size` | tuple | (8, 8) | 环境尺寸 (rows, cols) |
| `max_num_food` | int | 3 | 最大食物数量 |
| `sight` | int | 2 | 智能体视野范围 |
| `max_episode_steps` | int | 50 | 最大回合步数 |

### 等级设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `min_player_level` | int/list | 1 | 智能体最小等级 |
| `max_player_level` | int/list | 2 | 智能体最大等级 |
| `min_food_level` | int/list | 1 | 食物最小等级 |
| `max_food_level` | int/list | 2 | 食物最大等级 |

### 游戏机制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `force_coop` | bool | False | 是否强制协作（食物等级 = 最大智能体等级） |
| `observe_agent_levels` | bool | True | 观测中是否包含其他智能体等级 |

### 奖励设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `step_reward_factor` | float | 0.1 | 步数效率奖励因子 |
| `attraction_reward_factor` | float | 0.5 | 吸引力奖励因子 |
| `decay_rate` | float | 0.01 | 步数奖励衰减率 |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `render_mode` | str | None | 渲染模式 ("human", "rgb_array", None) |
| `seed` | int | None | 随机种子 |

## 🎮 环境使用示例

### 基本使用流程

```python
import lbforaging
import numpy as np

# 创建环境
env = lbforaging.create_env(
    num_agents=2,
    field_size=(8, 8),
    max_num_food=3,
    max_episode_steps=100
)

# 重置环境
observations, info = env.reset()
done = False

while not done:
    # 为每个智能体选择动作
    actions = []
    for i in range(env.num_agents):
        # 获取有效动作（如果需要）
        valid_actions = [action.value for action in env._valid_actions[i]]
        # 随机选择动作
        action = np.random.choice(valid_actions)
        actions.append(action)
    
    # 执行动作
    next_observations, reward, done, truncated, info = env.step(actions)
    
    # 处理结果
    print(f"Reward: {reward}, Done: {done}")
    observations = next_observations

env.close()
```

### 动作空间

每个智能体有6种可能的动作：

```python
from lbforaging.foraging import Action

# 动作定义
NONE = 0    # 不动
NORTH = 1   # 向北移动
SOUTH = 2   # 向南移动  
WEST = 3    # 向西移动
EAST = 4    # 向东移动
LOAD = 5    # 加载相邻食物
```

### 观测空间

观测向量包含：
- **食物信息**: 每个食物的相对位置和等级 `[rel_y, rel_x, level]`
- **智能体信息**: 其他智能体的相对位置和等级 `[rel_y, rel_x, level]`

```python
# 观测结构示例 (max_num_food=3, num_agents=2, observe_agent_levels=True)
observation = [
    # 食物1: [相对y, 相对x, 等级] 
    1, 2, 3,
    # 食物2: [相对y, 相对x, 等级]
    -1, -1, 0,  # -1表示不存在
    # 食物3: [相对y, 相对x, 等级]
    -1, -1, 0,
    # 其他智能体1: [相对y, 相对x, 等级]
    2, -1, 2,
    # 其他智能体2: [相对y, 相对x, 等级]  
    -1, -1, 0   # 不在视野内
]
```

## 🔧 Field类 - 高级交互

Field类是环境状态管理的核心，提供了丰富的API用于智能体与环境的交互。

### Field类结构

```python
from lbforaging import Field, FieldType, FieldPoint

# 创建Field实例
field = Field((10, 10))
```

### 核心数据结构

```python
# Field类包含两个核心数据结构：
field.field_points   # FieldPoint对象的二维数组
field.agent_ids      # 整数二维数组 (-1=空位, >=0=智能体ID)
field.agent_positions # 智能体位置映射 {agent_id: (row, col)}
```

### 智能体操作

```python
# 放置智能体
success = field.place_agent(agent_id=0, row=1, col=1, level=2)

# 移动智能体
success = field.move_agent(agent_id=0, new_row=2, new_col=1)

# 获取智能体信息
position = field.get_agent_position(agent_id=0)  # 返回 (row, col) 或 None
level = field.get_agent_level(agent_id=0)        # 返回等级
```

### 食物操作

```python
# 放置食物
success = field.place_food(row=3, col=3, level=2)

# 移除食物
success = field.remove_food(row=3, col=3)

# 获取所有食物信息
food_infos = field.get_all_food_infos()  # [(row, col, level), ...]
total_food = field.get_total_food_level()
```

### 位置查询

```python
# 基本查询
is_empty = field.is_empty(row=1, col=1)
is_agent = field.is_agent_at(row=1, col=1)
is_food = field.is_food_at(row=1, col=1)

# 获取位置上的内容
agent_id = field.get_agent_id(row=1, col=1)  # -1表示无智能体
field_point = field.get_field_point(row=1, col=1)

# 区域查询
adjacent_pos = field.get_adjacent_positions(row=1, col=1)
agents_in_area = field.get_agents_in_area(center_row=5, center_col=5, radius=2)
```

### 实用方法

```python
# 获取所有智能体位置
all_agent_positions = field.get_all_agent_positions()  # {agent_id: (row, col)}

# 转换为整数数组（兼容性）
int_array = field.to_int_array()

# 字符串表示
print(field)  # 打印可视化的field状态
```

## 🤖 智能体交互示例

### 自定义智能体类

```python
class MyAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
    
    def select_action(self, observation_dict, is_training=False):
        obs = observation_dict['obs']
        valid_actions = observation_dict['actions']
        
        # 简单策略：优先朝最近的食物移动
        # 解析观测中的食物信息
        foods = []
        for i in range(0, len(obs), 3):
            if obs[i] != -1:  # 存在食物
                foods.append((obs[i], obs[i+1], obs[i+2]))
        
        if foods:
            # 朝最近的食物移动
            closest_food = min(foods, key=lambda f: abs(f[0]) + abs(f[1]))
            if closest_food[0] > 0:
                return 2 if 2 in valid_actions else 0  # SOUTH
            elif closest_food[0] < 0:
                return 1 if 1 in valid_actions else 0  # NORTH
            elif closest_food[1] > 0:
                return 4 if 4 in valid_actions else 0  # EAST
            elif closest_food[1] < 0:
                return 3 if 3 in valid_actions else 0  # WEST
            else:
                return 5 if 5 in valid_actions else 0  # LOAD
        
        return np.random.choice(valid_actions)
```

### 使用智能体运行环境

```python
# 创建智能体
agents = [MyAgent(i) for i in range(env.num_agents)]

# 运行环境
trajectories, final_reward, steps = env.run(
    agents=agents,
    is_training=True,
    render=True,
    sleep_time=0.5
)

print(f"Episode completed in {steps} steps with reward {final_reward}")
```

### 访问Field状态

```python
# 在运行过程中访问Field状态
def analyze_environment(env):
    field = env.field
    
    print("=== 环境状态分析 ===")
    print(f"Field尺寸: {field.field_size}")
    print(f"智能体位置: {field.get_all_agent_positions()}")
    print(f"食物信息: {field.get_all_food_infos()}")
    print(f"总食物等级: {field.get_total_food_level()}")
    
    # 检查智能体周围情况
    for agent_id, pos in field.get_all_agent_positions().items():
        if pos:
            row, col = pos
            adjacent = field.get_adjacent_positions(row, col)
            print(f"智能体{agent_id}周围位置: {adjacent}")
            
            # 检查相邻是否有食物
            for adj_row, adj_col in adjacent:
                if field.is_food_at(adj_row, adj_col):
                    food_level = field.get_field_point(adj_row, adj_col).get_level()
                    print(f"  相邻食物: 位置({adj_row}, {adj_col}), 等级{food_level}")

# 使用分析函数
analyze_environment(env)
```

## 🏆 奖励机制

环境使用复合奖励系统：

1. **基础完成奖励**: 收集完所有食物获得1.0奖励
2. **部分完成奖励**: 根据收集比例给予0.5倍比例奖励  
3. **失败惩罚**: 未完成任务扣除-1.0奖励
4. **吸引力奖励**: 根据智能体朝食物移动的趋势给予额外奖励
5. **效率奖励**: 根据完成步数给予效率奖励

```python
# 奖励计算示例
final_reward = base_reward + attraction_reward + efficiency_reward

# 其中：
# base_reward = 1.0 (全部完成) 或 success_rate * 0.5 (部分完成) 或 -1.0 (失败)
# attraction_reward = sigmoid(average_distance_change) * attraction_reward_factor
# efficiency_reward = base_reward * step_reward_factor * exp(-decay_rate * steps)
```

## 🎯 预定义环境ID格式

环境ID遵循以下格式：
```
Foraging{partial_obs}-{size}x{size}-{players}p-{foods}f{coop}{max_food_level}{penalty}-v3
```

示例：
- `Foraging-8x8-2p-3f-v3`: 8x8环境，2个智能体，3个食物
- `Foraging-8x8-2p-3f-coop-v3`: 强制协作模式
- `Foraging-2s-8x8-2p-3f-v3`: 部分观测（视野=2）

## 🔍 调试和可视化

### 环境状态可视化

```python
# 打印Field状态
print(env.field)
# 输出示例：
# A0 -- F2 -- --
# -- F1 -- -- --  
# -- -- A1 -- F3
# -- -- -- -- --
# -- -- -- -- --

# 渲染环境（如果支持）
env.render()
```

### 调试信息

```python
# 获取详细的环境信息
print(f"当前步数: {env.current_step}")
print(f"游戏结束: {env.game_over}")
print(f"智能体位置: {env.agent_positions}")
print(f"智能体等级: {env.agent_levels}")
print(f"有效动作: {env._valid_actions}")
```

## 📚 进阶用法

### 自定义奖励函数

```python
class CustomForagingEnv(ForagingEnv):
    def _calculate_final_reward(self):
        # 自定义奖励逻辑
        base_reward = super()._calculate_final_reward()
        
        # 添加协作奖励
        cooperation_bonus = self._calculate_cooperation_bonus()
        
        return base_reward + cooperation_bonus
    
    def _calculate_cooperation_bonus(self):
        # 计算协作奖励的逻辑
        return 0.1 * len(self.reward_events)
```

### 环境包装器

```python
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, actions):
        obs, reward, done, truncated, info = self.env.step(actions)
        
        # 添加步骤奖励塑造
        shaped_reward = reward - 0.01  # 步骤惩罚
        
        return obs, shaped_reward, done, truncated, info

# 使用包装器
env = RewardShapingWrapper(lbforaging.create_env())
```

## 🛠️ 开发和贡献

### 环境扩展

要扩展环境功能，可以：

1. 继承`ForagingEnv`类
2. 重写相关方法
3. 添加新的Field操作
4. 自定义观测和动作空间

### 测试

```python
# 运行测试脚本
python agent_field_mapping_example.py
```

## 📖 参考文献

如果您在研究中使用了此环境，请引用：

```bibtex
@software{lbforaging2024,
  title={LBForaging: Level-Based Multi-Agent Foraging Environment},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/lbforaging}
}
```

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 支持和社区

- **Issues**: [GitHub Issues](https://github.com/your-repo/lbforaging/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/lbforaging/discussions)
- **Email**: your.email@example.com

---

**Happy Foraging! 🌾🤖**

