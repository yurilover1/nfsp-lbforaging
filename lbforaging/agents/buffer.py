import numpy as np
import random
from collections import deque


class replay_buffer:
    """标准经验回放缓冲区"""
    
    def __init__(self, buffer_size):
        """初始化参数和缓冲区"""
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def store(self, state, action, reward, next_state, done):
        """存储经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """从缓冲区随机采样一批经验"""
        # 确保有足够的样本
        batch_size = min(batch_size, len(self.buffer))
        
        # 随机采样
        batch = random.sample(self.buffer, batch_size)
        
        # 分解批次
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)


class n_step_replay_buffer:
    """多步(n-step)经验回放缓冲区"""
    
    def __init__(self, buffer_size, n_step, gamma):
        """初始化参数和缓冲区"""
        self.buffer_size = buffer_size
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=buffer_size)
        self.n_step_buffer = deque(maxlen=n_step)
    
    def _get_n_step_info(self):
        """计算n步累积奖励和n步后的状态"""
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        
        # 计算n步后的奖励（考虑折扣）
        for i in range(len(self.n_step_buffer) - 2, -1, -1):
            r, s, d = self.n_step_buffer[i][-3:]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state, done = s, d
        
        return reward, next_state, done
    
    def store(self, state, action, reward, next_state, done):
        """存储经验到缓冲区"""
        # 添加到n步缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # 如果n步缓冲区未满，则返回
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # 获取n步信息
        reward, next_state, done = self._get_n_step_info()
        
        # 存储n步经验到主缓冲区
        state, action = self.n_step_buffer[0][:2]
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """从缓冲区随机采样一批经验"""
        # 确保有足够的样本
        batch_size = min(batch_size, len(self.buffer))
        
        # 随机采样
        batch = random.sample(self.buffer, batch_size)
        
        # 分解批次
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        """返回主缓冲区当前大小"""
        return len(self.buffer)


class reservoir_buffer:
    """水库采样缓冲区，用于SL部分，能确保均匀采样"""
    
    def __init__(self, buffer_size):
        """初始化参数和缓冲区"""
        self.buffer_size = buffer_size
        self.buffer = []
        self.count = 0
    
    def store(self, state, action):
        """使用水库抽样算法存储经验"""
        self.count += 1
        
        # 如果缓冲区未满，直接添加
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action))
        else:
            # 使用水库抽样算法决定是否替换
            idx = np.random.randint(0, self.count)
            if idx < self.buffer_size:
                self.buffer[idx] = (state, action)
    
    def sample(self, batch_size):
        """从缓冲区随机采样一批经验"""
        # 确保有足够的样本
        batch_size = min(batch_size, len(self.buffer))
        
        # 随机采样
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # 分解批次
        states, actions = zip(*batch)
        
        return np.array(states), np.array(actions)
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer) 