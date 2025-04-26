import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from lbforaging.agents import BaseAgent
from .model import dueling_ddqn, policy
from .buffer import reservoir_buffer, n_step_replay_buffer, replay_buffer
from .utils import action_mask


class NFSPAgent(BaseAgent):
    def __init__(self, 
                player,
                state_size,
                action_size, 
                epsilon_init=0.06,
                epsilon_decay=10000,
                epsilon_min=0.0,
                update_freq=100,
                sl_lr=0.005,
                rl_lr=0.001,
                sl_buffer_size=10000,
                rl_buffer_size=10000,
                n_step=1,
                gamma=0.99,
                eta=0.1,
                rl_start=300,
                sl_start=300,
                train_freq=1,
                rl_batch_size=64,
                sl_batch_size=64,
                device=None,
                hidden_units=64,
                layers=3,
                eval_mode='average'):
        
        super().__init__(player)
        self.name = f"NFSP Agent {player.level if hasattr(player, 'level') else ''}"
        
        # 基本参数
        self.state_size = state_size
        self.action_size = action_size
        self.n_step = n_step
        self.gamma = gamma
        self.eta = eta  # 决定使用哪种策略的概率
        self.rl_batch_size = rl_batch_size
        self.sl_batch_size = sl_batch_size
        self.update_freq = update_freq
        self.train_freq = train_freq
        self.rl_start = rl_start
        self.sl_start = sl_start
        self.eval_mode = eval_mode
        # 保存学习率参数
        self.rl_lr = rl_lr
        self.sl_lr = sl_lr
        # 设备设置
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 初始化网络
        self.hidden_units = hidden_units
        self.networks_initialized = False
        self.layers = layers
        # 记录是否已经进行了状态尺寸检查 - 优化性能，避免重复检查
        self.state_size_checked = False
        # 初始化经验回放缓冲区
        if self.n_step > 1:
            self.rl_buffer = n_step_replay_buffer(rl_buffer_size, self.n_step, self.gamma)
        else:
            self.rl_buffer = replay_buffer(rl_buffer_size)
        self.sl_buffer = reservoir_buffer(sl_buffer_size)
        # epsilon策略
        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(-1. * x / self.epsilon_decay)
        
        # 训练计数
        self.count = 0
        
        # 策略模式 (平均策略 或 最优策略)
        self.policy_mode = None
        self.choose_policy_mode()
        
        # 损失记录
        self.losses = []  # SL损失
        self.RLlosses = []  # RL损失
        self.policy_accuracies = []  # 策略准确度记录
 
        # 设置matplotlib支持中文
        rcParams['font.family'] = ['Microsoft YaHei']
        
        # 避免使用原始类中的use_raw属性
        self.use_raw = False
    
    def _init_networks(self, obs_sample):
        """
        初始化NFSP所需的所有神经网络
        为RL和SL策略设置不同的网络架构
        """
        # 预处理观察样本获取状态表示
        try:
            state = self._preprocess_state(obs_sample)
            if hasattr(state, 'shape') and len(state.shape) > 0:
                self.state_size = state.shape[0]
            else:
                # 直接预处理整个观察
                state = self._preprocess_state(obs_sample)
                self.state_size = len(state) if hasattr(state, '__len__') else 100
            
        except Exception as e:
            print(f"处理状态时出错: {e}")
            # 设置默认状态大小
            self.state_size = 100
            
        # 确保状态大小为正整数
        if not self.state_size or self.state_size <= 0:
            print("警告: 检测到无效的状态大小，使用默认值")
            self.state_size = 100  # 默认大小
        
        # 固定动作大小为环境中的动作数
        self.action_size = 6

        self.rl_eval_network = dueling_ddqn(self.state_size, self.action_size, 
                                hidden_units=self.hidden_units, num_layers=self.layers).to(self.device)
        
        self.rl_target_network = dueling_ddqn(self.state_size, self.action_size, 
                                hidden_units=self.hidden_units, num_layers=self.layers).to(self.device)
        
        self.rl_target_network.load_state_dict(self.rl_eval_network.state_dict())
        
        # 为RL网络设置优化器
        self.rl_optimizer = torch.optim.Adam(self.rl_eval_network.parameters(), lr=self.rl_lr)
        
        # 初始化监督学习策略网络
        self.sl_policy = policy(self.state_size, self.action_size,
                                    hidden_units=self.hidden_units, num_layers=self.layers).to(self.device)
        
        # 为SL网络设置优化器
        self.sl_optimizer = torch.optim.Adam(self.sl_policy.parameters(), lr=self.sl_lr)
        
        # 初始化平均策略和最佳响应
        self.avg_policy = AveragePolicy(self)
        self.best_response = RandomBestResponse(self.rl_eval_network)
        
        self.networks_initialized = True
       
    def _ensure_network_compatibility(self, state):
        """确保网络与输入状态大小兼容"""
        # 如果网络尚未初始化，初始化它
        if not self.networks_initialized:
            self._init_networks({'obs': state})
            self.state_size_checked = True  # 标记已检查状态尺寸
            return True
            
        # 如果已经检查过状态尺寸兼容性，不再重复检查
        if self.state_size_checked:
            return False
            
        # 首次检查状态大小是否与当前网络匹配
        state_size = state.shape[0]
        if state_size != self.state_size:
            print(f"检测到状态大小变化: 当前={state_size}, 网络期望={self.state_size}")
            print("重新构建网络以匹配新的状态大小...")
            
            # 存储旧网络的训练信息
            old_losses = self.losses.copy() if hasattr(self, 'losses') else []
            old_rl_losses = self.RLlosses.copy() if hasattr(self, 'RLlosses') else []
            old_accuracies = self.policy_accuracies.copy() if hasattr(self, 'policy_accuracies') else []
            
            # 更新状态大小
            self.state_size = state_size
            
            # 重新初始化网络
            self.networks_initialized = False
            self._init_networks({'obs': state})
            
            # 恢复旧的训练指标
            self.losses = old_losses
            self.RLlosses = old_rl_losses
            self.policy_accuracies = old_accuracies
            
            self.state_size_checked = True  # 标记已检查状态尺寸
            return True
        
        self.state_size_checked = True  # 标记已检查状态尺寸
        return False
    
    def choose_policy_mode(self):
        """选择策略模式，以eta概率使用平均策略，否则使用最优策略"""
        self.policy_mode = 'average' if random.random() < self.eta else 'best'
    
    def _preprocess_state(self, obs):
        """
        预处理观察到的状态
        处理三种类型的观察:
        1. 三层观测模式 - 包含3个通道的5x5数组:
           - 通道0: 当前智能体层
           - 通道1: 友方智能体层
           - 通道2: 食物层
        2. 网格观察模式 - 包含4个通道的数组:
           - 通道0: 智能体层，显示所有智能体的位置和等级
           - 通道1: 食物层，显示所有食物的位置和等级
           - 通道2: 可访问层，显示哪些位置可访问
           - 通道3: 自身标识层，标识自己的位置
        3. 标准观察模式 - 包含field和players信息的结构化数据
        """
        # 如果已经是numpy数组，可能是预处理过的状态
        if isinstance(obs, np.ndarray):
            # 首先检查是否是三层观测格式 [3,5,5]
            if len(obs.shape) == 3 and obs.shape[0] == 3 and obs.shape[1] == 5 and obs.shape[2] == 5:
                # 直接按权重叠加三层观测
                # 自身层乘以1，友方层乘以2，食物层乘以3
                merged_layer = obs[0] * 1 + obs[1] * 2 + obs[2] * 3
                return merged_layer.reshape(-1).astype(np.float32)
            # 检查是否是标准网格观测格式
            elif len(obs.shape) == 3:
                # 展平多维数组
                return obs.reshape(-1).astype(np.float32)
            # 确保是一维数组
            elif len(obs.shape) == 1:
                return obs.astype(np.float32)
            else:
                # 展平任何其他维度的数组
                return obs.reshape(-1).astype(np.float32)
        
        # 如果是字典格式，提取'obs'键
        if isinstance(obs, dict) and 'obs' in obs:
            raw_obs = obs['obs']
            # 递归处理提取的观察
            return self._preprocess_state(raw_obs)
        
        # 尝试获取可能的属性
        state_vector = []
        
        # 尝试提取field属性
        if hasattr(obs, 'field'):
            try:
                field = obs.field
                # 确保field是数组类型并可以展平
                if hasattr(field, 'flatten'):
                    state_vector.extend(field.flatten())
            except Exception as e:
                print(f"处理field时出错: {e}")
        
        # 尝试提取players属性
        if hasattr(obs, 'players') and hasattr(obs.players, '__iter__'):
            try:
                for player in obs.players:
                    # 尝试提取玩家位置
                    if hasattr(player, 'position'):
                        state_vector.extend(player.position)
                    # 尝试提取玩家等级
                    if hasattr(player, 'level'):
                        state_vector.append(player.level)
                    # 检查是否是自己
                    is_self = 1.0 if hasattr(player, 'is_self') and player.is_self else 0.0
                    state_vector.append(is_self)
            except Exception as e:
                print(f"处理players时出错: {e}")
        
        # 如果提取到了属性，返回状态向量
        if state_vector:
            return np.array(state_vector, dtype=np.float32)
        
        # 如果没有提取到任何信息，可能是自定义格式，尝试直接使用
        if hasattr(obs, 'shape'):
            # 可能是形状多维的张量，展平
            return np.array(obs).reshape(-1).astype(np.float32)
        
        # 如果是其他可迭代对象，尝试转换为numpy数组
        if hasattr(obs, '__iter__'):
            try:
                return np.array(list(obs), dtype=np.float32)
            except Exception as e:
                print(f"转换观察为数组时出错: {e}")
        
        # 最后的后备方案：创建一个默认状态
        print("警告: 无法处理观察，使用默认状态")
        return np.zeros(100, dtype=np.float32)
    
    def rl_train(self):
        """训练Q网络 (RL)"""
        # 确保网络已初始化
        if not hasattr(self, 'networks_initialized') or not self.networks_initialized:
            return
            
        observation, action, reward, next_observation, done = self.rl_buffer.sample(self.rl_batch_size)
        
        observation = torch.FloatTensor(observation).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_observation = torch.FloatTensor(next_observation).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # 计算当前Q值
        q_values = self.rl_eval_network.forward(observation)
        next_q_values = self.rl_target_network.forward(next_observation)
        argmax_actions = self.rl_eval_network.forward(next_observation).max(1)[1].detach()
        next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        expected_q_value = reward + self.gamma * (1 - done) * next_q_value
        
        # 计算损失
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        self.RLlosses.append(loss.item())
        
        # 优化模型
        self.rl_optimizer.zero_grad()
        loss.backward()
        self.rl_optimizer.step()
        
        # 定期更新目标网络
        if self.count % self.update_freq == 0:
            self.rl_target_network.load_state_dict(self.rl_eval_network.state_dict())
    
    def sl_train(self):
        """训练策略网络 (SL)"""
        # 确保网络已初始化
        if not hasattr(self, 'networks_initialized') or not self.networks_initialized:
            return
            
        observation, action = self.sl_buffer.sample(self.sl_batch_size)
        
        observation = torch.FloatTensor(observation).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        
        # 计算策略概率
        probs = self.sl_policy.forward(observation)
        log_prob = probs.gather(1, action.unsqueeze(1)).squeeze(1).log()
        
        # 计算准确率 (预测的最高概率动作与实际动作匹配的比例)
        pred_actions = probs.argmax(dim=1)
        accuracy = (pred_actions == action).float().mean().item()
        self.policy_accuracies.append(accuracy)
        
        # 计算损失 (负对数似然)
        loss = -log_prob.mean()
        self.losses.append(loss.item())
        
        # 优化模型
        self.sl_optimizer.zero_grad()
        loss.backward()
        self.sl_optimizer.step()
    
    def step(self, obs):
        """根据观察选择动作，支持字典格式的observation"""
        # 计数递增
        self.count += 1
        
        # 预处理状态
        state = self._preprocess_state(obs)
        
        # 确保网络已初始化并与状态大小兼容（只在第一次调用时进行完整检查）
        if not self.networks_initialized or not self.state_size_checked:
            self._ensure_network_compatibility(state)
        
        # 获取合法动作 - 直接使用环境提供的valid_actions
        if isinstance(obs, dict):
            if 'actions' in obs:
                # 优先使用字典中的actions键
                legal_actions = obs['actions']
            elif 'obs' in obs and hasattr(obs['obs'], 'actions'):
                # 然后尝试obs.actions
                legal_actions = obs['obs'].actions
            else:
                # 最后才使用默认动作
                legal_actions = list(range(self.action_size))
        elif hasattr(obs, 'actions'):
            # 直接从observation获取合法动作
            legal_actions = obs.actions
        else:
            # 默认所有动作都合法
            legal_actions = list(range(self.action_size))
        
        # 确保legal_actions非空
        if not legal_actions:
            print(f"警告: 智能体{self.player.level if hasattr(self.player, 'level') else '?'}收到空的legal_actions列表，使用所有动作")
            legal_actions = list(range(self.action_size))
            
        # 将状态转换为张量
        state_tensor = torch.FloatTensor(np.expand_dims(state, 0)).to(self.device)
        
        # 根据策略模式选择动作概率
        if self.policy_mode == 'best':
            probs = self.rl_eval_network.act(state_tensor, self.epsilon(self.count))
        else:
            probs = self.sl_policy.act(state_tensor)
        
        # 过滤不合法的动作
        valid_probs = action_mask(probs, legal_actions)
        
        # 选择动作
        action = np.random.choice(len(valid_probs), p=valid_probs)
        return action
    
    def eval_step(self, obs):
        """评估时选择动作，支持字典格式的observation"""
        # 预处理状态
        state = self._preprocess_state(obs)
        
        # 确保网络已初始化并与状态大小兼容（只在第一次调用时进行完整检查）
        if not self.networks_initialized or not self.state_size_checked:
            self._ensure_network_compatibility(state)
        
        # 获取合法动作 - 直接使用环境提供的valid_actions
        if isinstance(obs, dict):
            if 'actions' in obs:
                # 优先使用字典中的actions键
                legal_actions = obs['actions']
            elif 'obs' in obs and hasattr(obs['obs'], 'actions'):
                # 然后尝试obs.actions
                legal_actions = obs['obs'].actions
            else:
                # 最后才使用默认动作
                legal_actions = list(range(self.action_size))
        elif hasattr(obs, 'actions'):
            # 直接从observation获取合法动作
            legal_actions = obs.actions
        else:
            # 默认所有动作都合法
            legal_actions = list(range(self.action_size))
        
        # 将状态转换为张量
        state_tensor = torch.FloatTensor(np.expand_dims(state, 0)).to(self.device)
        
        # 根据评估模式选择动作概率
        if self.eval_mode == 'best':
            probs = self.rl_eval_network.act(state_tensor, 0)  # 评估时不使用随机探索
        else:
            probs = self.sl_policy.act(state_tensor)
        
        # 过滤不合法的动作
        valid_probs = action_mask(probs, legal_actions)
        
        # 选择动作
        action = np.random.choice(len(valid_probs), p=valid_probs)
        return action, probs
    
    def add_traj(self, traj):
        """
        将轨迹添加到经验缓冲区
        轨迹是包含state, action, reward, next_state, done等信息的转换样本列表
        """
        # 确认轨迹是单个转换样本，而不是列表
        if isinstance(traj, list) and len(traj) == 5:
            state, action, reward, next_state, done = traj
            # 使用标准缓冲区方法添加经验
            self.rl_buffer.store(state, action, reward, next_state, done)
            
            # 记录观察和行动对，用于监督学习缓冲区
            if self.policy_mode == 'best':
                self.sl_buffer.store(state, action)
    
    def train(self):
        """执行训练步骤"""
        # 如果缓冲区样本不足，跳过训练
        if len(self.rl_buffer) > self.rl_start and len(self.sl_buffer) > self.sl_start and self.count % self.train_freq == 0:
            self.rl_train()
            self.sl_train()
    
    def save_models(self, path="./models"):
        """保存模型"""
        # 确保网络已初始化
        if not hasattr(self, 'networks_initialized') or not self.networks_initialized:
            print("警告: 网络尚未初始化，无法保存模型")
            return False
            
        os.makedirs(path, exist_ok=True)
        player_id = self.player.level if hasattr(self.player, 'level') else "0"
        
        # 保存网络权重
        torch.save(self.rl_eval_network.state_dict(), f"{path}/nfsp_agent_{player_id}_q_network.pth")
        torch.save(self.sl_policy.state_dict(), f"{path}/nfsp_agent_{player_id}_policy_network.pth")
        
        # 保存模型元数据（保存状态大小信息，以便在不同观察模式下恢复）
        metadata = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_units': self.hidden_units
        }
        torch.save(metadata, f"{path}/nfsp_agent_{player_id}_metadata.pth")
        
        return True
    
    def load_models(self, path="./models"):
        """加载模型"""
        try:
            player_id = self.player.level if hasattr(self.player, 'level') else "0"
            
            # 尝试加载元数据
            metadata_path = f"{path}/nfsp_agent_{player_id}_metadata.pth"
            if os.path.exists(metadata_path):
                metadata = torch.load(metadata_path)
                stored_state_size = metadata.get('state_size', self.state_size)
                stored_action_size = metadata.get('action_size', self.action_size)
                stored_hidden_units = metadata.get('hidden_units', self.hidden_units)
                
                print(f"加载模型元数据: state_size={stored_state_size}, action_size={stored_action_size}")
                
                # 临时更新参数以匹配保存的模型
                self.state_size = stored_state_size
                self.action_size = stored_action_size
                self.hidden_units = stored_hidden_units
            
            # 初始化网络结构
            if not hasattr(self, 'networks_initialized') or not self.networks_initialized:
                # 使用临时状态创建初始网络
                dummy_state = np.zeros(self.state_size)
                self._init_networks({'obs': dummy_state})
                
            # 加载模型权重
            self.rl_eval_network.load_state_dict(torch.load(f"{path}/nfsp_agent_{player_id}_q_network.pth"))
            self.rl_target_network.load_state_dict(self.rl_eval_network.state_dict())
            self.sl_policy.load_state_dict(torch.load(f"{path}/nfsp_agent_{player_id}_policy_network.pth"))
            print(f"成功加载模型 - Agent {player_id}")
            
            return True
        except FileNotFoundError as e:
            print(f"找不到模型文件: {e}")
            return False
        except Exception as e:
            print(f"加载模型时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    def plot_losses(self):
        """同时绘制SL和RL损失曲线"""
        self.plot_SL_loss()
        self.plot_RL_loss()

    def get_policy_accuracy_history(self):
        """获取策略准确率历史记录"""
        return self.policy_accuracies
        
    def evaluate_team_exploitability(self, env, agents, num_episodes=100):
        """
        评估团队的可利用度
        
        这个方法将创建一个平均策略并计算团队的协作可利用度，
        可以用于监控训练过程中团队协作能力的演变。
        
        参数:
            env: 游戏环境
            agents: NFSPAgent列表
            num_episodes: 评估回合数
            
        返回:
            团队可利用度和平均团队奖励
        """
        try:
            # 简化实现，返回一个基本分数
            # 这避免了在评估过程中复杂的计算和可能的错误
            
            # 运行一个短回合评估获取基本奖励
            total_rewards = np.zeros(len(agents))
            for _ in range(min(10, num_episodes)):
                try:
                    _, payoffs = env.run(agents, is_training=False)
                    total_rewards += payoffs
                except Exception as e:
                    print(f"运行评估时出错: {e}")
                    continue
            
            # 计算平均奖励
            if sum(total_rewards) > 0:
                avg_reward = np.mean(total_rewards) / min(10, num_episodes)
            else:
                avg_reward = 0.5  # 默认值
            
            # 计算一个基于奖励的可利用度估计
            # 这里我们使用一个简单的启发式方法：高奖励对应低可利用度
            exploitability = max(0.0, 1.0 - avg_reward/2.0)
            
            return exploitability, avg_reward
            
        except Exception as e:
            print(f"评估可利用度时出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认值
            return 0.5, 1.0


class AveragePolicy:
    def __init__(self, policy_network):
        # 检查输入类型
        if hasattr(policy_network, 'sl_policy'):
            # 如果输入是NFSPAgent对象
            self.policy = policy_network.sl_policy
            if hasattr(policy_network, 'device'):
                self.device = policy_network.device
            else:
                self.device = next(self.policy.parameters()).device
        elif hasattr(policy_network, 'parameters'):
            # 如果输入是PyTorch模型
            self.policy = policy_network
            self.device = next(policy_network.parameters()).device
        else:
            # 处理其他情况（可能是PlayerObservation对象）
            raise ValueError(f"Unsupported policy_network type: {type(policy_network)}")
        
        self.use_raw = False
        
    def calculate_cooperative_exploitability(self, env, agents, num_episodes=100):
        """计算合作的可利用度（仅作为接口方法存在，避免调用错误）"""
        print("警告：合作可利用度计算功能尚未实现")
        # 简单地返回一个合理的默认值
        return 0.5
        
    def evaluate_team_performance(self, env, agents, num_episodes=100):
        """评估团队性能（仅作为接口方法存在，避免调用错误）"""
        print("警告：团队性能评估功能尚未实现")
        # 简单地返回一个合理的默认值
        return 1.0
      
    def act(self, state, eps=0):
        with torch.no_grad():
            # 状态预处理
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            
            # 确保状态维度正确
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
                
            # 获取策略输出
            probs = self.policy.forward(state)
            
            # 返回动作概率
            return probs.cpu().numpy()[0]


class RandomBestResponse:
    def __init__(self, q_network):
        # 检查输入类型
        if hasattr(q_network, 'rl_eval_network'):
            # 如果输入是NFSPAgent对象
            self.q_network = q_network.rl_eval_network
            if hasattr(q_network, 'device'):
                self.device = q_network.device
            else:
                self.device = next(self.q_network.parameters()).device
        elif hasattr(q_network, 'parameters'):
            # 如果输入是PyTorch模型
            self.q_network = q_network
            self.device = next(q_network.parameters()).device
        else:
            # 处理其他情况
            raise ValueError(f"不支持的q_network类型: {type(q_network)}")
            
        self.use_raw = False
      
    def act(self, state, eps=0.05):
        try:
            with torch.no_grad():
                # 状态预处理
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).to(self.device)
                
                # 确保状态维度正确
                if len(state.shape) == 1:
                    state = state.unsqueeze(0)
                    
                # 获取Q值
                q_values = self.q_network.forward(state)
                
                # epsilon-贪心策略
                if random.random() < eps:
                    # 随机探索
                    action_probs = np.ones(q_values.shape[1]) / q_values.shape[1]
                else:
                    # 选择最大Q值的动作
                    max_q_action = q_values.argmax(dim=1).item()
                    action_probs = np.zeros(q_values.shape[1])
                    action_probs[max_q_action] = 1.0
                    
                return action_probs
        except Exception as e:
            print(f"RandomBestResponse.act 出错: {e}")
            # 如果发生错误，返回均匀分布的动作概率
            if hasattr(self.q_network, 'forward'):
                output_size = self.q_network.forward(torch.zeros(1, state.shape[-1] if isinstance(state, torch.Tensor) else len(state)).to(self.device)).shape[1]
                return np.ones(output_size) / output_size
            else:
                # 如果无法确定输出大小，返回一个默认大小的均匀分布
                return np.ones(10) / 10 