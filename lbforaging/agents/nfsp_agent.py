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

# 导入模型和缓冲区类，需要在同一目录下创建这些文件
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
                rl_lr=0.01,
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
                eval_mode='average'):
        
        super().__init__(player)
        self.name = f"NFSP Agent {player.level if hasattr(player, 'level') else ''}"
        
        # 初始化动作历史记录
        self.history = []
        
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
        
        # 设备设置
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 初始化网络
        self.RL_eval = dueling_ddqn(self.state_size, self.action_size).to(self.device)
        self.RL_target = dueling_ddqn(self.state_size, self.action_size).to(self.device)
        self.SL_policy = policy(self.state_size, self.action_size).to(self.device)
        
        # 复制Q网络到目标网络
        self.RL_target.load_state_dict(self.RL_eval.state_dict())
        
        # 初始化优化器
        self.RL_optimizer = torch.optim.Adam(self.RL_eval.parameters(), lr=rl_lr)
        self.SL_optimizer = torch.optim.Adam(self.SL_policy.parameters(), lr=sl_lr)
        
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
    
    def choose_policy_mode(self):
        """选择策略模式，以eta概率使用平均策略，否则使用最优策略"""
        self.policy_mode = 'average' if random.random() < self.eta else 'best'
    
    def _preprocess_state(self, obs):
        """预处理观察将其转换为模型的输入状态"""
        # 检查输入类型，兼容不同格式的输入
        if isinstance(obs, dict) and 'obs' in obs:
            # 原始格式：字典中包含带有field属性的obs对象
            if hasattr(obs['obs'], 'field'):
                try:
                    # 提取场地信息
                    field = np.copy(obs['obs'].field).flatten()
                    
                    # 提取玩家信息
                    players_info = []
                    for player in obs['obs'].players:
                        pos = player.position
                        level = player.level
                        is_self = 1 if player.is_self else 0
                        players_info.extend([pos[0], pos[1], level, is_self])
                    
                    # 组合状态
                    state = np.concatenate([field, np.array(players_info)])
                    return state
                except Exception as e:
                    print(f"预处理ForagingEnv观察时出错: {e}")
                    import traceback
                    traceback.print_exc()
            # 字典中包含numpy数组
            elif isinstance(obs['obs'], np.ndarray):
                return obs['obs']
        
        # 如果是预处理过的numpy数组，直接返回
        if isinstance(obs, np.ndarray):
            return obs
        
        # 如果是包含字典键"obs"，并且值是预处理后的numpy数组
        if isinstance(obs, dict) and 'obs' in obs and isinstance(obs['obs'], np.ndarray):
            return obs['obs']
            
        # 直接处理BaseAgent接收的原始观察
        if hasattr(obs, 'field'):
            try:
                # 提取场地信息
                field = np.copy(obs.field).flatten()
                
                # 提取玩家信息
                players_info = []
                for player in obs.players:
                    pos = player.position
                    level = player.level
                    is_self = 1 if player.is_self else 0
                    players_info.extend([pos[0], pos[1], level, is_self])
                
                # 组合状态
                state = np.concatenate([field, np.array(players_info)])
                return state
            except Exception as e:
                print(f"预处理原始观察时出错: {e}")
                import traceback
                traceback.print_exc()
                
        # 打印不支持的输入类型的更多信息
        print(f"输入类型: {type(obs)}")
        if isinstance(obs, dict):
            for key, value in obs.items():
                print(f"  键 '{key}' 类型: {type(value)}")
                
        # 不支持的输入类型
        raise ValueError(f"不支持的输入类型: {type(obs)}, {obs}")
    
    def rl_train(self):
        """训练Q网络 (RL)"""
        observation, action, reward, next_observation, done = self.rl_buffer.sample(self.rl_batch_size)
        
        observation = torch.FloatTensor(observation).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_observation = torch.FloatTensor(next_observation).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # 计算当前Q值
        q_values = self.RL_eval.forward(observation)
        next_q_values = self.RL_target.forward(next_observation)
        argmax_actions = self.RL_eval.forward(next_observation).max(1)[1].detach()
        next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        expected_q_value = reward + self.gamma * (1 - done) * next_q_value
        
        # 计算损失
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        self.RLlosses.append(loss.item())
        
        # 优化模型
        self.RL_optimizer.zero_grad()
        loss.backward()
        self.RL_optimizer.step()
        
        # 定期更新目标网络
        if self.count % self.update_freq == 0:
            self.RL_target.load_state_dict(self.RL_eval.state_dict())
    
    def sl_train(self):
        """训练策略网络 (SL)"""
        observation, action = self.sl_buffer.sample(self.sl_batch_size)
        
        observation = torch.FloatTensor(observation).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        
        # 计算策略概率
        probs = self.SL_policy.forward(observation)
        log_prob = probs.gather(1, action.unsqueeze(1)).squeeze(1).log()
        
        # 计算准确率 (预测的最高概率动作与实际动作匹配的比例)
        pred_actions = probs.argmax(dim=1)
        accuracy = (pred_actions == action).float().mean().item()
        self.policy_accuracies.append(accuracy)
        
        # 计算损失 (负对数似然)
        loss = -log_prob.mean()
        self.losses.append(loss.item())
        
        # 优化模型
        self.SL_optimizer.zero_grad()
        loss.backward()
        self.SL_optimizer.step()
    
    def step(self, obs):
        """根据观察选择动作，支持字典格式的observation"""
        # 计数递增
        self.count += 1
        
        # 预处理状态
        state = self._preprocess_state(obs)
        
        # 获取合法动作
        if hasattr(obs, 'actions'):
            legal_actions = obs.actions
        else:
            legal_actions = list(range(self.action_size))
        
        # 将状态转换为张量
        state_tensor = torch.FloatTensor(np.expand_dims(state, 0)).to(self.device)
        
        # 根据策略模式选择动作概率
        if self.policy_mode == 'best':
            probs = self.RL_eval.act(state_tensor, self.epsilon(self.count))
        else:
            probs = self.SL_policy.act(state_tensor)
        
        # 过滤不合法的动作
        valid_probs = action_mask(probs, legal_actions)
        
        # 选择动作
        action = np.random.choice(len(valid_probs), p=valid_probs)
        return action
    
    def eval_step(self, obs):
        """评估时选择动作，支持字典格式的observation"""
        # 预处理状态
        state = self._preprocess_state(obs)
        
        # 获取合法动作
        if hasattr(obs, 'actions'):
            legal_actions = obs.actions
        else:
            legal_actions = list(range(self.action_size))
        
        # 将状态转换为张量
        state_tensor = torch.FloatTensor(np.expand_dims(state, 0)).to(self.device)
        
        # 根据评估模式选择动作概率
        if self.eval_mode == 'best':
            probs = self.RL_eval.act(state_tensor, 0)  # 评估时不使用随机探索
        else:
            probs = self.SL_policy.act(state_tensor)
        
        # 过滤不合法的动作
        valid_probs = action_mask(probs, legal_actions)
        
        # 选择动作
        action = np.random.choice(len(valid_probs), p=valid_probs)
        return action, valid_probs
    
    def add_traj(self, traj):
        """存储轨迹到缓冲区"""
        obs, action, reward, next_obs, done = traj
        
        # 存储到RL缓冲区
        self.rl_buffer.store(obs, action, reward, next_obs, done)
        
        # 如果使用的是最优策略，也存储到SL缓冲区用于训练平均策略
        if self.policy_mode == 'best':
            self.sl_buffer.store(obs, action)
    
    def train(self):
        """执行训练步骤"""
        # 如果缓冲区样本不足，跳过训练
        if len(self.rl_buffer) > self.rl_start and len(self.sl_buffer) > self.sl_start and self.count % self.train_freq == 0:
            self.rl_train()
            self.sl_train()
    
    def plot_SL_loss(self, smooth_factor=0.1):
        """绘制SL损失曲线"""
        # 创建图形
        plt.figure(figsize=(10, 6))
        
        if len(self.losses) > 0:
            # 平滑处理
            smooth_losses = np.copy(self.losses)
            for i in range(1, len(self.losses)):
                smooth_losses[i] = smooth_factor * self.losses[i] + (1 - smooth_factor) * smooth_losses[i-1]
            
            plt.plot(smooth_losses, label="平滑监督学习损失")
            plt.plot(self.losses, alpha=0.3, label="原始监督学习损失")
            plt.xlabel("训练步数")
            plt.ylabel("损失")
            plt.title("SL训练损失随时间变化图")
            plt.legend()
        
        # 确保存在保存目录
        os.makedirs("./plots", exist_ok=True)
        plt.savefig(f"./plots/nfsp_agent_sl_losses.png")
        plt.close()
    
    def plot_RL_loss(self, smooth_factor=0.1):
        """绘制RL损失曲线"""
        # 创建图形
        plt.figure(figsize=(10, 6))
        
        if len(self.RLlosses) > 0:
            # 平滑处理
            smooth_RL_losses = np.copy(self.RLlosses)
            for i in range(1, len(self.RLlosses)):
                smooth_RL_losses[i] = smooth_factor * self.RLlosses[i] + (1 - smooth_factor) * smooth_RL_losses[i-1]
            
            plt.plot(smooth_RL_losses, label="平滑RL损失")
            plt.plot(self.RLlosses, alpha=0.3, label="原始RL损失")
            plt.xlabel("训练步数")
            plt.ylabel("损失")
            plt.title("RL训练损失随时间变化图")
            plt.legend()
        
        # 确保存在保存目录
        os.makedirs("./plots", exist_ok=True)
        plt.savefig(f"./plots/nfsp_agent_rl_losses.png")
        plt.close()
    
    def save_models(self, path="./models"):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        player_id = self.player.level if hasattr(self.player, 'level') else "0"
        torch.save(self.RL_eval.state_dict(), f"{path}/nfsp_agent_{player_id}_q_network.pth")
        torch.save(self.SL_policy.state_dict(), f"{path}/nfsp_agent_{player_id}_policy_network.pth")
    
    def load_models(self, path="./models"):
        """加载模型"""
        try:
            player_id = self.player.level if hasattr(self.player, 'level') else "0"
            self.RL_eval.load_state_dict(torch.load(f"{path}/nfsp_agent_{player_id}_q_network.pth"))
            self.RL_target.load_state_dict(self.RL_eval.state_dict())
            self.SL_policy.load_state_dict(torch.load(f"{path}/nfsp_agent_{player_id}_policy_network.pth"))
            print(f"成功加载模型 - Agent {player_id}")
            return True
        except FileNotFoundError:
            print(f"找不到模型文件")
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
        # 创建一个AveragePolicy实例用于计算
        avg_policy = AveragePolicy(self)
        
        # 计算可利用度
        exploitability = avg_policy.calculate_cooperative_exploitability(env, agents, num_episodes)
        
        # 创建基于平均策略的智能体
        avg_agents = [AveragePolicy(agent) for agent in agents]
        
        # 评估平均奖励
        avg_reward = avg_policy.evaluate_team_performance(env, avg_agents, num_episodes)
        
        return exploitability, avg_reward


class AveragePolicy:
    def __init__(self, agent):
        self.policy = agent.SL_policy
        self.device = agent.device
        self.use_raw = False

    def step(self, state):
        # 处理不同格式的状态输入
        if isinstance(state, dict) and 'obs' in state:
            obs = state['obs']
            # 提取合法动作
            if 'legal_actions' in state:
                legal_actions = state['legal_actions']
            elif hasattr(obs, 'actions'):
                legal_actions = obs.actions
            else:
                legal_actions = list(range(6))  # 默认6个动作
        else:
            # 如果state本身是观察对象
            if hasattr(state, 'field'):  # 原始观察对象
                obs = state
                legal_actions = state.actions if hasattr(state, 'actions') else list(range(6))
            else:
                # 假设state已经是预处理后的数组
                obs = state
                legal_actions = list(range(6))
                
        # 转换为张量进行推理
        if not isinstance(obs, np.ndarray):
            # 需要预处理
            if hasattr(obs, 'field'):
                # 提取场地和玩家信息
                field = np.copy(obs.field).flatten()
                players_info = []
                for player in obs.players:
                    pos = player.position
                    level = player.level
                    is_self = 1 if player.is_self else 0
                    players_info.extend([pos[0], pos[1], level, is_self])
                obs = np.concatenate([field, np.array(players_info)])
                
        # 转换为张量
        obs_tensor = torch.FloatTensor(np.expand_dims(obs, 0)).to(self.device)
        
        # 获取动作概率并过滤合法动作
        probs = self.policy.act(obs_tensor)
        probs = action_mask(probs, legal_actions)
        
        # 基于概率分布选择动作
        return np.random.choice(len(probs), p=probs)

    def eval_step(self, state):
        return self.step(state), None
        
    def calculate_cooperative_exploitability(self, env, agents, num_episodes=100):
        """
        计算合作环境下当前平均策略的可利用度
        
        在合作环境中，可利用度表示当前平均策略距离最优协作策略还有多远。
        计算方法是比较随机最优响应策略与当前平均策略之间的性能差距。
        
        参数:
            env: 游戏环境
            agents: NFSPAgent列表
            num_episodes: 评估回合数
            
        返回:
            可利用度值（越小表示策略越不可利用）
        """
        # 创建使用当前平均策略的智能体
        avg_agents = [AveragePolicy(agent) for agent in agents]
        
        # 初始化随机最优响应智能体
        br_agents = [RandomBestResponse(env, i) for i in range(len(agents))]
        
        # 计算各类策略组合的表现
        br_team_performance = self.evaluate_team_performance(env, br_agents, num_episodes)
        avg_team_performance = self.evaluate_team_performance(env, avg_agents, num_episodes)
        
        # 可利用度 = 随机最优响应团队表现 - 平均策略团队表现
        # 注：在合作环境中，差距越大，表示当前平均策略越容易被利用
        exploitability = br_team_performance - avg_team_performance
        
        return exploitability
    
    def evaluate_team_performance(self, env, agents, num_episodes=100):
        """
        使用环境的run函数评估团队在合作任务中的性能
        
        参数:
            env: 游戏环境
            agents: 智能体列表
            num_episodes: 评估回合数
            
        返回:
            平均每回合的团队总奖励
        """
        total_team_rewards = 0
        
        for _ in range(num_episodes):
            # 使用环境的run函数运行整个回合
            _, payoffs = env.run(agents, is_training=False)
            
            # 获取团队总奖励
            team_reward = np.sum(payoffs)
            total_team_rewards += team_reward
        
        # 返回平均每回合的团队总奖励
        return total_team_rewards / num_episodes


class RandomBestResponse:
    """随机最优响应智能体，作为可利用度计算的对照"""
    
    def __init__(self, env, player_id):
        self.env = env
        self.player_id = player_id
        self.use_raw = False
    
    def step(self, state):
        # 检查state格式并提取合法动作
        if isinstance(state, dict) and 'legal_actions' in state:
            legal_actions = state['legal_actions']
        elif isinstance(state, dict) and 'obs' in state:
            # 对于环境传入的观察格式
            if hasattr(state['obs'], 'actions'):
                legal_actions = state['obs'].actions
            else:
                # 如果没有指定合法动作，则使用所有可能的动作
                legal_actions = list(range(6))  # 假设有6个可能的动作
        else:
            # 如果state本身是观察对象
            if hasattr(state, 'actions'):
                legal_actions = state.actions
            else:
                # 默认使用所有动作
                legal_actions = list(range(6))
                
        # 随机选择一个合法动作
        return np.random.choice(legal_actions)
    
    def eval_step(self, state):
        # 使用相同的逻辑处理state
        action = self.step(state)
        # 创建一个均匀概率分布
        probs = np.zeros(6)  # 假设最多6个动作
        probs[action] = 1.0
        return action, probs 