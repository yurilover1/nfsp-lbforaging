import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import os

# PPO超参数
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.2
UPDATE_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5

class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.feature_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Actor网络（策略网络）
        self.actor = nn.Linear(prev_dim, output_dim)
        
        # Critic网络（价值网络）
        self.critic = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        # 特征提取
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        
        # 策略输出
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # 价值输出
        value = self.critic(x)
        
        return action_probs, value

class PPOAgent:
    """PPO智能体，适配env.run执行方式"""
    def __init__(self, input_dim, hidden_dims, output_dim, device="cpu", player=None):
        self.device = device
        self.actor_critic = ActorCritic(input_dim, hidden_dims, output_dim).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=LEARNING_RATE)
        self.name = f"PPO Agent {player.level if hasattr(player, 'level') else ''}推进式策略优化"
        
        # 添加与NFSP兼容的属性
        self.player = player  # 玩家信息
        self.last_ego_action = None  # 上一次动作
        
        # 轨迹存储
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # 训练统计
        self.losses = []  # 总损失记录
        self.actor_losses = []  # Actor损失记录
        self.critic_losses = []  # Critic损失记录
        self.entropies = []  # 熵记录
        
        # 训练计数器
        self.count = 0
        
        # 避免使用原始类中的use_raw属性
        self.use_raw = False
        
    def _preprocess_state(self, obs):
        """
        预处理观察到的状态，与NFSP兼容的接口
        处理不同类型的观察数据，包括字典格式
        """
        # 如果已经是numpy数组，直接返回
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32)
        
        # 如果是字典格式，提取'obs'键
        if isinstance(obs, dict) and 'obs' in obs:
            return self._preprocess_state(obs['obs'])
        
        # 尝试获取可能的属性，处理环境对象
        state_vector = []
        
        # 尝试提取field属性 (如果有)
        if hasattr(obs, 'field'):
            try:
                field = obs.field
                # 确保field是数组类型并可以展平
                if hasattr(field, 'flatten'):
                    state_vector.extend(field.flatten())
            except Exception as e:
                print(f"处理field时出错: {e}")
        
        # 尝试提取players属性 (如果有)
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
        
        # 无法处理的情况，返回默认向量
        return np.zeros(12, dtype=np.float32)
    
    def select_action(self, state):
        """选择动作"""
        processed_state = self._preprocess_state(state)
        state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)  # 添加批次维度
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        self.last_ego_action = action.item()  # 保存当前动作以与NFSP兼容
        return action.item(), log_prob.item()
    
    def _step(self, obs):
        """
        BaseAgent接口，由环境调用
        环境会传入字典格式的observation: {'obs': actual_obs, 'actions': valid_actions}
        """
        # 保存动作到历史
        action = self.step(obs)
        return action
    
    def step(self, obs):
        """与NFSP兼容的动作选择接口，支持字典格式的observation"""
        self.count += 1
        
        # 处理字典格式的观察
        if isinstance(obs, dict) and 'actions' in obs:
            valid_actions = obs['actions']
            state = self._preprocess_state(obs)
            
            # 获取动作概率
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs, _ = self.actor_critic(state_tensor)
                action_probs = action_probs.cpu().numpy()[0]
            
            # 创建有效动作的掩码
            mask = np.zeros_like(action_probs)
            mask[valid_actions] = 1
            masked_probs = action_probs * mask
            
            # 重新归一化
            if np.sum(masked_probs) > 0:
                masked_probs = masked_probs / np.sum(masked_probs)
            else:
                # 如果所有概率都为0，使用均匀分布
                masked_probs = np.zeros_like(action_probs)
                for a in valid_actions:
                    masked_probs[a] = 1.0 / len(valid_actions)
            
            # 根据概率选择动作
            action = np.random.choice(len(action_probs), p=masked_probs)
            self.last_ego_action = action
            return action
        else:
            # 直接使用select_action函数
            action, _ = self.select_action(obs)
            return action
    
    def add_traj(self, traj):
        """
        与NFSP兼容的轨迹添加接口，适配env.run方法
        
        参数:
            traj: [obs_dict, action, reward, next_obs_dict, done]格式的轨迹
        """
        if len(traj) != 5:
            return
            
        obs_dict, action, reward, next_obs_dict, done = traj
        
        # 预处理状态
        state = self._preprocess_state(obs_dict)
        next_state = self._preprocess_state(next_obs_dict) if not done else state
        
        # 存储轨迹
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(1 if done else 0)
        
        # 计算并存储价值估计
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.actor_critic(state_tensor)
            self.values.append(value.item())
            
            # 计算动作的对数概率
            action_probs, _ = self.actor_critic(state_tensor)
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(torch.tensor(action).to(self.device))
            self.log_probs.append(log_prob.item())
    
    def clear_trajectory(self):
        """清空当前存储的轨迹"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def evaluate_actions(self, states, actions):
        """评估动作"""
        states = torch.FloatTensor(states).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(0)  # 添加批次维度
            
        action_probs, values = self.actor_critic(states)
        dist = Categorical(action_probs)
        
        actions = torch.LongTensor(actions).to(self.device)
        if len(actions.shape) == 0:
            actions = actions.unsqueeze(0)  # 添加批次维度
            
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        return log_probs, values, entropy
    
    def update(self, states=None, actions=None, old_log_probs=None, returns=None, advantages=None):
        """更新策略"""
        # 如果没有提供参数，使用存储的轨迹计算
        if states is None and self.states:
            # 计算优势和回报
            advantages, returns = self._compute_gae(
                self.rewards, 
                self.values, 
                self.dones
            )
            states = np.array(self.states)
            actions = np.array(self.actions)
            old_log_probs = np.array(self.log_probs)
            
            # 更新后清空轨迹
            self.clear_trajectory()
        
        # 如果没有轨迹可训练，直接返回
        if states is None or len(states) == 0:
            return
            
        # 确保所有输入都是2D张量 [batch_size, feature_dim]
        states = torch.FloatTensor(states).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
            
        actions = torch.LongTensor(actions).to(self.device)
        if len(actions.shape) == 0:
            actions = actions.unsqueeze(0)
            
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        if len(old_log_probs.shape) == 0:
            old_log_probs = old_log_probs.unsqueeze(0)
            
        returns = torch.FloatTensor(returns).to(self.device)
        if len(returns.shape) == 0:
            returns = returns.unsqueeze(0)
            
        advantages = torch.FloatTensor(advantages).to(self.device)
        if len(advantages.shape) == 0:
            advantages = advantages.unsqueeze(0)
        
        # 标准化优势
        if len(advantages) > 1:  # 只有在batch size > 1时才标准化
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor_loss_total = 0
        critic_loss_total = 0
        entropy_total = 0
        
        for _ in range(UPDATE_EPOCHS):
            # 计算新的动作概率和价值
            log_probs, values, entropy = self.evaluate_actions(states, actions)
            
            # 计算比率
            ratio = torch.exp(log_probs - old_log_probs)
            
            # 计算PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_loss = F.mse_loss(values.squeeze(-1), returns)
            
            # 计算总损失
            loss = actor_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
            
            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 累积损失
            actor_loss_total += actor_loss.item()
            critic_loss_total += value_loss.item()
            entropy_total += entropy.item()
        
        # 计算平均损失
        actor_loss_avg = actor_loss_total / UPDATE_EPOCHS
        critic_loss_avg = critic_loss_total / UPDATE_EPOCHS
        entropy_avg = entropy_total / UPDATE_EPOCHS
        total_loss_avg = actor_loss_avg + VALUE_COEF * critic_loss_avg - ENTROPY_COEF * entropy_avg
        
        # 记录损失
        self.actor_losses.append(actor_loss_avg)
        self.critic_losses.append(critic_loss_avg)
        self.entropies.append(entropy_avg)
        self.losses.append(total_loss_avg)
    
    def train(self):
        """
        与NFSP兼容的训练接口，使用存储的轨迹进行训练
        在env.run方法执行完成后调用
        """
        if len(self.states) > 0:
            self.update()
    
    def eval_step(self, obs):
        """
        评估时选择动作
        与NFSP兼容的接口
        """
        processed_state = self._preprocess_state(obs)
        state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_tensor)
            action_probs_np = action_probs.cpu().numpy()[0]
            
            # 如果有有效动作列表
            if isinstance(obs, dict) and 'actions' in obs:
                valid_actions = obs['actions']
                # 创建有效动作的掩码
                mask = np.zeros_like(action_probs_np)
                mask[valid_actions] = 1
                masked_probs = action_probs_np * mask
                
                # 重新归一化
                if np.sum(masked_probs) > 0:
                    masked_probs = masked_probs / np.sum(masked_probs)
                else:
                    # 如果所有概率都为0，使用均匀分布
                    masked_probs = np.zeros_like(action_probs_np)
                    for a in valid_actions:
                        masked_probs[a] = 1.0 / len(valid_actions)
                
                action = np.argmax(masked_probs)  # 采用贪婪策略
                return action, masked_probs
            else:
                action = np.argmax(action_probs_np)  # 采用贪婪策略
                return action, action_probs_np
            
    def _compute_gae(self, rewards, values, dones, gamma=GAMMA, lambda_=LAMBDA):
        """计算广义优势估计（GAE）"""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lambda_ * (1 - dones[t]) * last_gae
        
        returns = advantages + np.array(values)
        return advantages, returns
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'entropies': self.entropies
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        if not os.path.exists(path):
            print(f"警告: 模型文件不存在: {path}")
            return False
            
        try:
            checkpoint = torch.load(path)
            self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载训练统计（如果存在）
            if 'losses' in checkpoint:
                self.losses = checkpoint['losses']
            if 'actor_losses' in checkpoint:
                self.actor_losses = checkpoint['actor_losses']
            if 'critic_losses' in checkpoint:
                self.critic_losses = checkpoint['critic_losses']
            if 'entropies' in checkpoint:
                self.entropies = checkpoint['entropies']
                
            print(f"成功加载模型: {path}")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False
            
    def save_models(self, path="./models"):
        """与NFSP兼容的多模型保存接口"""
        model_path = os.path.join(path, f"ppo_agent_{self.player.level if hasattr(self.player, 'level') else 0}_model.pt")
        return self.save_model(model_path)
        
    def load_models(self, path="./models"):
        """与NFSP兼容的多模型加载接口"""
        model_path = os.path.join(path, f"ppo_agent_{self.player.level if hasattr(self.player, 'level') else 0}_model.pt")
        return self.load_model(model_path) 