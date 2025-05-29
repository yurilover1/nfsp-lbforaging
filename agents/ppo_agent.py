import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import os
from .utils import compute_gae, action_mask

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
    def __init__(self, input_dim, hidden_dims, output_dim, device="cpu", player=None,
                 gamma=0.99, lambda_=0.95, clip_epsilon=0.2, update_epochs=10, batch_size=64,
                 learning_rate=3e-4, entropy_coef=0.01, value_coef=0.5):
        self.device = device
        self.actor_critic = ActorCritic(input_dim, hidden_dims, output_dim).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
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

        # 超参数
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def _preprocess_state(self, obs):
        """
        预处理观察到的状态，与NFSP兼容的接口
        处理不同类型的观察数据，包括字典格式
        """
        if isinstance(obs, dict):
            return self._preprocess_state(obs['obs'])
        return obs.astype(np.float32)

    def select_action(self, state, valid_actions=None):
        """选择动作"""
        processed_state = self._preprocess_state(state)
        state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)  # 添加批次维度
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_tensor)
            action_probs = action_mask(action_probs, valid_actions)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        self.last_ego_action = action.item()  # 保存当前动作以与NFSP兼容
        return action.item(), log_prob.item()
    
    def _step(self, obs, is_train=True):
        """
        BaseAgent接口，由环境调用
        环境会传入字典格式的observation: {'obs': actual_obs, 'actions': valid_actions}
        """
        # 保存动作到历史
        action = self.step(obs) if is_train else self.eval_step(obs)
        return action
    
    def step(self, obs):
        """与NFSP兼容的动作选择接口，支持字典格式的observation"""
        self.count += 1

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

        masked_probs = masked_probs / np.sum(masked_probs)  \
            if np.sum(masked_probs) > 0 else np.ones_like(action_probs) / len(action_probs)
        
        # 根据概率选择动作
        action = np.random.choice(len(action_probs), p=masked_probs)
        self.last_ego_action = action
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
        
        # 确保values的维度与其他张量一致
        values = values.view(-1)  # 将values展平为1D张量
        
        return log_probs, values, entropy
    
    def update(self, states=None, actions=None, old_log_probs=None, 
        returns=None, advantages=None):
        """更新策略"""
        # 如果没有提供参数，使用存储的轨迹计算
        if states is None and self.states:
            # 计算优势和回报
            advantages, returns = compute_gae(
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
            
        # 转换为张量并确保维度正确
        states = torch.FloatTensor(states).to(self.device
            ).view(-1, states.shape[-1] if len(states.shape) > 1 else 1)
        actions = torch.LongTensor(actions).to(self.device).view(-1)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device
            ).view(-1)
        returns = torch.FloatTensor(returns).to(self.device).view(-1)
        advantages = torch.FloatTensor(advantages).to(self.device).view(-1)
        
        # 标准化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() 
                + 1e-8)
        
        actor_loss_total = 0
        critic_loss_total = 0
        entropy_total = 0
        
        for _ in range(self.update_epochs):
            # 计算新的动作概率和价值
            log_probs, values, entropy = self.evaluate_actions(states, actions)
            
            # 计算比率
            ratio = torch.exp(log_probs - old_log_probs)
            
            # 计算PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失 - 确保维度匹配
            value_loss = F.mse_loss(values.view(-1), returns)
            
            # 计算总损失
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 累积损失
            actor_loss_total += actor_loss.item()
            critic_loss_total += value_loss.item()
            entropy_total += entropy.item()
        
        # 计算平均损失
        actor_loss_avg = actor_loss_total / self.update_epochs
        critic_loss_avg = critic_loss_total / self.update_epochs
        entropy_avg = entropy_total / self.update_epochs
        total_loss_avg = actor_loss_avg + self.value_coef * critic_loss_avg - self.entropy_coef * entropy_avg
        
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
        valid_actions = obs['actions']
        action, _ = self.select_action(obs, valid_actions)
        return action
            
    def save_models(self, path):
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

    def load_models(self, path):
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
