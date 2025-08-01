import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .agent import BaseAgent
from .model import policy, ActorNet, CriticNet


def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    advantages = []
    gae = 0
    next_value = 0 if dones[-1] else values[-1]  # Simplified
    for r, v, done in reversed(list(zip(rewards, values, dones))):
        delta = r + gamma * next_value * (1 - done) - v
        gae = delta + gamma * lambda_ * (1 - done) * gae
        advantages.insert(0, gae)
        next_value = v
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

def action_mask(probs, valid_actions):
    masked = np.zeros_like(probs)
    for a in valid_actions:
        masked[a] = probs[a]
    if np.sum(masked) == 0:
        masked = np.ones_like(probs)
    return masked / np.sum(masked)

class ReservoirBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.total_seen = 0

    def add(self, sample):
        self.total_seen += 1
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(sample)
        else:
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.buffer_size:
                self.buffer[idx] = sample

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(f"Reservoir buffer中样本不足，当前{len(self.buffer)}，需要{batch_size}")
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class IntegratedNFSPAgent(BaseAgent):
    def __init__(self, player, state_size, action_size, device, hidden_units, layers, gamma, eta, rl_lr, sl_lr, sl_buffer_size, eval_mode, entropy_coef, batch_size):
        super().__init__(player)
        self.name = f"Integrated NFSP-PPO Agent {player.level if hasattr(player, 'level') else ''}"
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.hidden_units = hidden_units
        self.layers = layers
        self.gamma = gamma
        self.eta = eta
        self.eval_mode = eval_mode

        # Integrated PPO parameters
        hidden_dims = [256] * 6
        self.actor = ActorNet(self.state_size, hidden_dims, self.action_size).to(device)
        self.critic = CriticNet(self.state_size, hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=rl_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=rl_lr)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=500, gamma=0.9)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=500, gamma=0.9)

        # Trajectory storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []

        # Training stats
        self.losses = []  # PPO total losses
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.policy_accuracies = []  # From SL

        # PPO hyperparameters
        self.lambda_ = 0.95
        self.clip_epsilon = 0.2
        self.update_epochs = 3
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = 0.5

        self.old_actor = ActorNet(self.state_size, hidden_dims, self.action_size).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())

        # SL part
        self.sl_policy = policy(self.state_size, self.action_size, hidden_units=self.hidden_units, num_layers=self.layers).to(self.device)
        self.sl_optimizer = torch.optim.Adam(self.sl_policy.parameters(), lr=sl_lr)
        self.sl_memory = ReservoirBuffer(sl_buffer_size)
        self.policy_mode = None

    # Integrated methods from PPOAgent
    def _preprocess_state(self, obs):
        """修复状态预处理逻辑，统一处理各种输入格式"""
        # 处理tuple格式 (obs_array, mask)
        if isinstance(obs, tuple) and len(obs) == 2:
            obs = obs[0]  # 只取obs_array部分
        
        # 处理字典格式
        if isinstance(obs, dict) and 'obs' in obs:
            raw_obs = obs['obs']
            return self._preprocess_state(raw_obs)
        
        # 处理numpy数组
        if isinstance(obs, np.ndarray):
            if len(obs.shape) == 3:
                return obs.reshape(-1).astype(np.float32)
            elif len(obs.shape) == 1:
                return obs.astype(np.float32)
            else:
                return obs.astype(np.float32)
        
        # 处理其他格式
        return np.array(obs, dtype=np.float32)

    def select_action(self, obs, is_training=False):
        if random.random() < self.eta:
            self.policy_mode = 'best'
            action = self.step(obs, deterministic=False)
            self.sl_memory.add((self._preprocess_state(obs), action))
        else:
            self.policy_mode = 'average'
            action = self.step(obs, deterministic=False)
        return action

    def add_traj2buffer(self, traj):
        if len(traj) != 5:
            return
        obs, action, reward, next_obs, done = traj

        obs = np.array(obs, dtype=np.float32)
        next_obs = np.array(next_obs, dtype=np.float32)
            
        # 计算当前状态的价值估计和对数概率
        state_tensor = torch.FloatTensor(np.expand_dims(obs, 0)).to(self.device)
        value = self.critic(state_tensor)
        value_scalar = value.detach().cpu().item()

        # 计算动作的对数概率 - 采样时用旧策略网络
        with torch.no_grad():
            action_probs = self.old_actor(state_tensor)
            dist = Categorical(action_probs)
            action_tensor = torch.tensor(action, dtype=torch.long).to(self.device)
            log_prob = dist.log_prob(action_tensor)
            log_prob_scalar = log_prob.item()

        # 转换数据类型
        action_scalar = int(action)
        reward_scalar = float(reward)
        done_scalar = bool(done)

        # 存储轨迹数据
        self.states.append(obs)
        self.actions.append(action_scalar)
        self.rewards.append(reward_scalar)
        self.next_states.append(next_obs)
        self.dones.append(done_scalar)
        self.values.append(value_scalar)
        self.log_probs.append(log_prob_scalar)

        # 检查缓冲区大小，如果过大则强制训练
        max_buffer_size = self.batch_size * 8
        if len(self.states) >= max_buffer_size:
            self.update()

    def clear_trajectory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []

    def update(self, states=None, actions=None, old_log_probs=None,
               returns=None, advantages=None):
        if not self.states:
            return
        if not hasattr(self, 'update_count'):
            self.update_count = 0
        self.update_count += 1
        
        # 计算GAE和returns
        if states is None:
            advantages_, returns_ = compute_gae(
                self.rewards,
                self.values,
                self.dones
            )
            states = np.array(self.states)
            actions = np.array(self.actions)
            old_log_probs = np.array(self.log_probs)
            returns = np.array(returns_)
            advantages = np.array(advantages_)
        else:
            states = np.array(states)
            actions = np.array(actions)
            old_log_probs = np.array(old_log_probs)
            returns = np.array(returns)
            advantages = np.array(advantages)
        
        # 标准化advantages和returns
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 转换为torch张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 记录训练损失
        actor_losses = []
        critic_losses = []
        entropies = []
        
        # 多次更新
        for epoch in range(self.update_epochs):
            # 计算当前策略的动作概率和状态价值
            action_probs = self.actor(states)
            values = self.critic(states).squeeze(-1)
            
            # 计算新的对数概率和熵
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 计算PPO目标函数
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # 计算actor和critic损失
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            critic_loss = F.mse_loss(values, returns)
            total_loss = actor_loss + self.value_coef * critic_loss
            
            # 记录损失值
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())
            
            # 更新actor网络
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()
            
            # 更新critic网络
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()
            
        # 更新学习率调度器
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # 计算平均损失
        actor_loss_avg = np.mean(actor_losses)
        critic_loss_avg = np.mean(critic_losses)
        entropy_avg = np.mean(entropies)
        total_loss_avg = actor_loss_avg + self.value_coef * critic_loss_avg
        
        # 记录训练统计
        self.losses.append(total_loss_avg)
        self.actor_losses.append(actor_loss_avg)
        self.critic_losses.append(critic_loss_avg)
        self.entropies.append(entropy_avg)
        
        # 清空轨迹缓冲区
        self.clear_trajectory()
        
        # 更新old_actor
        self.old_actor.load_state_dict(self.actor.state_dict())

    def save_models(self, path="./models", agent_id=0):
        import os
        os.makedirs(path, exist_ok=True)
        
        # 保存PPO网络
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
            'losses': self.losses,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'entropies': self.entropies
        }, f"{path}/integrated_nfsp_agent_{agent_id}_ppo_network.pth")
        
        # 保存SL网络
        torch.save(self.sl_policy.state_dict(), f"{path}/integrated_nfsp_agent_{agent_id}_policy_network.pth")
        
        # 保存元数据
        metadata = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_units': self.hidden_units
        }
        torch.save(metadata, f"{path}/integrated_nfsp_agent_{agent_id}_metadata.pth")
        return True

    def load_models(self, path, agent_id=0):
        """加载模型"""
        ppo_success = False
        sl_success = False

        # 加载PPO网络
        ppo_path = os.path.join(path, f"integrated_nfsp_agent_{agent_id}_ppo_network.pth")
        if os.path.exists(ppo_path):
            checkpoint = torch.load(ppo_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.actor.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.actor.load_state_dict(checkpoint)
            ppo_success = True

        # 加载SL网络
        sl_path = os.path.join(path, f"integrated_nfsp_agent_{agent_id}_policy_network.pth")
        if os.path.exists(sl_path):
            checkpoint = torch.load(sl_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.sl_policy.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.sl_policy.load_state_dict(checkpoint)
            sl_success = True

        return ppo_success or sl_success

    def rollout_and_train(self, env, teammate_agent=None, min_batch_size=None, max_steps=200, sl_batch_size=64, ppo_update_epochs=4, render=False, by_episode=False):
        if min_batch_size is None:
            min_batch_size = self.batch_size
        
        # 统计变量
        batch_rewards = []
        batch_steps = []
        batch_actor_losses = []
        batch_critic_losses = []
        batch_total_losses = []
        
        steps_collected = 0
        episode_count = 0
        
        # 清空PPO buffer
        self.clear_trajectory()
        
        max_episodes = min_batch_size * 2
        episode_safety_count = 0
        target_episodes = min_batch_size if by_episode else max_episodes

        while episode_count < target_episodes and episode_safety_count < max_episodes:
            agents = [self, teammate_agent] if teammate_agent else [self]

            try:
                trajectories, episode_reward, current_step, reward_detail = env.run(
                    agents=agents,
                    is_training=True,
                    render=render
                )

                batch_rewards.append(episode_reward)
                batch_steps.append(current_step)
                episode_count += 1
                episode_safety_count += 1

                if not by_episode:
                    steps_collected += current_step
                    if steps_collected >= min_batch_size:
                        break

            except Exception as e:
                episode_safety_count += 1
                continue
                
        # 执行训练
        if self.states:
            self.update()
        self.sl_train(sl_batch_size)

        # 获取训练损失
        batch_actor_losses = [self.actor_losses[-1] if self.actor_losses else None] * episode_count
        batch_critic_losses = [self.critic_losses[-1] if self.critic_losses else None] * episode_count
        batch_total_losses = [self.losses[-1] if self.losses else None] * episode_count
            
        return batch_rewards, batch_steps, batch_actor_losses, batch_critic_losses, batch_total_losses

    def sl_train(self, batch_size=64):
        """SL训练"""
        if len(self.sl_memory) < batch_size:
            return

        batch = self.sl_memory.sample(batch_size)
        states, actions = zip(*batch)
        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        logits = self.sl_policy(states)
        loss = torch.nn.CrossEntropyLoss()(logits, actions)

        # 计算准确率
        with torch.no_grad():
            pred_actions = torch.argmax(logits, dim=1)
            accuracy = (pred_actions == actions).float().mean().item()
            self.policy_accuracies.append(accuracy)

        self.sl_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sl_policy.parameters(), max_norm=1.0)
        self.sl_optimizer.step()

        if not hasattr(self, 'sl_losses'):
            self.sl_losses = []
        self.sl_losses.append(loss.item())

    def eval_step(self, obs):
        self.policy_mode = self.eval_mode
        return self.step(obs, deterministic=True)

    def debug_sl_buffer(self):
        print(f"[DEBUG] SL buffer: {len(self.sl_memory)} / {self.sl_memory.buffer_size}")
        if len(self.sl_memory) > 0:
            actions = [a for _, a in self.sl_memory.buffer]
            print("[DEBUG] 动作分布:", np.bincount(actions, minlength=self.action_size))

    def debug_compare_policy_distributions(self, state_batch):
        with torch.no_grad():
            ppo_probs = self.actor(state_batch).cpu().numpy()
            sl_probs = self.sl_policy(state_batch).cpu().numpy()
            kl = np.sum(ppo_probs * (np.log(ppo_probs + 1e-8) - np.log(sl_probs + 1e-8)), axis=1)
            print("[DEBUG] KL散度均值:", kl.mean())

    def get_policy_accuracy_history(self):
        return self.policy_accuracies

    def step(self, obs, deterministic=False):
        """状态预处理和动作选择"""
        if isinstance(obs, dict) and 'obs' in obs:
            state = self._preprocess_state(obs)
            valid_actions = obs.get('actions', list(range(self.action_size)))
        else:
            state = self._preprocess_state(obs)
            valid_actions = list(range(self.action_size))

        state_tensor = torch.FloatTensor(np.expand_dims(state, 0)).to(self.device)

        # 策略选择
        if self.policy_mode == 'best':
            probs = self.act(state_tensor)
        else:
            probs = self.sl_policy.act(state_tensor)

        # 只保留有效动作的概率，并重新归一化
        valid_probs = action_mask(probs, valid_actions)

        # 选择动作
        if deterministic:
            action = np.argmax(valid_probs)
        else:
            action = np.random.choice(self.action_size, p=valid_probs)

        return action

    def act(self, state):
        """
        参数:
            state: 状态张量
        返回:
            probs: 动作概率分布
        """
        with torch.no_grad():
            action_probs = self.actor(state).cpu().numpy()[0]

            # 使用温度参数控制探索
            temperature = 0.1
            scaled_probs = action_probs / temperature
            exp_probs = np.exp(scaled_probs - np.max(scaled_probs))
            probs = exp_probs / np.sum(exp_probs)

            return probs