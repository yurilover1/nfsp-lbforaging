import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .model import ActorNet, CriticNet
from .utils import compute_gae, action_mask


def _extract_obs(obs):
    if isinstance(obs, tuple):
        return obs[0]
    return obs

def extract_obs_recursive(obs):
    while isinstance(obs, dict) and 'obs' in obs:
        obs = obs['obs']
    return obs

class PPOAgent:
    """PPO智能体，适配env.run执行方式"""
    def __init__(self, input_dim, hidden_dims, output_dim, device="cpu", player=None,
                 # PPO算法参数
                 gamma=0.99, lambda_=0.95, clip_epsilon=0.2, update_epochs=3,  # 减少update_epochs到3
                 # 训练参数
                 batch_size=256, actor_lr=3e-4, critic_lr=3e-4, entropy_coef=0.01, value_coef=0.5):  # 调整学习率和熵系数
        self.device = device
        self.actor = ActorNet(input_dim, hidden_dims, output_dim).to(device)
        self.critic = CriticNet(input_dim, hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 修改学习率调度器：更频繁的衰减
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=500, gamma=0.9)  # 更频繁衰减
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=500, gamma=0.9)
        self.name = f"PPO Agent {player.level if hasattr(player, 'level') else ''}分离结构"
        
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
        self.next_states = []  # 添加next_states属性
        self.valid_actions_list = [] # 存储每一步的valid_actions
        
        # 训练统计
        self.losses = []  # 总损失记录
        self.actor_losses = []  # Actor损失记录
        self.critic_losses = []  # Critic损失记录
        self.entropies = []  # 熵记录
        
        # 训练计数器
        self.count = 0
        
        # 避免使用原始类中的use_raw属性
        self.use_raw = False

        # 超参数 - 调整为更稳定的设置
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef  # 降低熵系数，减少随机性
        self.value_coef = value_coef
        self.debug_info = []  # 新增：用于存储调试信息

        self.old_actor = ActorNet(input_dim, hidden_dims, output_dim).to(device)
        # 初始化时同步参数
        self.old_actor.load_state_dict(self.actor.state_dict())

    def _preprocess_state(self, obs):
        """
        预处理观察到的状态，与NFSP兼容的接口
        处理不同类型的观察数据，包括字典格式
        """
        if isinstance(obs, dict):
            return self._preprocess_state(obs['obs'])
        return obs.astype(np.float32)

    def select_action(self, obs_dict, is_training=True):
        processed_state = self._preprocess_state(obs_dict)
        state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)  # ActorNet只返回action_probs
            action_probs_masked = action_mask(action_probs.cpu().numpy()[0], obs_dict['actions'])
            # 修复除以零问题：检查概率和是否大于0
            prob_sum = np.sum(action_probs_masked)
            if prob_sum > 1e-10:
                action_probs_masked = action_probs_masked / prob_sum
            else:
                # 如果所有概率都为0，使用均匀分布
                action_probs_masked = np.ones_like(action_probs_masked) / len(action_probs_masked)
            action_probs_tensor = torch.FloatTensor(action_probs_masked).unsqueeze(0).to(self.device)
            dist = Categorical(action_probs_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        assert action.item() in obs_dict['actions'], f"[ERROR][PPOAgent.select_action] 采样动作{action.item()}不在合法动作{obs_dict['actions']}中"
        self.last_ego_action = action.item()
        return action.item(), log_prob.item()
    
    def act(self, state, epsilon=0):
        """
        与NFSP兼容的动作选择接口
        参数:
            state: 状态张量
            epsilon: 探索率（PPO中不使用，但保持接口兼容）
        返回:
            probs: 动作概率分布
            is_greedy: 是否为贪婪动作
        """
        with torch.no_grad():
            action_probs = self.actor(state)  # ActorNet只返回action_probs
            action_probs = action_probs.cpu().numpy()[0]
            
            # 使用温度参数控制探索
            temperature = 0.1
            scaled_probs = action_probs / temperature
            exp_probs = np.exp(scaled_probs - np.max(scaled_probs))
            probs = exp_probs / np.sum(exp_probs)
            
            # 判断是否为贪婪动作
            is_greedy = True  # PPO通常使用随机策略，这里简化处理
            
            return probs, is_greedy
    
    def _step(self, obs, is_train=True):
        """
        BaseAgent接口，由环境调用
        环境会传入字典格式的observation: {'obs': actual_obs, 'actions': valid_actions}
        """
        # 保存动作到历史
        action = self.step(obs) if is_train else self.eval_step(obs)
        return action
    
    def step(self, obs):
        self.count += 1
        valid_actions = obs['actions']
        state = self._preprocess_state(obs)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)  # 1. 策略网络输出
            action_probs_np = action_probs.cpu().numpy()[0]
            action_probs_masked = action_mask(action_probs_np, valid_actions)
            prob_sum = np.sum(action_probs_masked)
            if prob_sum > 1e-10:
                action_probs_masked = action_probs_masked / prob_sum
                sampled_action = np.random.choice(len(action_probs_masked), p=action_probs_masked)
            else:
                action_probs_masked = np.ones_like(action_probs_masked) / len(action_probs_masked)

            # 添加温度参数来增加探索 - 防止灾难性遗忘
            temperature = max(0.3, 1.0 - self.count / 50000)  # 从1.0逐渐降低到0.3，降低速度更慢
            action_probs_masked = np.power(action_probs_masked, 1/temperature)
            action_probs_masked = action_probs_masked / np.sum(action_probs_masked)

            action_probs_tensor = torch.FloatTensor(action_probs_masked).unsqueeze(0).to(self.device)
            dist = Categorical(action_probs_tensor)
            action = dist.sample().item()
        self.last_ego_action = action
        assert action in valid_actions, f"[ERROR][PPOAgent.step] 采样动作{action}不在合法动作{valid_actions}中"
        return action
    
    def add_traj2buffer(self, traj):
        try:
            if len(traj) != 5:
                print(f"[ERROR] 轨迹格式错误，期望5个元素，实际{len(traj)}个")
                return
            obs_dict, action, reward, next_obs_dict, done = traj
            # 提取观测数据
            obs = extract_obs_recursive(obs_dict)
            next_obs = extract_obs_recursive(next_obs_dict)
            obs = np.array(obs, dtype=np.float32)
            next_obs = np.array(next_obs, dtype=np.float32)
            
            # 计算当前状态的价值估计和对数概率
            state_tensor = torch.FloatTensor(np.expand_dims(obs, 0)).to(self.device)
            value = self.critic(state_tensor)
            value_scalar = value.detach().cpu().numpy().flatten()[0]
            
            # 计算动作的对数概率 - 采样时用旧策略网络
            with torch.no_grad():
                action_probs = self.old_actor(state_tensor)
                dist = Categorical(action_probs)
                # 确保action是标量
                if isinstance(action, torch.Tensor):
                    action_tensor = action.clone().detach().to(self.device)
                else:
                    action_tensor = torch.tensor(action).to(self.device)
                if action_tensor.dim() > 0:
                    action_tensor = action_tensor.flatten()[0]
                log_prob = dist.log_prob(action_tensor)
                log_prob_scalar = log_prob.detach().cpu().numpy().flatten()[0] if log_prob.dim() > 0 else log_prob.item()
            
            # 确保所有数据都是正确的类型，安全地处理各种输入类型
            if isinstance(action, torch.Tensor):
                action_scalar = int(action.detach().cpu().numpy().flatten()[0])
            elif isinstance(action, np.ndarray):
                action_scalar = int(action.flatten()[0])
            else:
                action_scalar = int(action)
                
            if isinstance(reward, torch.Tensor):
                reward_scalar = float(reward.detach().cpu().numpy().flatten()[0])
            elif isinstance(reward, np.ndarray):
                reward_scalar = float(reward.flatten()[0])
            else:
                reward_scalar = float(reward)
                
            if isinstance(done, torch.Tensor):
                done_scalar = bool(done.detach().cpu().numpy().flatten()[0])
            elif isinstance(done, np.ndarray):
                done_scalar = bool(done.flatten()[0])
            else:
                done_scalar = bool(done)
            
            # 原子性地存储所有轨迹数据
            self.states.append(obs)
            self.actions.append(action_scalar)
            self.rewards.append(reward_scalar)
            self.next_states.append(next_obs)
            self.dones.append(done_scalar)
            self.values.append(value_scalar)
            self.log_probs.append(log_prob_scalar)
            
            # 检查缓冲区大小，如果过大则强制训练
            max_buffer_size = self.batch_size * 8  # 最大缓冲区大小为batch_size的8倍
            if len(self.states) >= max_buffer_size:
                print(f"[WARNING] 缓冲区过大 ({len(self.states)} >= {max_buffer_size})，强制训练")
                self.update()
            
        except Exception as e:
            print(f"[WARNING] 轨迹数据添加到buffer失败: {e}")
            return

        
        # 保存采样时actor参数
        # self.old_actor.load_state_dict(self.actor.state_dict()) # 删除采样期间的self.old_actor.load_state_dict(self.actor.state_dict())
    
    def clear_trajectory(self):
        """清空当前存储的轨迹"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.valid_actions_list = []
    
    def evaluate_actions(self, states, actions, valid_actions_list=None):
        """评估动作，使用掩码"""
        states = torch.FloatTensor(states).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(0)  # 添加批次维度
            
        action_probs, values = self.actor(states)
        
        # 如果有valid_actions_list，使用掩码
        if valid_actions_list is not None:
            masked_probs_list = []
            for i, (probs, valid_actions) in enumerate(zip(action_probs.detach().cpu().numpy(), valid_actions_list)):
                masked_probs = action_mask(probs, valid_actions)
                prob_sum = np.sum(masked_probs)
                if prob_sum > 1e-10:
                    masked_probs = masked_probs / prob_sum
                else:
                    masked_probs = np.ones_like(masked_probs) / len(masked_probs)
                masked_probs_list.append(masked_probs)
            
            action_probs = torch.FloatTensor(np.array(masked_probs_list)).to(self.device)
        
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
        if not self.states or len(self.states) == 0:
            print("[WARNING] PPO update called with empty buffer, skipping update.")
            return
        if not hasattr(self, 'update_count'):
            self.update_count = 0
        self.update_count += 1
        
        # 计算GAE和returns
        if states is None and self.states:
            # 检查列表长度是否一致
            if not (len(self.rewards) == len(self.values) == len(self.dones)):
                print(f"[ERROR] 列表长度不一致! rewards: {len(self.rewards)}, values: {len(self.values)}, dones: {len(self.dones)}")
                return
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        actor_losses = []
        critic_losses = []
        entropies = []
        for epoch in range(self.update_epochs):
            action_probs = self.actor(states)
            values = self.critic(states).squeeze(-1)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            critic_loss = F.mse_loss(values, returns)
            total_loss = actor_loss + self.value_coef * critic_loss
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        actor_loss_avg = np.mean(actor_losses)
        critic_loss_avg = np.mean(critic_losses)
        entropy_avg = np.mean(entropies)
        total_loss_avg = actor_loss_avg + self.value_coef * critic_loss_avg
        self.losses.append(total_loss_avg)
        self.actor_losses.append(actor_loss_avg)
        self.critic_losses.append(critic_loss_avg)
        self.entropies.append(entropy_avg)
        self.clear_trajectory()
        self.old_actor.load_state_dict(self.actor.state_dict())
        if self.update_count % 10 == 0:
            print(f"[PPO Update {self.update_count}] Actor Loss: {actor_loss_avg:.4f}, Critic Loss: {critic_loss_avg:.4f}, Total Loss: {total_loss_avg:.4f}, Entropy: {entropy_avg:.4f}")
    
    def eval_step(self, obs):
        """
        评估时选择动作
        与NFSP兼容的接口
        """
        valid_actions = obs['actions']
        action, _ = self.select_action(obs, valid_actions)
        return action
            
    def save_models(self, path, agent_id=0):
        """保存模型，与NFSP兼容"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
            'losses': self.losses,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'entropies': self.entropies
        }, f"{path}/nfsp_agent_{agent_id}_ppo_network.pth")
    
    def load_models(self, path, agent_id=0):
        """加载模型，与NFSP兼容"""
        try:
            # 加载模型，并指定设备
            checkpoint = torch.load(f"{path}/nfsp_agent_{agent_id}_ppo_network.pth", map_location=self.device)
            
            # 检查是否包含model_state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 加载模型状态
                self.actor.load_state_dict(checkpoint['model_state_dict'])
                
                # 加载其他统计信息（可选）
                self.losses = checkpoint.get('losses', [])
                self.actor_losses = checkpoint.get('actor_losses', [])
                self.critic_losses = checkpoint.get('critic_losses', [])
                self.entropies = checkpoint.get('entropies', [])
            else:
                # 如果不是预期的格式，尝试直接加载
                self.actor.load_state_dict(checkpoint)
            
            print(f"成功加载PPO模型 - Agent {agent_id}")
            return True
        except Exception as e:
            print(f"PPO模型加载失败: {e}")
            return False
