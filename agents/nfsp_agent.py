import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from .agent import BaseAgent
from .model import policy
from .ppo_agent import PPOAgent


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


class NFSPAgent(BaseAgent):
    def __init__(self, player, state_size, action_size, device, hidden_units, layers, gamma, eta, rl_lr, sl_lr, sl_buffer_size, eval_mode, entropy_coef, batch_size):
        super().__init__(player)
        self.name = f"NFSP-PPO Agent {player.level if hasattr(player, 'level') else ''}"
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.hidden_units = hidden_units
        self.layers = layers
        self.gamma = gamma
        self.eta = eta
        self.eval_mode = eval_mode
        # 初始化PPO智能体
        hidden_dims = [256] * 6
        self.rl_agent = PPOAgent(
            input_dim=self.state_size,
            hidden_dims=hidden_dims,
            output_dim=self.action_size,
            device=self.device,
            player=self.player,
            gamma=self.gamma,
            actor_lr=rl_lr,  # 使用传入的rl_lr
            critic_lr=rl_lr,  # 使用相同的学习率
            entropy_coef=entropy_coef,  # 使用传入的entropy_coef
            batch_size=batch_size
        )
        # 初始化监督学习网络（兼容性保留）
        self.sl_policy = policy(self.state_size, self.action_size,
                               hidden_units=self.hidden_units, num_layers=self.layers).to(self.device)
        self.sl_optimizer = torch.optim.Adam(self.sl_policy.parameters(), lr=sl_lr)
        self.sl_memory = ReservoirBuffer(sl_buffer_size)
        self.losses = []
        self.RLlosses = []
        self.policy_accuracies = []
        self.policy_mode = None
        self.eval_flag = False
        # 添加全局经验计数器，用于正确的reservoir采样
        self.global_step = 0

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

    def _extract_obs(self, obs):
        # 如果是tuple，取第一个元素
        if isinstance(obs, tuple):
            return obs[0]
        return obs

    def select_action(self, obs, is_training=False):
        if random.random() < self.eta:
            self.policy_mode = 'best'
            action = self.step(obs, deterministic=False)
            self.sl_memory.add((self._preprocess_state(obs), action))
        else:
            self.policy_mode = 'average'
            action = self.step(obs, deterministic=False)
        return action
    def step(self, obs, deterministic=False):
        obs = self._extract_obs(obs)
        """修复状态预处理，确保传入正确的格式"""
        # 处理环境返回的原始obs格式
        if isinstance(obs, dict) and 'obs' in obs:
            state = self._preprocess_state(obs)
            # 获取有效动作列表
            valid_actions = obs.get('actions', list(range(self.action_size)))
        else:
            state = self._preprocess_state(obs)
            valid_actions = list(range(self.action_size))
            
        state_tensor = torch.FloatTensor(np.expand_dims(state, 0)).to(self.device)
        
        # 策略选择
        if self.policy_mode == 'best':
            probs, _ = self.rl_agent.act(state_tensor, 0)
        else:
            probs = self.sl_policy.act(state_tensor)
        
        # 只保留有效动作的概率，并重新归一化
        valid_probs = np.zeros(self.action_size)
        for action in valid_actions:
            if 0 <= action < len(probs):
                valid_probs[action] = probs[action]
        
        # 重新归一化概率
        prob_sum = np.sum(valid_probs)
        if prob_sum > 0:
            valid_probs = valid_probs / prob_sum
        else:
            # 如果所有概率都为0，使用均匀分布
            for action in valid_actions:
                valid_probs[action] = 1.0 / len(valid_actions)
        
        # 选择动作
        if deterministic:
            action = np.argmax(valid_probs)
        else:
            action = np.random.choice(self.action_size, p=valid_probs)
        
        # 确保选择的动作在有效动作列表中
        if action not in valid_actions:
            # 如果选择的动作无效，从有效动作中随机选择
            action = np.random.choice(valid_actions)
        
        return action

    def rollout_and_train(self, env, teammate_agent=None, min_batch_size=None, max_steps=200, sl_batch_size=64, ppo_update_epochs=4, render=False, by_episode=False):
        if min_batch_size is None:
            min_batch_size = self.rl_agent.batch_size
        
        # 统计变量
        batch_rewards = []
        batch_steps = []
        batch_actor_losses = []
        batch_critic_losses = []
        batch_total_losses = []
        
        steps_collected = 0
        episode_count = 0
        
        # 清空PPO buffer
        self.rl_agent.clear_trajectory()
        
        max_episodes = min_batch_size * 2  # 最大回合数限制
        episode_safety_count = 0
        
        # 使用环境的run方法替换游戏执行部分
        target_episodes = min_batch_size if by_episode else max_episodes
        
        while episode_count < target_episodes and episode_safety_count < max_episodes:
            # 创建智能体列表
            agents = [self]
            if teammate_agent is not None:
                agents.append(teammate_agent)
            else:
                # 如果没有传入队友智能体，抛出异常
                raise ValueError("必须提供teammate_agent参数")
            
            # 使用环境的run方法执行回合
            try:
                # 运行回合
                trajectories, episode_reward, current_step, reward_detail = env.run(
                    agents=agents,
                    is_training=True,
                    render=render
                )

                # 收集统计信息
                batch_rewards.append(episode_reward)
                batch_steps.append(current_step)
                episode_count += 1
                episode_safety_count += 1
                
                # 如果按步数采集，累计步数
                if not by_episode:
                    steps_collected += current_step
                    # 检查是否收集到足够的步数
                    if steps_collected >= min_batch_size:
                        break
                
            except Exception as e:
                print(f"[WARNING] 回合执行出错: {e}")
                episode_safety_count += 1
                continue
        # 执行训练
        if hasattr(self.rl_agent, 'states') and len(self.rl_agent.states) > 0:
            self.rl_agent.update()
        else:
            print("[WARNING] PPO buffer is empty, skip update this batch.")
        self.sl_train(sl_batch_size)
        # 获取训练损失
        if hasattr(self.rl_agent, 'actor_losses') and self.rl_agent.actor_losses:
            batch_actor_losses = [self.rl_agent.actor_losses[-1]] * episode_count
        else:
            batch_actor_losses = [None] * episode_count
        if hasattr(self.rl_agent, 'critic_losses') and self.rl_agent.critic_losses:
            batch_critic_losses = [self.rl_agent.critic_losses[-1]] * episode_count
        else:
            batch_critic_losses = [None] * episode_count
        if hasattr(self.rl_agent, 'losses') and self.rl_agent.losses:
            batch_total_losses = [self.rl_agent.losses[-1]] * episode_count
        else:
            batch_total_losses = [None] * episode_count
        return batch_rewards, batch_steps, batch_actor_losses, batch_critic_losses, batch_total_losses

    def sl_train(self, batch_size=64):
        """修复SL训练，添加policy_accuracies更新"""
        if len(self.sl_memory) < batch_size:
            return
        
        batch = self.sl_memory.sample(batch_size)
        states, actions = zip(*batch)
        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        logits = self.sl_policy(states)
        loss = torch.nn.CrossEntropyLoss()(logits, actions)
        
        # 修复：计算并更新policy_accuracies
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            pred_actions = torch.argmax(probs, dim=1)
            accuracy = (pred_actions == actions).float().mean().item()
            self.policy_accuracies.append(accuracy)
        print(f"[SL] 当前准确率: {accuracy:.4f}")
        self.sl_optimizer.zero_grad()
        loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.sl_policy.parameters(), max_norm=1.0)
        self.sl_optimizer.step()
        self.losses.append(loss.item())
        # 打印SL训练信息
        if len(self.losses) % 20 == 0:  # 每20次训练打印一次
            print(f"[SL Train] Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    def eval_step(self, obs):
        self.policy_mode = self.eval_mode
        self.eval_flag = True
        return self.step(obs, deterministic=True)

    def save_models(self, path="./models", agent_id=0):
        import os
        os.makedirs(path, exist_ok=True)
        self.rl_agent.save_models(path, agent_id)
        torch.save(self.sl_policy.state_dict(), f"{path}/nfsp_agent_{agent_id}_policy_network.pth")
        metadata = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_units': self.hidden_units
        }
        torch.save(metadata, f"{path}/nfsp_agent_{agent_id}_metadata.pth")
        return True

    def load_models(self, path, agent_id=0):
        """加载模型"""
        ppo_success = False
        sl_success = False
        
        try:
            # 加载PPO网络
            ppo_path = os.path.join(path, f"nfsp_agent_{agent_id}_ppo_network.pth")
            if os.path.exists(ppo_path):
                try:
                    # 加载模型，并指定设备
                    checkpoint = torch.load(ppo_path, map_location=self.device)
                    
                    # 检查是否是字典格式并提取模型状态
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.rl_agent.actor.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.rl_agent.actor.load_state_dict(checkpoint)
                        
                    ppo_success = True
                    print(f"PPO网络加载成功: {ppo_path}")
                except Exception as e:
                    print(f"PPO网络加载失败: {e}")
            else:
                print(f"PPO网络文件不存在: {ppo_path}")
            
            # 加载SL网络
            sl_path = os.path.join(path, f"nfsp_agent_{agent_id}_policy_network.pth")
            if os.path.exists(sl_path):
                try:
                    # 加载模型，并指定设备
                    checkpoint = torch.load(sl_path, map_location=self.device)
                    
                    # 检查是否是字典格式并提取模型状态
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.sl_policy.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.sl_policy.load_state_dict(checkpoint)
                        
                    sl_success = True
                    print(f"SL网络加载成功: {sl_path}")
                except Exception as e:
                    print(f"SL网络加载失败: {e}")
            else:
                print(f"SL网络文件不存在: {sl_path}")
            
            # 只要有一个网络加载成功，就返回True
            return ppo_success or sl_success
            
        except Exception as e:
            print(f"模型加载过程中发生错误: {e}")
            return False

    def get_policy_accuracy_history(self):
        return self.policy_accuracies

    def add_traj2buffer(self, traj):
        """轨迹添加到PPO agent的缓冲区"""
        if hasattr(self, 'rl_agent') and hasattr(self.rl_agent, 'add_traj2buffer'):
            self.rl_agent.add_traj2buffer(traj)
        else:
            print(f"[WARNING] NFSP agent: rl_agent或add_traj2buffer方法不存在")

    def debug_sl_buffer(self):
        print(f"[DEBUG] SL buffer 当前容量: {len(self.sl_memory)} / {self.sl_memory.buffer_size}")
        if len(self.sl_memory) > 0:
            actions = [a for _, a in self.sl_memory.buffer]
            print("[DEBUG] SL buffer 动作分布:", np.bincount(actions, minlength=self.action_size))
        else:
            print("[DEBUG] SL buffer 为空")

    def debug_sl_policy_structure(self):
        print("[DEBUG] SL policy 网络结构:")
        print(self.sl_policy)
        param_count = sum(p.numel() for p in self.sl_policy.parameters())
        print(f"[DEBUG] SL policy参数量: {param_count}")

    def debug_sl_train_steps(self, batch_size=64, repeat=10):
        print(f"[DEBUG] SL policy 训练 {repeat} 个 batch，每个 batch 大小 {batch_size}")
        for i in range(repeat):
            self.sl_train(batch_size)

    def debug_sl_lr(self):
        print(f"[DEBUG] 当前SL policy学习率: {self.sl_optimizer.param_groups[0]['lr']}")

    def debug_compare_policy_distributions(self, state_batch):
        # state_batch: torch.FloatTensor [batch, state_dim]
        with torch.no_grad():
            ppo_probs = self.rl_agent.actor(state_batch).cpu().numpy()
            sl_probs = self.sl_policy(state_batch).cpu().numpy()
            kl = np.sum(ppo_probs * (np.log(ppo_probs + 1e-8) - np.log(sl_probs + 1e-8)), axis=1)
            print("[DEBUG] KL散度均值:", kl.mean())
            print("[DEBUG] PPO分布样例:", ppo_probs[0])
            print("[DEBUG] SL分布样例:", sl_probs[0])

    def debug_plot_sl_loss(self):
        if hasattr(self, 'losses') and len(self.losses) > 0:
            plt.figure()
            plt.plot(self.losses)
            plt.title('SL policy loss curve')
            plt.xlabel('SL train steps')
            plt.ylabel('Loss')
            plt.show()
        else:
            print("[DEBUG] SL policy loss 记录为空")
