import numpy as np
import gymnasium as gym
import logging
import torch

logger = logging.getLogger(__name__)

def evaluate(env, agents, eval_episodes=100, calculate_exploitability=False, eval_env=None):
    """
    评估智能体性能
    
    参数:
        env: 游戏环境
        agents: 要评估的智能体列表
        eval_episodes: 评估回合数
        calculate_exploitability: 是否计算可利用度
        eval_env: 用于评估的环境，如果为None，则使用原环境的副本
    
    返回:
        如果calculate_exploitability为False，返回每个智能体的平均奖励；
        否则，返回(平均奖励，可利用度)元组
    """
    if eval_env is None:
        # 创建一个新的环境用于评估
        eval_env = gym.make(env.unwrapped.spec.id, render_mode=None)
    
    # 分类智能体
    nfsp_agents = [agent for agent in agents if hasattr(agent, 'add_traj')]
    ppo_agents = [agent for agent in agents if hasattr(agent, 'update') and not hasattr(agent, 'add_traj')]
    
    # 检查是否有PPO智能体
    has_ppo = len(ppo_agents) > 0
    
    # 如果包含PPO智能体，使用PPO的评估方式
    if has_ppo:
        total_rewards = np.zeros(len(agents))
        
        for _ in range(eval_episodes):
            obs, _ = eval_env.reset()
            episode_rewards = np.zeros(len(agents))
            done = False
            truncated = False
            
            while not (done or truncated):
                actions = []
                
                # 对每个智能体获取动作
                for i, agent in enumerate(agents):
                    action = agent.step(obs[i])
                    actions.append(action)
                
                # 执行动作
                obs, rewards, done_list, truncated_list, _ = eval_env.step(actions)
                
                # 更新奖励
                episode_rewards += np.array(rewards)
                
                # 更新done和truncated标志
                if isinstance(done_list, (list, tuple)):
                    done = all(done_list)
                else:
                    done = done_list
                    
                if isinstance(truncated_list, (list, tuple)):
                    truncated = all(truncated_list)
                else:
                    truncated = truncated_list
            
            # 累计奖励
            total_rewards += episode_rewards
    
    # 使用环境的run方法（对非PPO智能体更合适）
    elif hasattr(eval_env, 'run'):
        # 使用环境的run方法执行评估
        total_rewards = np.zeros(len(agents))
        for _ in range(eval_episodes):
            _, payoffs, _ = eval_env.run(agents, is_training=False)
            total_rewards += payoffs.sum()
    
    # 计算平均奖励
    avg_rewards = total_rewards / eval_episodes
    
    # 如果需要计算可利用度
    if calculate_exploitability and len(nfsp_agents) > 0:
        exploitability = 0.0
        # 从NFSP智能体计算可利用度
        for agent in nfsp_agents:
            if hasattr(agent, 'evaluate_team_exploitability'):
                agent_exploitability, _ = agent.evaluate_team_exploitability(eval_env, agents, eval_episodes=10)
                exploitability += agent_exploitability
        
        # 平均可利用度
        if nfsp_agents:
            exploitability /= len(nfsp_agents)
        
        return avg_rewards, exploitability
    
    return avg_rewards