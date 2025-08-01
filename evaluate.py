import logging
import re

logger = logging.getLogger(__name__)

def evaluate(env, agents, eval_episodes=100, eval_env=None):
    """
    评估智能体性能
    
    参数:
        env: 游戏环境
        agents: 要评估的智能体列表
        eval_episodes: 评估回合数
        eval_env: 用于评估的环境，如果为None，则使用原环境的副本
    
    返回:
        返回每个智能体的平均奖励
    """
    if eval_env is None:
        # 创建一个新的环境用于评估
        from lbforaging.foraging.environment import ForagingEnv
        eval_env = ForagingEnv(
            num_agents=2,
            min_player_level=1,
            max_player_level=1,
            min_food_level=2,
            max_food_level=2,
            field_size=(5, 5),
            max_num_food=2,
            sight=2,
            max_episode_steps=50,
            force_coop=False,
            observe_agent_levels=True,
            render_mode=None,
            step_reward_factor=0.1,
            attraction_reward_factor=0.1,
            decay_rate=0.01
        )

    # 使用环境的run方法执行评估
    agents[0].policy_mode='best'
    total_rewards = 0
    for _ in range(eval_episodes):
        _, payoffs, steps, _ = eval_env.run(agents, is_training=False)
        total_rewards += payoffs
    # 计算平均奖励
    avg_rewards = total_rewards/ eval_episodes

    return avg_rewards.sum()

# 查找所有env.run调用，将其解包为4个变量
# 例如：_, payoffs, steps = env.run(...)  ->  _, payoffs, steps, _ = env.run(...)

with open('evaluate.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'env.run(' in line and '=' in line:
        # 匹配解包
        m = re.match(r'(\s*)([\w, ]+)=\s*env.run\(', line)
        if m:
            indent = m.group(1)
            left = m.group(2)
            # 统一用4个变量
            new_line = re.sub(r'([\w, ]+)=\s*env.run\(', '_, payoffs, steps, _ = env.run(', line)
            new_lines.append(new_line)
            continue
    new_lines.append(line)
with open('evaluate.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)