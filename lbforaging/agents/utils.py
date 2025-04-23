import numpy as np
import matplotlib.pyplot as plt
import os


def action_mask(probs, legal_actions):
    """
    过滤不合法的动作，只保留合法动作的概率
    
    参数:
        probs: 动作概率分布
        legal_actions: 合法动作列表
    
    返回:
        过滤后的概率分布
    """
    # 确保legal_actions非空
    if not legal_actions:
        print("警告: legal_actions为空，仅使用NONE动作(索引0)")
        legal_actions = [0]  # 只使用NONE动作(索引0)，而不是所有动作
    
    # 处理probs中可能存在的NaN或inf值
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 创建掩码数组，初始化为全0
    masked_probs = np.zeros(len(probs))
    
    # 只设置合法动作的概率
    for action in legal_actions:
        if 0 <= action < len(probs):  # 确保动作索引有效
            masked_probs[action] = probs[action]
    
    # 如果所有概率和为0，则采用均匀分布
    prob_sum = np.sum(masked_probs)
    if prob_sum <= 1e-10:  # 使用一个很小的阈值而不是精确的0
        for action in legal_actions:
            if 0 <= action < len(masked_probs):  # 确保动作索引有效
                masked_probs[action] = 1.0 / len(legal_actions)
    else:
        # 重新归一化概率分布
        masked_probs = masked_probs / prob_sum
    
    # 最后再次检查是否包含NaN或inf
    masked_probs = np.nan_to_num(masked_probs, nan=1.0/len(legal_actions), posinf=1.0, neginf=0.0)
    
    # 确保概率和为1
    if np.sum(masked_probs) <= 1e-10:  # 极端情况下仍然可能概率和为0
        if legal_actions:
            # 使用均匀分布
            for action in legal_actions:
                if 0 <= action < len(masked_probs):
                    masked_probs[action] = 1.0 / len(legal_actions)
        else:
            # 如果一切都失败了，使用NONE动作（索引0）
            masked_probs[0] = 1.0
    else:
        # 再次归一化，确保和为1
        masked_probs = masked_probs / np.sum(masked_probs)
    
    return masked_probs