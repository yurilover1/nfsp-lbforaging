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
    # 创建掩码数组，初始化为全0
    masked_probs = np.zeros(len(probs))
    
    # 只设置合法动作的概率
    for action in legal_actions:
        masked_probs[action] = probs[action]
    
    # 如果所有概率和为0，则采用均匀分布
    prob_sum = np.sum(masked_probs)
    if prob_sum == 0:
        for action in legal_actions:
            masked_probs[action] = 1.0 / len(legal_actions)
    else:
        # 重新归一化概率分布
        masked_probs = masked_probs / prob_sum
    
    return masked_probs
