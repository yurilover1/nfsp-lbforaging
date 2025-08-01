#!/usr/bin/env python3
"""
训练效果分析脚本
分析NFSP智能体在LBF环境中的训练效果
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置matplotlib参数以处理大量数据点
mpl.rcParams['agg.path.chunksize'] = 10000  # 增加chunksize参数

def analyze_training_log(log_file='logs/train_debug.csv'):
    """分析训练日志文件"""
    print("=== NFSP智能体训练效果分析 ===\n")
    
    if not os.path.exists(log_file):
        print(f"错误：找不到日志文件 {log_file}")
        return
    
    # 读取训练日志
    df = pd.read_csv(log_file)
    
    print(f"训练总回合数: {len(df)}")
    print(f"训练数据时间范围: 回合 {df['episode'].min()} - {df['episode'].max()}")
    
    # 分析奖励
    rewards = df['reward_total']
    print(f"\n=== 奖励分析 ===")
    print(f"平均奖励: {rewards.mean():.4f}")
    print(f"奖励标准差: {rewards.std():.4f}")
    print(f"最小奖励: {rewards.min():.4f}")
    print(f"最大奖励: {rewards.max():.4f}")
    
    # 分析奖励分布
    positive_rewards = rewards[rewards > 0]
    negative_rewards = rewards[rewards < 0]
    zero_rewards = rewards[rewards == 0]
    
    print(f"正奖励回合数: {len(positive_rewards)} ({len(positive_rewards)/len(rewards)*100:.2f}%)")
    print(f"负奖励回合数: {len(negative_rewards)} ({len(negative_rewards)/len(rewards)*100:.2f}%)")
    print(f"零奖励回合数: {len(zero_rewards)} ({len(zero_rewards)/len(rewards)*100:.2f}%)")
    
    # 分析训练趋势
    print(f"\n=== 训练趋势分析 ===")
    
    # 将训练分为几个阶段
    n_stages = 5
    stage_size = len(df) // n_stages
    
    for i in range(n_stages):
        start_idx = i * stage_size
        end_idx = (i + 1) * stage_size if i < n_stages - 1 else len(df)
        stage_rewards = rewards[start_idx:end_idx]
        
        print(f"阶段 {i+1} (回合 {start_idx}-{end_idx-1}): "
              f"平均奖励={stage_rewards.mean():.4f}, "
              f"正奖励比例={len(stage_rewards[stage_rewards>0])/len(stage_rewards)*100:.2f}%")
    
    # 分析PPO更新
    if 'rl_loss' in df.columns:
        rl_losses = df['rl_loss'].dropna()
        if len(rl_losses) > 0:
            print(f"\n=== PPO训练分析 ===")
            print(f"PPO更新次数: {len(rl_losses)}")
            print(f"平均PPO损失: {rl_losses.mean():.4f}")
            print(f"PPO损失标准差: {rl_losses.std():.4f}")
    
    # 分析策略熵
    if 'policy_entropy' in df.columns:
        entropies = df['policy_entropy'].dropna()
        if len(entropies) > 0:
            print(f"\n=== 策略熵分析 ===")
            print(f"平均策略熵: {entropies.mean():.6f}")
            print(f"策略熵标准差: {entropies.std():.6f}")
    
    # 分析策略准确率
    if 'policy_accuracy' in df.columns:
        accuracies = df['policy_accuracy'].dropna()
        if len(accuracies) > 0:
            print(f"\n=== 策略准确率分析 ===")
            print(f"平均策略准确率: {accuracies.mean():.4f}")
            print(f"策略准确率标准差: {accuracies.std():.4f}")
            print(f"最低准确率: {accuracies.min():.4f}")
            print(f"最高准确率: {accuracies.max():.4f}")
    
    # 对数据进行降采样以减少绘图数据点
    sample_size = min(5000, len(df))  # 最多使用5000个数据点
    if len(df) > sample_size:
        sample_indices = np.linspace(0, len(df)-1, sample_size, dtype=int)
        df_sampled = df.iloc[sample_indices]
    else:
        df_sampled = df
    
    # 绘制训练分析图
    plt.figure(figsize=(18, 12))
    
    # 1. Reward趋势
    plt.subplot(2, 3, 1)
    plt.plot(df_sampled['episode'], df_sampled['reward_total'], alpha=0.6, linewidth=0.5, color='blue')
    plt.title('Training Reward Trend')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    window_size = min(100, len(df_sampled) // 10)
    if window_size > 1:
        moving_avg = df_sampled['reward_total'].rolling(window=window_size).mean()
        plt.plot(df_sampled['episode'], moving_avg, 'r-', linewidth=2, label=f'Moving Avg (Window={window_size})')
        plt.legend()
    
    # 2. Actor Loss趋势
    plt.subplot(2, 3, 2)
    if 'actor_loss' in df_sampled.columns:
        actor_losses = df_sampled['actor_loss'].dropna()
        if len(actor_losses) > 0:
            plt.plot(range(len(actor_losses)), actor_losses, alpha=0.7, linewidth=1, color='green')
            plt.title('Actor Loss Trend')
            plt.xlabel('Update Step')
            plt.ylabel('Actor Loss')
            plt.grid(True, alpha=0.3)
            if len(actor_losses) > window_size:
                moving_avg_actor = pd.Series(actor_losses).rolling(window=window_size).mean()
                plt.plot(range(len(moving_avg_actor)), moving_avg_actor, 'r-', linewidth=2, label=f'Moving Avg (Window={window_size})')
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No Actor Loss Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Actor Loss Trend')
    elif 'rl_loss' in df_sampled.columns:
        rl_losses = df_sampled['rl_loss'].dropna()
        if len(rl_losses) > 0:
            plt.plot(range(len(rl_losses)), rl_losses, alpha=0.7, linewidth=1, color='green')
            plt.title('Actor Loss Trend (Using RL Loss)')
            plt.xlabel('Update Step')
            plt.ylabel('Actor Loss')
            plt.grid(True, alpha=0.3)
            if len(rl_losses) > window_size:
                moving_avg_actor = pd.Series(rl_losses).rolling(window=window_size).mean()
                plt.plot(range(len(moving_avg_actor)), moving_avg_actor, 'r-', linewidth=2, label=f'Moving Avg (Window={window_size})')
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No Actor Loss Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Actor Loss Trend')
    else:
        plt.text(0.5, 0.5, 'No Actor Loss Data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Actor Loss Trend')
    
    # 3. Critic Loss趋势
    plt.subplot(2, 3, 3)
    if 'critic_loss' in df_sampled.columns:
        critic_losses = df_sampled['critic_loss'].dropna()
        if len(critic_losses) > 0:
            plt.plot(range(len(critic_losses)), critic_losses, alpha=0.7, linewidth=1, color='orange')
            plt.title('Critic Loss Trend')
            plt.xlabel('Update Step')
            plt.ylabel('Critic Loss')
            plt.grid(True, alpha=0.3)
            if len(critic_losses) > window_size:
                moving_avg_critic = pd.Series(critic_losses).rolling(window=window_size).mean()
                plt.plot(range(len(moving_avg_critic)), moving_avg_critic, 'r-', linewidth=2, label=f'Moving Avg (Window={window_size})')
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No Critic Loss Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Critic Loss Trend')
    elif 'rl_loss' in df_sampled.columns:
        rl_losses = df_sampled['rl_loss'].dropna()
        if len(rl_losses) > 0:
            plt.plot(range(len(rl_losses)), rl_losses, alpha=0.7, linewidth=1, color='orange')
            plt.title('Critic Loss Trend (Using RL Loss)')
            plt.xlabel('Update Step')
            plt.ylabel('Critic Loss')
            plt.grid(True, alpha=0.3)
            if len(rl_losses) > window_size:
                moving_avg_critic = pd.Series(rl_losses).rolling(window=window_size).mean()
                plt.plot(range(len(moving_avg_critic)), moving_avg_critic, 'r-', linewidth=2, label=f'Moving Avg (Window={window_size})')
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No Critic Loss Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Critic Loss Trend')
    else:
        plt.text(0.5, 0.5, 'No Critic Loss Data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Critic Loss Trend')
    
    # 4. 每局步长变化
    plt.subplot(2, 3, 4)
    if 'steps' in df_sampled.columns:
        steps = df_sampled['steps']
        plt.plot(df_sampled['episode'], steps, alpha=0.7, linewidth=1, color='purple')
        plt.title('Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True, alpha=0.3)
        if len(steps) > window_size:
            moving_avg_steps = steps.rolling(window=window_size).mean()
            plt.plot(df_sampled['episode'], moving_avg_steps, 'r-', linewidth=2, label=f'Moving Avg (Window={window_size})')
            plt.legend()
    else:
        default_steps = [50] * len(df_sampled)
        plt.plot(df_sampled['episode'], default_steps, alpha=0.7, linewidth=1, color='purple')
        plt.title('Steps per Episode (Default: 50)')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True, alpha=0.3)
        plt.text(0.5, 0.5, 'Using Default 50 Steps\n(No step data in log)', 
                ha='center', va='center', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    # 5. Total Loss趋势
    plt.subplot(2, 3, 5)
    if 'total_loss' in df_sampled.columns:
        total_losses = df_sampled['total_loss'].dropna()
        if len(total_losses) > 0:
            plt.plot(range(len(total_losses)), total_losses, alpha=0.7, linewidth=1, color='brown')
            plt.title('Total Loss Trend')
            plt.xlabel('Update Step')
            plt.ylabel('Total Loss')
            plt.grid(True, alpha=0.3)
            if len(total_losses) > window_size:
                moving_avg_total = pd.Series(total_losses).rolling(window=window_size).mean()
                plt.plot(range(len(moving_avg_total)), moving_avg_total, 'r-', linewidth=2, label=f'Moving Avg (Window={window_size})')
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No Total Loss Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Total Loss Trend')
    else:
        plt.text(0.5, 0.5, 'No Total Loss Data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Total Loss Trend')
        
    # 6. Policy Accuracy趋势
    plt.subplot(2, 3, 6)
    if 'policy_accuracy' in df_sampled.columns:
        policy_accuracies = df_sampled['policy_accuracy'].dropna()
        if len(policy_accuracies) > 0:
            plt.plot(range(len(policy_accuracies)), policy_accuracies, alpha=0.7, linewidth=1, color='green')
            plt.title('Policy Accuracy Trend')
            plt.xlabel('Update Step')
            plt.ylabel('Policy Accuracy')
            plt.grid(True, alpha=0.3)
            if len(policy_accuracies) > window_size:
                moving_avg_acc = pd.Series(policy_accuracies).rolling(window=window_size).mean()
                plt.plot(range(len(moving_avg_acc)), moving_avg_acc, 'r-', linewidth=2, label=f'Moving Avg (Window={window_size})')
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No Policy Accuracy Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Policy Accuracy Trend')
    else:
        plt.text(0.5, 0.5, 'No Policy Accuracy Data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Policy Accuracy Trend')
    
    plt.tight_layout()
    try:
        plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n训练分析图表已保存至: training_analysis.png")
    except Exception as e:
        print(f"\n保存图表时出错: {e}")
        print("尝试降低DPI和采样率...")
        try:
            plt.savefig('training_analysis.png', dpi=100, bbox_inches='tight')
            print(f"训练分析图表已保存至: training_analysis.png (低分辨率)")
        except Exception as e2:
            print(f"再次保存失败: {e2}")
    finally:
        plt.close()
    
    # 总结
    print(f"\n=== 训练总结 ===")
    if rewards.mean() > -0.5:
        print("✅ 训练效果良好：平均奖励较高")
    elif rewards.mean() > -0.8:
        print("⚠️  训练效果一般：平均奖励中等")
    else:
        print("❌ 训练效果较差：平均奖励较低")
    
    if len(positive_rewards) / len(rewards) > 0.1:
        print("✅ 智能体能够获得正奖励")
    else:
        print("⚠️  智能体很少获得正奖励")
    
    print(f"\n建议：")
    if rewards.mean() < -0.8:
        print("- 考虑增加训练回合数")
        print("- 调整学习率或网络结构")
        print("- 检查环境设置是否合理")
    else:
        print("- 训练效果可以接受")
        print("- 可以尝试进一步优化超参数")

if __name__ == "__main__":
    analyze_training_log() 