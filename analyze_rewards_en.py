#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load training history data
data = np.load('results/training_history.npz')
rewards = data['episode_rewards'].sum(axis=1)
eval_rewards = data['eval_rewards']
eval_episodes = data['eval_episodes']
exploitability = data['exploitability']

# Calculate basic statistics
print(f'Total Average Reward: {rewards.mean():.4f}')
print(f'Max Reward: {rewards.max():.4f}')
print(f'Min Reward: {rewards.min():.4f}')

# Calculate average reward per 1000 episodes
chunk_size = 1000
for i in range(0, len(rewards), chunk_size):
    end = min(i + chunk_size, len(rewards))
    chunk = rewards[i:end]
    print(f'Episodes {i}-{end-1} Average Reward: {chunk.mean():.4f}')

# Plot training reward curve
plt.figure(figsize=(12, 8))

# Plot episode rewards (with smoothing)
window_size = 100
smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(smoothed_rewards))+window_size//2, smoothed_rewards, 
         label=f'Episode Rewards (window={window_size})')
plt.xlabel('Training Episodes')
plt.ylabel('Team Reward')
plt.title('Reward Trend During Training')
plt.grid(True)
plt.legend()

# Plot evaluation rewards and exploitability
plt.subplot(2, 1, 2)
plt.plot(eval_episodes, eval_rewards, 'b-o', label='Evaluation Rewards')
plt.plot(eval_episodes, exploitability, 'r-o', label='Exploitability')
plt.xlabel('Training Episodes')
plt.ylabel('Value')
plt.title('Evaluation Rewards and Exploitability')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('rewards_analysis_en.png')
print('Analysis chart saved to rewards_analysis_en.png')

# Calculate variance and standard deviation
print(f'Reward Variance: {rewards.var():.4f}')
print(f'Reward Standard Deviation: {rewards.std():.4f}')

# Further analyze reward stability
average_reward = rewards.mean()
above_avg = rewards > average_reward
below_avg = rewards < average_reward
print(f'Episodes above average reward: {np.sum(above_avg)} ({np.sum(above_avg)/len(rewards)*100:.2f}%)')
print(f'Episodes below average reward: {np.sum(below_avg)} ({np.sum(below_avg)/len(rewards)*100:.2f}%)')

# Check for trend in rewards as training progresses (simple linear regression)
episodes = np.arange(len(rewards))
slope, intercept, r_value, p_value, std_err = stats.linregress(episodes, rewards)
print(f'Reward trend slope: {slope:.6f} (p-value: {p_value:.6f}, R-squared: {r_value**2:.6f})')
print(f'Slope standard error: {std_err:.6f}')

if p_value < 0.05:
    print(f'Conclusion: Rewards show a statistically significant {"increasing" if slope > 0 else "decreasing"} trend')
else:
    print('Conclusion: Rewards show no statistically significant trend')

# Add RL and SL loss analysis
if 'rl_losses' in data and 'sl_losses' in data:
    plt.figure(figsize=(12, 8))
    
    # Plot RL losses
    rl_losses = data['rl_losses']
    plt.subplot(2, 1, 1)
    plt.plot(rl_losses)
    plt.xlabel('Training Steps (every 10 episodes)')
    plt.ylabel('Loss')
    plt.title('RL Loss During Training')
    plt.grid(True)
    
    # Plot SL losses
    sl_losses = data['sl_losses']
    plt.subplot(2, 1, 2)
    plt.plot(sl_losses)
    plt.xlabel('Training Steps (every 10 episodes)')
    plt.ylabel('Loss')
    plt.title('SL Loss During Training')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_analysis.png')
    print('Loss analysis chart saved to loss_analysis.png')
    
    # Check if losses are converging
    # For RL loss
    if len(rl_losses) > 20:
        first_20_avg = np.mean(rl_losses[:20])
        last_20_avg = np.mean(rl_losses[-20:])
        print(f'RL Loss - First 20 avg: {first_20_avg:.6f}, Last 20 avg: {last_20_avg:.6f}')
        print(f'RL Loss change: {(last_20_avg-first_20_avg)/first_20_avg*100:.2f}%')
    
    # For SL loss
    if len(sl_losses) > 20:
        first_20_avg = np.mean(sl_losses[:20])
        last_20_avg = np.mean(sl_losses[-20:])
        print(f'SL Loss - First 20 avg: {first_20_avg:.6f}, Last 20 avg: {last_20_avg:.6f}')
        print(f'SL Loss change: {(last_20_avg-first_20_avg)/first_20_avg*100:.2f}%') 