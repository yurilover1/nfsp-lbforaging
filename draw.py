import matplotlib.pyplot as plt
import numpy as np

def plot_layers_comparison(layer_nums, type='nfsp'):
    """
    绘制不同网络层数的损失和奖励对比图
    
    参数:
    - layer_nums: 要比较的网络层数列表
    - type: 模型类型，默认为'nfsp'
    """
    # 设置颜色映射
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    for i, layer_num in enumerate(layer_nums):
        # 为每个层数选择一个颜色
        color = colors[i % len(colors)]
        
        # 尝试加载对应的历史数据
        try:
            history_file = f'./results/training_history_{layer_num}.npz'
            history = np.load(history_file, allow_pickle=True)
            
            # 提取RL损失数据
            if 'rl_losses' in history:
                losses = history['rl_losses'].reshape(-1)
                
                # 降采样
                if len(losses) > 1000:
                    step = len(losses) // 1000
                    losses = losses[::step]
                
                # 平滑处理
                window_size = min(20, len(losses))
                if window_size > 1:
                    smooth_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                    # 绘制平滑后的损失曲线
                    ax1.plot(smooth_losses, '-', color=color, linewidth=2, label=f'Layer {layer_num}')
            
            # 提取评估奖励数据
            if 'eval_rewards' in history and 'eval_batches' in history:
                eval_rewards = history['eval_rewards']
                eval_batches = history['eval_batches']
                
                # 绘制评估奖励曲线
                ax2.plot(eval_batches, eval_rewards, '-', color=color, linewidth=2, label=f'Layer {layer_num}')
                
                # 添加可选的平滑曲线（如果有足够的数据点）
                if len(eval_batches) > 5:
                    # 使用简单的移动平均来平滑曲线
                    window_size = min(10, len(eval_batches) // 2)
                    if window_size > 1:
                        # 对奖励进行平滑处理
                        smooth_rewards = []
                        for j in range(len(eval_rewards) - window_size + 1):
                            smooth_rewards.append(np.mean(eval_rewards[j:j+window_size]))
                        
                        # 由于窗口平滑，x坐标需要调整
                        smooth_x = eval_batches[:len(smooth_rewards)]
                        ax2.plot(smooth_x, smooth_rewards, '--', color=color, linewidth=1.5, 
                                alpha=0.7, label=f'Smooth Layer {layer_num}')
            
            print(f"成功加载并绘制层数 {layer_num} 的数据")
        except Exception as e:
            print(f"无法加载或处理层数 {layer_num} 的数据: {e}")
    
    # 设置第一个子图（损失）的标题和标签
    ax1.set_title('RL Losses Comparison Across Different Layers', fontsize=16)
    ax1.set_xlabel('Batch', fontsize=14)
    ax1.set_ylabel('Losses', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    
    # 设置第二个子图（奖励）的标题和标签
    ax2.set_title('Evaluation Rewards Comparison Across Different Layers', fontsize=16)
    ax2.set_xlabel('Batch', fontsize=14)
    ax2.set_ylabel('Rewards', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f'./results/layers_comparison_{type}.png')
    print(f"层数对比图已保存至: ./results/layers_comparison_{type}.png")
    plt.close()

if __name__ == "__main__":
    # 示例用法
    layer_nums = [7]  # 要比较的网络层数
    plot_layers_comparison(layer_nums, type='nfsp')