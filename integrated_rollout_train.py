import argparse

from agents.integrated_nfsp_agent import IntegratedNFSPAgent
from utils import *

logger = logging.getLogger(__name__)


def main(args):
    print('integrated_rollout_train.py running')
    import lbforaging.foraging.environment
    print('ENV FILE:', lbforaging.foraging.environment.__file__)
    """主函数，训练并测试集成NFSP智能体"""
    # 配置日志记录
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # 设置环境和智能体
    render_mode = "human" if args.render else None
    from lbforaging.foraging.environment import ForagingEnv
    env = ForagingEnv(
        num_agents=2,
        min_player_level=2,
        max_player_level=2,
        min_food_level=1,
        max_food_level=2,
        field_size=(5, 5),
        max_num_food=2,
        sight=5,
        max_episode_steps=30,
        force_coop=False,
        observe_agent_levels=True,
        render_mode=render_mode,
        step_reward_factor=0.4,
        attraction_reward_factor=0.2,
        decay_rate=0.01
    )
    action_size = 6  # (NONE, NORTH, SOUTH, WEST, EAST, LOAD)

    env.reset()
    args.agent_player = env.players[0]

    # 创建主智能体（集成NFSP）
    main_agent = IntegratedNFSPAgent(
        # 基本参数
        player=args.agent_player,
        state_size=calculate_state_size(env),
        action_size=action_size,
        device=args.device,
        # 网络结构参数
        hidden_units=args.hidden_units,
        layers=args.layers,
        # 训练参数
        gamma=args.gamma,
        eta=args.eta,  # 策略选择概率
        # 学习率
        rl_lr=args.rl_lr,  # PPO学习率
        sl_lr=args.sl_lr,  # 监督学习率
        # 缓冲区参数
        sl_buffer_size=30000,
        # 评估模式
        eval_mode='average',
        # 新增参数
        entropy_coef=args.entropy_coef,
        batch_size=args.batch_size
    )
    teammate_pool = [teammate_generate(6, device=args.device, id=i) for i in range(8)]
    # 训练模式
    if not args.test:
        print(f"\n开始训练集成NFSP智能体...\n")
        from tqdm import tqdm
        import time
        import csv
        import os
        
        # 创建logs目录
        os.makedirs('logs', exist_ok=True)
        
        # 初始化日志文件
        log_file = 'logs/train_debug.csv'
        csv_header = [
            'episode', 'reward_total', 'reward_base', 'reward_attraction', 'reward_step', 'reward_penalty',
            'rl_loss', 'actor_loss', 'critic_loss', 'sl_loss', 'policy_entropy', 'entropy_history', 'buffer_size',
            'eta', 'policy_mode', 'param_mean', 'param_std', 'steps', 'total_loss', 'policy_accuracy'
        ]
        
        # 清空旧文件，写入新表头
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
        
        # 统计变量
        episode_rewards = []
        episode_steps = []
        episode_actor_losses = []
        episode_critic_losses = []
        episode_total_losses = []
        
        num_batches = args.episodes // args.batch_size
        start_time = time.time()
        episode_counter = 0
        
        for i in tqdm(range(num_batches), desc="训练进度", ncols=80):
            # 执行训练，每次采集batch_size个episode
            teammate = random.choice(teammate_pool)
            batch_rewards, batch_steps, batch_actor_losses, batch_critic_losses, batch_total_losses = main_agent.rollout_and_train(env, teammate, min_batch_size=args.batch_size, by_episode=True, render=args.render)
            
            # 收集统计数据
            episode_rewards.extend(batch_rewards)
            episode_steps.extend(batch_steps)
            episode_actor_losses.extend(batch_actor_losses)
            episode_critic_losses.extend(batch_critic_losses)
            episode_total_losses.extend(batch_total_losses)
            
            # 写入日志
            for j, (reward, steps, actor_loss, critic_loss, total_loss) in enumerate(zip(batch_rewards, batch_steps, batch_actor_losses, batch_critic_losses, batch_total_losses)):
                episode_counter += 1
                
                # 计算奖励分解（简化版本）
                reward_base = reward if reward > 0 else 0
                reward_attraction = 0  # 简化处理
                reward_step = -0.01 * steps  # 步数惩罚
                reward_penalty = 0  # 如有reward_detail可取其penalty
                
                # 获取损失值
                rl_loss = total_loss if total_loss is not None else None
                sl_loss = main_agent.sl_losses[-1] if hasattr(main_agent, 'sl_losses') and main_agent.sl_losses else None
                policy_entropy = main_agent.entropies[-1] if hasattr(main_agent, 'entropies') and main_agent.entropies else None
                
                # 写入CSV
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode_counter,
                        reward,
                        reward_base,
                        reward_attraction,
                        reward_step,
                        reward_penalty,
                        rl_loss,
                        actor_loss,
                        critic_loss,
                        sl_loss,
                        policy_entropy,
                        '[]',  # entropy_history
                        len(main_agent.sl_memory) if hasattr(main_agent, 'sl_memory') else 0,
                        main_agent.eta,
                        main_agent.policy_mode,
                        0,  # param_mean
                        0,  # param_std
                        steps,
                        total_loss,  # total_loss
                        main_agent.policy_accuracies[-1] if hasattr(main_agent, 'policy_accuracies') and main_agent.policy_accuracies else None  # policy_accuracy
                    ])
                
                # 每1000局输出一次SL相关debug信息
                if episode_counter % 1000 == 0:
                    print("\n==== 集成NFSP SL Debug ====")
                    main_agent.debug_sl_buffer()
                    # 采样一批状态用于分布对比
                    if len(main_agent.sl_memory) > 0:
                        states = [s for s, _ in random.sample(main_agent.sl_memory.buffer, min(32, len(main_agent.sl_memory)))]
                        import torch
                        state_tensor = torch.FloatTensor(np.stack(states)).to(main_agent.device)
                        main_agent.debug_compare_policy_distributions(state_tensor)
            
            # 每200局打印一次统计信息
            if len(episode_rewards) >= 200:
                recent_rewards = episode_rewards[-200:]
                recent_steps = episode_steps[-200:]
                recent_actor_losses = episode_actor_losses[-200:]
                recent_critic_losses = episode_critic_losses[-200:]
                recent_total_losses = episode_total_losses[-200:]
                
                avg_reward = np.mean(recent_rewards)
                avg_steps = np.mean(recent_steps)
                avg_actor_loss = np.mean([x for x in recent_actor_losses if x is not None])
                avg_critic_loss = np.mean([x for x in recent_critic_losses if x is not None])
                avg_total_loss = np.mean([x for x in recent_total_losses if x is not None])
                
                elapsed_time = time.time() - start_time
                episodes_completed = episode_counter
                print(f"\n[进度 {episodes_completed}/{args.episodes}] "
                      f"平均奖励: {avg_reward:.4f} | "
                      f"平均步长: {avg_steps:.1f} | "
                      f"Actor Loss: {avg_actor_loss:.4f} | "
                      f"Critic Loss: {avg_critic_loss:.4f} | "
                      f"Total Loss: {avg_total_loss:.4f} | "
                      f"用时: {elapsed_time:.1f}s")
                
                # 清空统计列表以节省内存
                episode_rewards = episode_rewards[-100:]
                episode_steps = episode_steps[-100:]
                episode_actor_losses = episode_actor_losses[-100:]
                episode_critic_losses = episode_critic_losses[-100:]
                episode_total_losses = episode_total_losses[-100:]
        
        print("\n训练完成！\n")
        # 训练结束后保存模型
        main_agent.save_models(path="./models", agent_id=1)  # 使用不同的agent_id避免覆盖原始NFSP模型
        print("模型已保存到 ./models/")
        # 训练后自动评估
        print("\n开始自动评估训练好的模型...\n")
        for i in range(args.eval_episodes):
            teammate = random.choice(teammate_pool)
            agents = [main_agent, teammate]
            _, team_reward, steps, _ = env.run(agents)
            print(f"Test Episode {i+1}: reward={team_reward}, steps={steps}")
        print("\n自动评估完成！\n")
    # 测试模式
    else:
        print(f"\n开始测试集成NFSP智能体...\n")
        try:
            print("尝试加载模型...")
            main_agent.load_models("./models", agent_id=1)  # 使用不同的agent_id加载集成NFSP模型
            print("模型加载成功！")
            teammate = teammate_generate(6, device=args.device, id=0)
            
            # 是否使用渲染
            use_render = args.render
            print(f"渲染模式: {'开启' if use_render else '关闭'}")
            
            for i in range(args.eval_episodes):
                try:
                    _, team_reward, steps, _ = env.run([main_agent, teammate], render=use_render)
                    print(f"Test Episode {i+1}: reward={team_reward}, steps={steps}")
                except Exception as e:
                    print(f"运行回合时出错: {e}")
                    break
            print("\n测试完成！\n")
        except Exception as e:
            print(f"测试模式出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # 配置日志记录
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("integrated_debug.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("程序开始运行")
    
    try:
        parser = argparse.ArgumentParser(description="训练和测试集成NFSP智能体")
        
        # 基本选项
        parser.add_argument("--episodes", type=int, default=40000, help="训练回合数")
        parser.add_argument("--render", action="store_true", help="启用环境渲染")
        parser.add_argument("--render_interval", type=int, default=5, help="训练期间渲染的回合间隔")
        parser.add_argument("--eval_interval", type=int, default=1, help="评估间隔（批次数，每批次100回合）")
        parser.add_argument("--test", action="store_true", help="测试模式：加载预训练模型")
        parser.add_argument("--seed", type=int, default=None, help="随机种子")
        parser.add_argument("--verbose", action="store_true", help="启用详细日志")
        parser.add_argument("--eval_episodes", type=int, default=50, help="评估回合数")
        parser.add_argument("--device", type=str, default="cpu", help="设备类型")

        # 集成NFSP智能体参数
        parser.add_argument("--layers", type=int, default=4, help="神经网络层数")
        parser.add_argument("--hidden_units", type=int, default=512, help="隐藏层单元数")
        parser.add_argument("--eta", type=float, default=0.9, help="策略选择概率（1.0=纯PPO，0.1=NFSP）")
        parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
        parser.add_argument("--rl_lr", type=float, default=0.0002, help="PPO学习率")
        parser.add_argument("--sl_lr", type=float, default=0.0005, help="监督学习率")
        parser.add_argument("--entropy_coef", type=float, default=0.25, help="PPO熵系数")
        parser.add_argument("--batch_size", type=int, default=128, help="PPO batch size")

        args = parser.parse_args()
        
        logger.info("开始执行main函数")
        main(args)
        logger.info("程序正常结束")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("程序退出")