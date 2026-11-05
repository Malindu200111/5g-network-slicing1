import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.agent import DQNAgent
from src.environment import FiveGNetworkSlicingEnv

def main():
    os.makedirs("outputs", exist_ok=True)
    env = FiveGNetworkSlicingEnv()
    agent = DQNAgent()

    num_episodes = 50
    episode_rewards = []

    print(">>> 5G Network Slicing DQN Training Pipeline Started <<<")

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        step_count = 0
        losses = []

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.learn()
            if loss > 0:
                losses.append(loss)

            state = next_state
            total_reward += reward
            step_count += 1
            if done:
                break

        episode_rewards.append(total_reward)
        if ep % 5 == 0:
            agent.update_target_network()

        avg_loss = np.mean(losses) if losses else 0.0
        print(f"Ep {ep:02d}/{num_episodes:02d} | Steps: {step_count} | Reward: {total_reward:8.2f} | Loss: {avg_loss:.4f} | URLLC Latency: {info['urllc_delay']:.2f} ms")

    # Outputs Save කිරීම
    torch.save(agent.policy_net.state_dict(), "outputs/dqn_model.pth")
    plt.figure()
    plt.plot(episode_rewards)
    plt.title("DQN Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("outputs/dqn_reward.png")
    print(">>> Training complete. Model & plot saved to outputs/ <<<")

if __name__ == "__main__":
    main()