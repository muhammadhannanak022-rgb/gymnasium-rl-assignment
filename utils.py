
import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def save_training_plot(rewards, window=100, path="results/training.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(rewards, alpha=0.35, label="Episode reward")

    if len(rewards) >= window:
        averages = [
            sum(rewards[max(0, i - window + 1):i + 1])
            / len(rewards[max(0, i - window + 1):i + 1])
            for i in range(len(rewards))
        ]
        plt.plot(averages, label=f"{window}-episode moving average")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("FrozenLake Q-Learning Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_gif(frames, path="results/trained_agent.gif"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, duration=0.5, loop=0)
