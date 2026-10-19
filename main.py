
import os
import numpy as np
import gymnasium as gym

from agent import QLearningAgent
from utils import save_training_plot, save_gif


SEED = 42
EPISODES = 20000
MAX_STEPS = 100
EVAL_EPISODES = 100

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def train_agent():
    env = gym.make("FrozenLake-v1", is_slippery=True)

    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.8,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.9995,
        seed=SEED,
    )

    rewards = []
    successes = []

    for episode in range(EPISODES):
        state, info = env.reset(seed=SEED + episode)
        total_reward = 0
        success = False

        for step in range(MAX_STEPS):
            action = agent.choose_action(state, training=True)

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            agent.learn(
                state, action, reward, next_state, terminated
            )

            state = next_state
            total_reward += reward

            if terminated or truncated:
                success = bool(reward > 0)
                break

        agent.decay_epsilon()
        rewards.append(total_reward)
        successes.append(int(success))

        if (episode + 1) % 1000 == 0:
            recent = np.mean(successes[-1000:])
            print(
                f"Episode {episode + 1}/{EPISODES} | "
                f"Recent success rate: {recent:.2%} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    env.close()

    np.save("models/q_table.npy", agent.q_table)
    np.save("results/training_rewards.npy", np.array(rewards))
    np.save("results/training_successes.npy", np.array(successes))

    save_training_plot(rewards)

    return agent


def evaluate_agent(agent):
    env = gym.make("FrozenLake-v1", is_slippery=True)

    total_successes = 0
    episode_rewards = []

    for episode in range(EVAL_EPISODES):
        state, info = env.reset(seed=100000 + episode)
        total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        total_successes += int(total_reward > 0)
        episode_rewards.append(total_reward)

    env.close()

    success_rate = total_successes / EVAL_EPISODES

    print("\nEvaluation Results")
    print("------------------")
    print(f"Episodes: {EVAL_EPISODES}")
    print(f"Successful episodes: {total_successes}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average reward: {np.mean(episode_rewards):.3f}")

    return success_rate


def record_agent(agent):
    env = gym.make("FrozenLake-v1", is_slippery=True,
                   render_mode="rgb_array")

    best_frames = []
    best_reward = -1

    for episode in range(100):
        state, info = env.reset(seed=200000 + episode)
        frames = [env.render()]
        total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)

            frames.append(env.render())
            total_reward += reward

            if terminated or truncated:
                break

        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames

        if total_reward > 0:
            break

    env.close()
    save_gif(best_frames)

    if best_reward > 0:
        print("A successful episode was saved in results/trained_agent.gif")
    else:
        print("No success in recording attempts; saved the best attempt.")


if __name__ == "__main__":
    trained_agent = train_agent()
    evaluate_agent(trained_agent)
    record_agent(trained_agent)
