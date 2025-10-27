import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

def train_q_learning(env_name="Taxi-v3",
                     alpha=0.5,
                     gamma=0.99,
                     epsilon_start=0.3,
                     epsilon_end=0.01,
                     epsilon_decay_episodes=10000,
                     num_episodes=10000,
                     init_strategy="zeros"):
    """
    Trains a Q-learning agent on the specified Gymnasium environment.

    Returns:
      q_table               – the learned Q-table
      rewards_per_episode   – list of total reward per episode
      steps_per_episode     – list of step-count per episode
    """
    env = gym.make(env_name)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Q-table
    if init_strategy == "zeros":
        q_table = np.zeros((n_states, n_actions))
    elif init_strategy == "random":
        q_table = np.random.uniform(-1, 1, (n_states, n_actions))
    else:
        raise ValueError("Unknown init_strategy")

    rewards_per_episode = []
    steps_per_episode = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        # linearly decay epsilon
        eps = max(epsilon_end,
                  epsilon_start - (epsilon_start - epsilon_end) * (ep / epsilon_decay_episodes))

        while not done:
            # ε-greedy action selection
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, truncated, info = env.step(action)

            # Q-learning update
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

            state = next_state
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        # Optional: print progress
        if ep % (num_episodes // 10) == 0:
            avg_last_100 = np.mean(rewards_per_episode[-100:])
            print(f"Episode {ep}/{num_episodes} → Avg Reward (last 100): {avg_last_100:.2f}")

    env.close()
    return q_table, rewards_per_episode, steps_per_episode

def plot_metrics(rewards, steps, window=100):
    """ Plot reward and steps per episode (with moving average). """
    episodes = np.arange(1, len(rewards) + 1)
    # moving average
    ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ma_steps = np.convolve(steps, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(episodes, rewards, alpha=0.3, label="Reward per Episode")
    plt.plot(episodes[window-1:], ma_rewards, label=f"{window}-ep MA")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward over Episodes")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(episodes, steps, alpha=0.3, label="Steps per Episode")
    plt.plot(episodes[window-1:], ma_steps, label=f"{window}-ep MA")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Length over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- Hyperparameters 
    params = {
        "env_name": "Taxi-v3",
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon_start": 0.10,
        "epsilon_end": 0.01,
        "epsilon_decay_episodes": 20000,
        "num_episodes": 10000,
        "init_strategy": "zeros",   # "zeros" or "random"
    }

    # Train
    q_table, rewards, steps = train_q_learning(**params)

    # Plot results
    plot_metrics(rewards, steps)

    # Optionally, save the Q-table for later use:
    np.save("q_table.npy", q_table)
    print("Training complete. Q-table saved to q_table.npy")
