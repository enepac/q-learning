import numpy as np
import random
from environment import Env
from collections import defaultdict
import matplotlib.pyplot as plt  # For plotting

class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.3  # Initial learning rate (adaptive)
        self.discount_factor = 0.8  # Reduced to focus more on immediate rewards
        self.epsilon = 1.0  # Start with full exploration
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))
    rewards_per_episode = []
    steps_per_episode = []

    decay_rate = 0.35  # Forces full exploitation sooner

    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        steps = 0
        status = "IN PROGRESS"

        # Adjust learning rate dynamically
        agent.learning_rate = max(0.05, 0.3 * np.exp(-0.009 * episode))  # Earlier stabilization

        while True:
            env.render()
            action = agent.get_action(str(state))
            next_state, reward, done = env.step(action)

            # Apply step penalty scaling
            step_penalty = -8 - (steps / 6)  # Harder penalty for unnecessary moves            

            reward += step_penalty

            # Apply improved proximity-based bonus
            goal_coords = env.canvas.coords(env.circle)
            distance_to_goal = np.linalg.norm(np.array(next_state) - np.array(goal_coords))
            reward += max(0, 30 - 0.2 * distance_to_goal)

            agent.learn(str(state), action, reward, str(next_state))
            state = next_state
            total_reward += reward
            steps += 1
            if done:
                status = "SUCCESS" if reward > 0 else "FAILED"
                break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        env.print_value_all(agent.q_table)

        # Apply precise epsilon decay
        agent.epsilon = max(0.01, agent.epsilon * np.exp(-decay_rate))

        # LOGGING: Print compact one-line output per episode with status
        print(f"Episode {episode+1} | Epsilon: {agent.epsilon:.4f} | Learning Rate: {agent.learning_rate:.3f} | Reward: {total_reward} | Steps: {steps} | Status: {status}")

    # Save log to a file
    with open("q_learning_log.txt", "w", encoding="utf-8") as log_file:
        for i in range(len(rewards_per_episode)):
            log_file.write(f"Episode {i+1} | Reward: {rewards_per_episode[i]} | Steps: {steps_per_episode[i]} | Status: {'SUCCESS ✅' if rewards_per_episode[i] > 0 else 'FAILED ❌'}\n")

    print("Training completed. Log saved as 'q_learning_log.txt'.")

    # Plot total rewards
    plt.figure(figsize=(12, 5))
    plt.plot(rewards_per_episode)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("total_reward_per_episode.png")
    plt.show(block=False)

    # Plot steps per episode
    plt.figure(figsize=(12, 5))
    plt.plot(steps_per_episode)
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Number of Steps")
    plt.savefig("steps_per_episode.png")
    plt.show(block=False)
