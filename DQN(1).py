import gym
from gym import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


# Custom Windy Gridworld environment
class WindyGridworldEnv(gym.Env):
    def __init__(self):
        super(WindyGridworldEnv, self).__init__()
        # Define the dimensions of the gridworld.
        # The grid is 7 rows by 10 columns.
        self.grid_shape = (7, 10)

        # Define the start and goal states within the grid.
        # Coordinates are in the form (row, column).
        self.start_state = (3, 0)  # Start at row 3, column 0
        self.goal_state = (3, 7)  # Goal at row 3, column 7

        # Define the wind strength for each column.
        # The values indicate how many cells upward the agent is pushed
        # when it enters that column.
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        # Define the action space: Up=0, Down=1, Left=2, Right=3.
        self.action_space = spaces.Discrete(4)

        # Define the observation space as the agent's position (row, column).
        # Rows range from 0 to 6, columns from 0 to 9.
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([6, 9]), dtype=np.int32
        )

        self.state = None

    def reset(self):
        """Reset the environment to the start state."""
        self.state = self.start_state
        return np.array(self.state)

    def step(self, action):
        """Execute one step in the environment, applying wind and actions."""
        row, col = self.state

        # Apply the wind effect (push upward).
        wind_effect = self.wind[col]
        row -= wind_effect

        # Apply the selected action to move the agent.
        if action == 0:  # Up
            row -= 1
        elif action == 1:  # Down
            row += 1
        elif action == 2:  # Left
            col -= 1
        elif action == 3:  # Right
            col += 1

        # Ensure the agent stays within the boundaries of the grid.
        row = np.clip(row, 0, self.grid_shape[0] - 1)
        col = np.clip(col, 0, self.grid_shape[1] - 1)

        self.state = (row, col)

        # Check if the goal state has been reached.
        done = self.state == self.goal_state
        # Assign a reward. Here we give -1 every step.
        # (If you want a +1 for reaching the goal, you can adjust accordingly.)
        reward = -1 if not done else -1

        return np.array(self.state), reward, done, {}

    def render(self, mode="human"):
        """Render the grid state. 'A' marks the agent, 'G' marks the goal."""
        grid = np.full(self.grid_shape, ".", dtype=str)
        grid[self.goal_state] = "G"  # Mark goal
        if self.state:
            grid[self.state] = "A"  # Mark agent
        print("\n".join(["".join(row) for row in grid]))
        print()


# Define a simple DQN model with a two-layer fully connected network.
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay Buffer for storing transitions (experience).
class ReplayBuffer:
    def __init__(self, capacity):
        # Use a deque to store transitions, with a maximum length of 'capacity'.
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        # Add a transition tuple (state, action, reward, next_state, done) to the buffer.
        self.buffer.append(transition)

    def sample(self, batch_size):
        # Randomly sample a batch of transitions from the buffer.
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        # Return the current size of the buffer.
        return len(self.buffer)


# Epsilon-greedy policy for action selection.
def epsilon_greedy(policy_net, state, epsilon):
    """Select an action using epsilon-greedy strategy."""
    if random.random() < epsilon:
        # With probability epsilon, choose a random action.
        return random.choice([0, 1, 2, 3])
    else:
        # Otherwise, choose the action with the highest Q-value.
        with torch.no_grad():
            return torch.argmax(policy_net(torch.tensor(state, dtype=torch.float32))).item()


# Train the DQN agent.
def train_dqn(env, episodes=1000, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
              batch_size=64, buffer_capacity=10000, target_update_freq=10, lr=0.001):
    # Initialize the policy and target networks.
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    policy_net = DQN(input_dim=state_dim, output_dim=action_dim)
    target_net = DQN(input_dim=state_dim, output_dim=action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Set up the optimizer and the replay buffer.
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    rewards_per_episode = []
    losses_per_episode = []  # Track average loss per episode

    # Training loop over multiple episodes.
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_losses = []  # To record losses in the current episode

        while True:
            # Choose an action using the epsilon-greedy policy.
            action = epsilon_greedy(policy_net, state, epsilon)

            # Take the action in the environment, get the next state and reward.
            next_state, reward, done, _ = env.step(action)

            # Store the transition in the replay buffer.
            replay_buffer.add((state, action, reward, next_state, done))

            # Update the current state and cumulative reward.
            state = next_state
            total_reward += reward

            # Only start training when there's enough data in the replay buffer.
            if len(replay_buffer) >= batch_size:
                # Sample a batch of transitions.
                states, actions, rewards_batch, next_states, dones = replay_buffer.sample(batch_size)

                # Compute Q-values for the current states and actions.
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                # Compute the maximum Q-values for the next states using the target network.
                next_q_values = target_net(next_states).max(1)[0]

                # Compute the target Q-values.
                target_q_values = rewards_batch + gamma * next_q_values * (1 - dones)

                # Compute the loss (MSE) between current Q-values and target Q-values.
                loss = nn.functional.mse_loss(q_values, target_q_values)

                # Backpropagation to update policy network parameters.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Record the loss for this update step.
                episode_losses.append(loss.item())

            # If the episode is done (goal reached or terminal state), break.
            if done:
                break

        # Decay epsilon for future episodes, ensuring it doesn't fall below epsilon_min.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update the target network periodically to stabilize learning.
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Record the total reward achieved in this episode.
        rewards_per_episode.append(total_reward)


        # Print out the episode's results.
        # Compute the average loss for this episode if any losses were recorded.
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        losses_per_episode.append(avg_loss)

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}, Avg Loss: {avg_loss:.4f}")

    return policy_net, rewards_per_episode,losses_per_episode


# Plot the total rewards over episodes to visualize training performance.
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Performance")
    plt.show()

def plot_losses(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("DQN Training Performance (Average Loss)")
    plt.show()

def plot_losses(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("DQN Training Performance (Average Loss)")
    plt.show()
# Print out the optimal policy derived from the trained model.
def print_optimal_policy(env, trained_model):
    """Evaluate the trained model at each state and print the chosen action."""
    print("Optimal Policy:")
    for row in range(env.grid_shape[0]):
        for col in range(env.grid_shape[1]):
            state = np.array([row, col])
            with torch.no_grad():
                action = torch.argmax(trained_model(torch.tensor(state, dtype=torch.float32))).item()
            action_str = ["Up", "Down", "Left", "Right"][action]
            print(f"({row}, {col}): {action_str}", end="  ")
        print()


# Main function to run the training and evaluation.
if __name__ == "__main__":
    # Instantiate the Windy Gridworld environment.
    env = WindyGridworldEnv()

    # Train the DQN agent on the environment.
    trained_model, rewards, losses = train_dqn(env)
    plot_losses(losses)

    # Plot the training rewards over episodes.
    plot_rewards(rewards)

    # Print the derived optimal policy.
    print_optimal_policy(env, trained_model)

    # Test the trained model by running a single episode and rendering.
    state = env.reset()
    env.render()
    total_reward = 0
    while True:
        # Select the best action according to the trained model.
        action = torch.argmax(trained_model(torch.tensor(state, dtype=torch.float32))).item()
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            print(f"Total Reward: {total_reward}")
            break
