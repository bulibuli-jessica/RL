import numpy as np
import matplotlib.pyplot as plt

# Define constants
ACTION_LEFT = 0
START_STATE = 3
LEFT_TERMINAL = 0
RIGHT_TERMINAL = 6
GAMMA = 0.9  # Discount factor

# True values (calculated based on symmetry)
true_values = {
    0: 0.0,
    1: 0.056,
    2: 0.126,
    3: 0.241,
    4: 0.416,
    5: 0.683,
    6: 1.0
}

def generate_episode():
    """
    Generates a single episode (trajectory) and returns the trajectory and the rewards received at each step.
    """
    state = START_STATE
    trajectory = [state]
    rewards = []  # Stores the reward received at each step

    while True:
        # Randomly select an action (left or right with equal probability)
        action = np.random.binomial(1, 0.5)  # 50% probability to choose left or right
        state += -1 if action == ACTION_LEFT else 1
        trajectory.append(state)

        # Determine the reward for the new state
        if state == RIGHT_TERMINAL:
            rewards.append(1.0)  # Reward for reaching the right terminal state
            break
        elif state == LEFT_TERMINAL:
            rewards.append(0.0)  # No reward for reaching the left terminal state
            break
        else:
            rewards.append(0.0)  # No reward for intermediate states

    return trajectory, rewards

def monte_carlo_batch(values, alpha=0.01, batch_size=10, threshold=1e-4, max_iterations=10000):
    """
    Implements the batch update version of the Monte Carlo method with a convergence check.
    Updates the value function estimates based on batches of episodes until the value function converges or the maximum number of iterations is reached.

    Parameters:
    values (array): The current estimates of the value function.
    alpha (float): The learning rate.
    batch_size (int): The number of episodes to generate in each batch.
    threshold (float): The convergence threshold for the value function updates.
    max_iterations (int): The maximum number of iterations to run.

    Returns:
    values (array): The updated estimates of the value function.
    """
    for iteration in range(max_iterations):
        trajectories = []
        returns_list = []
        values_old = values.copy()  # Save a copy of the old value function estimates for convergence checking

        # Generate a batch of episodes
        for _ in range(batch_size):
            trajectory, rewards = generate_episode()  # Generate one episode
            trajectories.append(trajectory)
            returns_list.append(rewards)

        # Perform batch updates of the value function estimates
        for trajectory, rewards in zip(trajectories, returns_list):
            G = 0  # Initialize the return (cumulative reward)
            # Iterate backwards through the trajectory to calculate returns and update value estimates
            for t in range(len(trajectory) - 2, -1, -1):  # Exclude the terminal state
                G = rewards[t] + GAMMA * G  # Compute the return G_t
                state = trajectory[t]
                if state == LEFT_TERMINAL or state == RIGHT_TERMINAL:
                    continue  # Skip updating terminal states
                # Update the value estimate for the state
                values[state] += alpha * (G - values[state])  # Incremental update

        # Compute the maximum change in value function estimates for convergence checking
        delta = max(abs(values[state] - values_old[state]) for state in range(len(values)))

        if delta < threshold:
            # print(f"Value function has converged at iteration {iteration}, maximum change: {delta:.6f}")
            break

    return values

def run_multiple_experiments(num_experiments=5, alpha=0.01, batch_size=10):
    """
    Runs multiple experiments to estimate the value function and stores the results of each experiment.

    Parameters:
    num_experiments (int): The number of experiments to run.
    alpha (float): The learning rate.
    batch_size (int): The number of episodes in each batch.

    Returns:
    average_values (array): The average value function estimates over all experiments.
    all_values (list of arrays): The list of value function estimates from each experiment.
    """
    results = []

    for i in range(num_experiments):
        # Initialize the value function estimates
        values = np.zeros(7)
        values[RIGHT_TERMINAL] = 1.0
        values[LEFT_TERMINAL] = 0.0
        # Run the batch Monte Carlo method
        monte_carlo_batch(values, alpha=alpha, batch_size=batch_size)
        results.append(values.copy())  # Make sure to copy the array to avoid referencing the same array

    # Compute the average value function estimates
    average_values = np.mean(results, axis=0)

    print("Average value function from multiple experiments:")
    for i, value in enumerate(average_values):
        print(f"V({i}) = {value:.3f}")

    return average_values, results

def plot_multiple_estimates(all_values, true_values):
    """
    Plots the estimated value functions from multiple experiments to illustrate instability.

    Parameters:
    all_values (list of arrays): The list of value function estimates from each experiment.
    true_values (dict): The true value function for each state.
    """
    states = list(range(len(true_values)))
    true_values_list = [true_values[state] for state in states]

    plt.figure(figsize=(10, 6))

    # Plot the true value function
    plt.plot(states, true_values_list, label="True Values", color="blue", linewidth=2)

    # Plot the estimated value functions from each experiment
    for idx, values in enumerate(all_values):
        estimated_values = [values[state] for state in states]
        plt.plot(states, estimated_values, label=f"Estimate {idx+1}", linestyle="--", marker='o', alpha=0.7)

    plt.xlabel("State")
    plt.ylabel("Value")
    plt.title("Estimated Value Functions from Multiple Experiments")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_comparison(values, true_values):
    """
    Visualizes the comparison between the true value function and the estimated value function.

    Parameters:
    values (array): The estimated value function.
    true_values (dict): The true value function for each state.
    """
    states = list(range(len(values)))
    estimated_values = [values[state] for state in states]
    true_values_list = [true_values[state] for state in states]

    plt.figure(figsize=(8, 6))
    plt.plot(states, true_values_list, label="True Values", color="blue", linewidth=2)
    plt.plot(states, estimated_values, label="Estimated Values", marker="o", linestyle="--", color="black")
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.title("Comparison of Estimated vs True Value Function")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main program entry point
if __name__ == "__main__":
    # Set experiment parameters
    num_experiments = 5  # Reduce the number of experiments to highlight instability
    alpha = 0.01  # Learning rate for value function updates
    batch_size = 10  # Number of episodes in each batch

    # Run multiple experiments and get all value function estimates
    average_values, all_values = run_multiple_experiments(num_experiments=num_experiments, alpha=alpha, batch_size=batch_size)

    # Visualize the estimated value functions from each experiment
    plot_multiple_estimates(all_values, true_values)

    # Optionally, plot the average estimated value function
    # plot_comparison(average_values, true_values)
