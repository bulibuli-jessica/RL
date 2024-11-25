import numpy as np
import matplotlib.pyplot as plt

# Fix random seed
np.random.seed(42)

# Define constants
ACTION_LEFT = 0
ACTION_RIGHT = 1
VALUES = np.zeros(7, dtype=np.float64)  # High-precision initial estimates

TRUE_VALUES = {
    0: 0.0,
    1: 0.056,
    2: 0.126,
    3: 0.241,
    4: 0.416,
    5: 0.683,
    6: 1.0
}

GAMMA = 0.9  # Discount factor


def temporal_difference(values, alpha, max_steps=2000):
    """
    Temporal Difference (TD(0)) update value function
    """
    state = 3  # Initial state
    steps = 0
    while True:
        old_state = state
        action = np.random.binomial(1, 0.5)
        state = state - 1 if action == ACTION_LEFT else state + 1
        reward = 0 if state != 6 else 1
        values[old_state] += alpha * (reward + GAMMA * values[state] - values[old_state])
        steps += 1
        if state == 0 or state == 6 or steps >= max_steps:
            break


def rms_error(print_values=True):
    """
    Calculate RMS Error and plot error curves under different alpha
    """
    td_alphas = [0.01, 0.03, 0.05]  # Learning rates
    episodes = 5000 + 1  # Total number of episodes
    runs = 1  # Number of independent runs (can adjust to more to observe smoother curves)

    for alpha in td_alphas:
        total_errors = np.zeros(episodes)
        for r in range(runs):  # Removed tqdm progress bar
            current_values = np.copy(VALUES)
            errors = []
            for episode in range(episodes):
                # Call TD to update value function
                temporal_difference(current_values, alpha)
                # Calculate RMS Error
                true_values_array = np.array([TRUE_VALUES[s] for s in range(len(current_values))])
                errors.append(np.sqrt(np.mean((current_values - true_values_array) ** 2)))
            total_errors += np.array(errors)

        total_errors /= runs
        plt.plot(total_errors, label=f'Alpha={alpha:.2f}')
        if print_values:
            print(f"\nAlpha: {alpha:.2f} | Final Value Function: {current_values}")

    # Plot settings
    plt.xlabel('Episodes')
    plt.ylabel('RMS Error')
    plt.title('RMS Error with γ = 0.9')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    rms_error(print_values=True)
