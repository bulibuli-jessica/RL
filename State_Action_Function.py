import numpy as np

# Define the states and their possible actions
states = ['Facebook', 'Class1', 'Class2', 'Class3', 'sleep']
actions = {
    'Facebook': ['Loop', 'Quit'],#Loop represent that still stay in Facebook, it is beneficial for distinguish
    'Class1': ['Facebook', 'Study'],#When the state is Class1,there are two action to choose is Facebook and Study respectively
    'Class2': ['Sleep', 'Study'],
    'Class3': ['Pub', 'Study'],
    'sleep': ['.']  # Terminal state
}

# List all state-action pairs and assign indices to them
state_action_pairs = []
state_action_to_index = {}
index_to_state_action = {}

index = 0  # Initialize index counter
for state in states:  # Loop over all states
    for action in actions[state]:  # Loop over possible actions for each state
        pair = (state, action)  # Create a state-action pair
        state_action_pairs.append(pair)  # Add the pair to the list
        state_action_to_index[pair] = index  # Map the pair to its index
        index_to_state_action[index] = pair
        index += 1  # Increment the index for the next pair

num_pairs = len(state_action_pairs)  # Total number of state-action pairs

# Initialize Q-values and the reward array (both initialized to zeros)
Q_old = np.zeros(num_pairs)  # Old Q-values (to be updated)
Q_new = np.zeros(num_pairs)  # New Q-values (the updated values)
R = np.zeros(num_pairs)  # Rewards for state-action pairs

# Define the reward function R
rewards = {
    ('Facebook', 'Loop'): -1,
    ('Facebook', 'Quit'): 0,
    ('Class1', 'Facebook'): -1,
    ('Class1', 'Study'): -2,
    ('Class2', 'Sleep'): 0,
    ('Class2', 'Study'): -2,
    ('Class3', 'Pub'): 1,
    ('Class3', 'Study'): 10,
    ('sleep', '.'): 0  # Terminal state
}

# Assign the reward values to the R array
for pair, idx in state_action_to_index.items():
    R[idx] = rewards[pair]

gamma = 0.9  # Discount factor
threshold = 1e-6  # Convergence threshold
max_iterations = 100  # Maximum number of iterations

for iteration in range(max_iterations):
    Q_old = Q_new.copy()# Keep a copy of the current Q-values for the next iteration


    # Update Q-values for each state-action pair
    for pair in state_action_pairs:
        idx = state_action_to_index[pair]
        state, action = pair

        if state == 'Facebook':
            if action == 'Loop':
                idx_Loop = state_action_to_index[('Facebook', 'Loop')]
                idx_Quit = state_action_to_index[('Facebook', 'Quit')]
                Q_new[idx] = R[idx] + gamma * (
                    0.5 * Q_old[idx_Loop] + 0.5 * Q_old[idx_Quit]
                )
            elif action == 'Quit':
                idx_Class1_Facebook = state_action_to_index[('Class1', 'Facebook')]
                idx_Class1_Study = state_action_to_index[('Class1', 'Study')]
                Q_new[idx] = R[idx] + gamma * (
                    0.5 * Q_old[idx_Class1_Facebook] + 0.5 * Q_old[idx_Class1_Study]
                )
        elif state == 'Class1':
            if action == 'Facebook':
                idx_Loop = state_action_to_index[('Facebook', 'Loop')]
                idx_Quit = state_action_to_index[('Facebook', 'Quit')]
                Q_new[idx] = R[idx] + gamma * (
                    0.5 * Q_old[idx_Loop] + 0.5 * Q_old[idx_Quit]
                )
            elif action == 'Study':
                idx_Class2_Sleep = state_action_to_index[('Class2', 'Sleep')]
                idx_Class2_Study = state_action_to_index[('Class2', 'Study')]
                Q_new[idx] = R[idx] + gamma * (
                    0.5 * Q_old[idx_Class2_Sleep] + 0.5 * Q_old[idx_Class2_Study]
                )
        elif state == 'Class2':
            if action == 'Sleep':
                idx_sleep_dot = state_action_to_index[('sleep', '.')]
                Q_new[idx] = R[idx] + gamma * Q_old[idx_sleep_dot]
            elif action == 'Study':
                idx_Class3_Pub = state_action_to_index[('Class3', 'Pub')]
                idx_Class3_Study = state_action_to_index[('Class3', 'Study')]
                Q_new[idx] = R[idx] + gamma * (
                    0.5 * Q_old[idx_Class3_Pub] + 0.5 * Q_old[idx_Class3_Study]
                )
        elif state == 'Class3':
            if action == 'Pub':
                idx_Class1_Facebook = state_action_to_index[('Class1', 'Facebook')]
                idx_Class1_Study = state_action_to_index[('Class1', 'Study')]
                idx_Class2_Sleep = state_action_to_index[('Class2', 'Sleep')]
                idx_Class2_Study = state_action_to_index[('Class2', 'Study')]
                idx_Class3_Pub = state_action_to_index[('Class3', 'Pub')]
                idx_Class3_Study = state_action_to_index[('Class3', 'Study')]

                Q_new[idx] = R[idx] + gamma * (
                    0.2 * (0.5 * Q_old[idx_Class1_Facebook] + 0.5 * Q_old[idx_Class1_Study]) +
                    0.4 * (0.5 * Q_old[idx_Class2_Sleep] + 0.5 * Q_old[idx_Class2_Study]) +
                    0.4 * (0.5 * Q_old[idx_Class3_Pub] + 0.5 * Q_old[idx_Class3_Study])
                )
            elif action == 'Study':
                idx_sleep_dot = state_action_to_index[('sleep', '.')]
                Q_new[idx] = R[idx] + gamma * Q_old[idx_sleep_dot]
        elif state == 'sleep':
            # Terminal state, set Q-value to 0
            Q_new[idx] = 0
    print(f"Iteration {iteration + 1}:")
    for idx in range(num_pairs):
        pair = index_to_state_action[idx]
        q_value = Q_new[idx]
        print(f"Q{pair} = {q_value:.6f}")
    print('-' * 50)
    # Check for convergence
    delta = np.max(np.abs(Q_new - Q_old))
    if delta < threshold:
        print(f"The algorithm converged after {iteration + 1} iterations.")
        break
else:
    print("The algorithm did not converge within the maximum number of iterations.")

# Print the final Q-values
print("\nFinal Q-values:")
for idx in range(num_pairs):
    pair = index_to_state_action[idx]
    q_value = Q_new[idx]
    print(f"Q{pair} = {q_value:.6f}")

# Extract the optimal policy
policy = {}
for idx, (state, action) in enumerate(state_action_pairs):
    # Update the policy only if the current Q-value is greater than the existing one
    if state not in policy or Q_new[idx] > policy[state][1]:
        policy[state] = (action, Q_new[idx])

# Output the optimal policy
print('\nOptimal policy:')
for state in states:
    if state in policy:
        action, q_value = policy[state]
        print(f'In state "{state}", take action "{action}" with Q-value {q_value:.4f}')
    else:
        print(f'No available actions for state "{state}".')
