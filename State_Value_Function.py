import numpy as np
import matplotlib.pyplot as plt

def next_v(gamma, r_list, old_v, transition_matrix):
    # Compute new V using matrix operations: V(s) = R(s) + γ * P(s'|s) * V(s')
    new_v = r_list + gamma * transition_matrix.dot(old_v)
    return new_v

if __name__ == '__main__':
    # Discount factor γ
    my_gamma = 0.9

    # Reward vector R
    my_r_list = np.array([-2., -2., -2., 10., 1., -1., 0.])
    # immediate rewards for each state. The order of these reards is Class1,Class2,Class3,Pass,Pub,FB,and Sleep.
    # initialize an arrary all of zeros
    my_old_v = np.zeros(len(my_r_list))

    # Transition probability matrix
    transition_matrix = np.array([
        [0, 0.5, 0, 0, 0, 0.5, 0],   # Class1
        [0, 0, 0.8, 0, 0, 0, 0.2],   # Class2
        [0, 0, 0, 0.6, 0.4, 0, 0],   # Class3
        [0, 0, 0, 0, 0, 0, 1],       # Pass
        [0.2, 0.4, 0.4, 0, 0, 0, 0], # Pub
        [0.1, 0, 0, 0, 0, 0.9, 0],   # Facebook
        [0, 0, 0, 0, 0, 0, 1]        # Sleep
    ])

    # Perform value iteration
    '''
    The value iteration runs for 100 iterations. 
    In each iteration, we call the next_v function to update the value function V, 
    and print current value function.
    
    '''

    for i in range(100):
        my_new_v = next_v(my_gamma, my_r_list, my_old_v, transition_matrix)

        # Print the value of V for each iteration
        print(f"Iteration {i + 1}: V = {[f'{v:.2f}' for v in my_new_v]}")

        # Check for convergence
        if np.max(np.abs(my_new_v - my_old_v)) < 1e-4:
            print(f"Converged at iteration {i + 1}")
            break

        # Update old V with the new values
        my_old_v = my_new_v

    # Final V values
    print("\nMRP:")
    states = ["Class1", "Class2", "Class3", "Pass", "Pub", "Facebook", "Sleep"]
    for state, value in zip(states, my_new_v):
        print(f"{state}: {value:.2f}")
