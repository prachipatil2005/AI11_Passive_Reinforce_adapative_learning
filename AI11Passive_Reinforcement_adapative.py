import numpy as np  # Import the NumPy library for numerical operations and array handling.

# Define a function to calculate the utility of a state.
def return_state_utility(v, T, u, reward, gamma):  # Parameters: 
    # v = current state utilities, T = transition probabilities, u = utilities of states, 
    # reward = immediate reward, gamma = discount factor
    action_array = np.zeros(4)  # Initialize an array to hold utilities for each action (4 possible actions).

    # Loop through each of the four possible actions.
    for action in range(0, 4):
        # Calculate the expected utility of taking the current action.
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:, :, action])))

    # Return the total utility: immediate reward + discounted maximum expected utility from actions.
    return reward + gamma * np.max(action_array)

# Define the main function that executes the program.
def main():
    # Initialize a NumPy array representing the utilities of states in a grid.
    v = np.array([[0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0, 0.0,]])  # Last state is the terminal state with utility 1.0.

    # Load the transition matrix from a .npy file, specifying the paths for the states.
    T = np.load("D:\\c drive\\cs\\sem5-6\\sem-5\\AI\\T.npy")  # This file should contain the transition probabilities.

    # Initialize an array representing the expected utilities for various states.
    u = np.array([[0.812, 0.868, 0.918, 1.0,
                   0.762, 0.0, 0.660, -1.0,
                   0.705, 0.655, 0.611, 0.388]])  # Example utilities for different states.

    reward = -0.4  # Set the immediate reward for being in the current state (penalty).
    gamma = 1.0  # Set the discount factor, indicating the importance of future rewards.

    # Calculate the utility of state (1, 1) using the defined utility function.
    utility_11 = return_state_utility(v, T, u, reward, gamma)

    # Print the calculated utility for state (1, 1).
    print("Utility of state(1,1):" + str(utility_11))

# Check if the script is run as the main program, and execute the main function.
if __name__ == "__main__":
    main()
