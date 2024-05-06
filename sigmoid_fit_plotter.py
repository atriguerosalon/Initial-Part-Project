import numpy as np
import matplotlib.pyplot as plt

# Simulated 1D sigmoid fit data (replace with actual values from your fit)
#x_prime_fit = np.linspace(-3, 3, 100)  # Adjust as necessary
#y_prime_fit = 1 / (1 + np.exp(-1.5 * (x_prime_fit - 0)))  # Adjust as necessary
#Load the actual data here, 1st column is x_prime_fit, 2nd column is y_prime_fit
x_prime_fit, y_prime_fit = np.load('sigmoid_fit_values.npy')

# Convert 1D fit to 2D coordinates in the desired grid space (same size as x_prime_fit, y_prime_fit)
x_grid = np.linspace(0, 1, len(x_prime_fit	))
y_grid = np.linspace(0, 1, len(y_prime_fit))  # Replace with appropriate range for the y-axis
xx, yy = np.meshgrid(x_grid, y_grid)

# Function to align fitted 1D data to 2D grid (rotation and translation)
def rotate_and_translate(x_prime, y_prime, x_grid, y_grid):
    # Assuming the characteristic direction is at -45 degrees (slope = -1)
    angle = -45 * np.pi / 180  # In radians
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    grid_points = np.vstack((xx.ravel(), yy.ravel())).T
    transformed_points = np.dot(grid_points - [0.5, 0.5], rotation_matrix) + [0.5, 0.5]
    
    # Mapping the 1D sigmoid data onto the grid coordinates
    x_prime_values = np.interp(transformed_points[:, 0], x_grid, x_prime_fit)
    y_prime_values = np.interp(x_prime_values, x_prime_fit, y_prime_fit)
    return y_prime_values.reshape(xx.shape)

# Apply fitted sigmoid data to 2D grid
fitted_2d_values = rotate_and_translate(x_prime_fit, y_prime_fit, x_grid, y_grid)

# Plot the results
plt.imshow(fitted_2d_values, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar()
plt.xlabel('Reaction Rate (Normalized)')
plt.ylabel('Temperature Gradient (Normalized)')
plt.title('Fitted 2D Sigmoid Function')
plt.show()
