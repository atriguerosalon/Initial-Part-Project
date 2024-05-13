import numpy as np
import matplotlib.pyplot as plt

# Load the actual x and y coordinates for the sigmoid values
# The file should contain pairs: (x, y) and the associated sigmoid values
_, fitted_values = np.load('./Symbolic_Regression_for_NN/char_line_sigmoid_fitted_values.npy')
x_prime_fit, y_prime_fit = np.load('./Symbolic_Regression_for_NN/char_line_xy_values.npy')

# Create grid points for interpolation
grid_size = 100
x_grid = np.linspace(0, 1, grid_size)
y_grid = np.linspace(0, 1, grid_size)
xx, yy = np.meshgrid(x_grid, y_grid)

# Interpolate the fitted values onto the 2D grid
fitted_2d_values = np.zeros_like(xx)

for i in range(grid_size):
    for j in range(grid_size):
        # Calculate distances to each point in the sigmoid coordinates
        distances = np.sqrt((x_prime_fit - xx[i, j]) ** 2 + (y_prime_fit - yy[i, j]) ** 2)
        # Find the closest point and assign its value to the grid
        closest_point = np.argmin(distances)
        fitted_2d_values[i, j] = fitted_values[closest_point]

# Plot the 2D sigmoid function
plt.imshow(fitted_2d_values, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Fitted Sigmoid Values')

# Extract the fitted sigmoid parameters
fitted_params = np.load('./Symbolic_Regression_for_NN/char_line_sigmoid_parameters.npy')
beta_opt, x0_opt = fitted_params

#Plot text with the sigmoid function
#plt.text(0.5, 0.5, f'Sigmoid Function:\n1 / (1 + exp({beta_opt:.2f} * (x - {x0_opt:.2f})))', fontsize=12, ha='center', c='white')

plt.xlabel('Reaction Rate (Normalized)')
plt.ylabel('Temperature Gradient (Normalized)')
plt.title(f'Sigmoid Function:\n1 / (1 + exp({beta_opt:.2f} * (s - {x0_opt:.2f})))')

#Save the figure
plt.savefig('./Symbolic_Regression_for_NN/fitted_2d_sigmoid.pdf')
#Save the figure in python format
plt.savefig('./Symbolic_Regression_for_NN/fitted_2d_sigmoid.svg')
plt.show()


