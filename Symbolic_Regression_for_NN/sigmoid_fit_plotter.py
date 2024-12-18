import numpy as np
import matplotlib.pyplot as plt

# Filter size
filter_size = 1
# Load the actual x and y coordinates for the sigmoid values
# The file should contain pairs: (x, y) and the associated sigmoid values
_, fitted_values = np.load(f'./Symbolic_Regression_for_NN/char_line_sigmoid_fitted_values_{filter_size}.npy')
x_prime_fit, y_prime_fit = np.load(f'./Symbolic_Regression_for_NN/char_line_xy_values_{filter_size}.npy')
# Create grid points for interpolation
# Set the grid size according to the number of points in the fitted values
grid_size = len(fitted_values)  
print(grid_size) # Gives 1000
print(x_prime_fit.shape) # Gives (1000,)
x_grid = np.linspace(0, 5, grid_size)
y_grid = np.linspace(0, 5, grid_size)
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

# Set colorbar label font size to 12
plt.colorbar(label="$\\overline{\\Phi}_{Sig}$")

# Extract the fitted sigmoid parameters
fitted_params = np.load(f'./Symbolic_Regression_for_NN/char_line_sigmoid_parameters_{filter_size}.npy')
beta_opt, x0_opt = fitted_params

#Plot text with the sigmoid function
#plt.text(0.5, 0.5, f'Sigmoid Function:\n1 / (1 + exp({beta_opt:.2f} * (x - {x0_opt:.2f})))', fontsize=12, ha='center', c='white')

# Save the array of the fitted 2D sigmoid function
np.save(f'./Symbolic_Regression_for_NN/fitted_2d_sigmoid_{filter_size}.npy', fitted_2d_values)

plt.xlabel('$\\overline{\\omega}{c{T}}^*$', fontsize=12)
plt.ylabel('$| \\nabla \\tilde{c}_{T}|^*$', fontsize=12)
#plt.title(f'Sigmoid Function:\n1 / (1 + exp({beta_opt:.2f} * (s - {x0_opt:.2f})))')

#Save the figure
plt.savefig(f'./Symbolic_Regression_for_NN/fitted_2d_sigmoid_{filter_size}.pdf')
#Save the figure in python format
plt.savefig(f'./Symbolic_Regression_for_NN/fitted_2d_sigmoid_{filter_size}.svg')
plt.show()


