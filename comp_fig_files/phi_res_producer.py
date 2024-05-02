import numpy as np
import matplotlib.pyplot as plt
#import data_preparation, which is a file the parent directory
from data_preparation import f_exclude_boundary, filename_to_field


# Data paths
data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
data_path_reaction = 'wtemp-slice-B1-0000080000.raw'

n_points = 100  # Number of points for each axis in the grid
precision = 0.01  # The precision for the grid
nx, ny = 384, 384  # The shape of the reaction rate and temperature gradient fields

# Generating example fields (Replace these with your actual data)
# Assume reaction rate and temperature gradient fields range between 0 and 1
filter_size = 1
left_exclusion, right_exclusion = f_exclude_boundary(filter_size)

# Load the data and calculate the fields
wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, (left_exclusion, right_exclusion))


# Generating an example error array (Replace this with your actual error data)
error_array = np.random.rand(nx, ny) * 0.1  # Error values between 0 and 0.1 for illustration

# Initialize a grid to hold the aggregated errors
error_grid = np.zeros((n_points, n_points))

#Create grid with all possible values of temperature gradient and reaction rate
nabla_temp_values = np.linspace(0, 1, n_points)
reaction_rate_values = np.linspace(0, 1, n_points)
mesh_grid = np.meshgrid(nabla_temp_values, reaction_rate_values)

print(mesh_grid)

# Calculate the indices in the error grid for each point in the fields
for i in range(nx):
    for j in range(ny):
        # Get the error for the current point
        error = error_array[i, j]
        
        # Find the corresponding indices in the grid
        nabla_temp_index = int(ct_field_star[i, j] / precision)
        reaction_rate_index = int(wcr_field_star[i, j] / precision)
        
        # Increment the error value in the corresponding grid cell
        # Note: np.clip ensures that the index does not go out of bounds
        nabla_temp_index = np.clip(nabla_temp_index, 0, n_points - 1)
        reaction_rate_index = np.clip(reaction_rate_index, 0, n_points - 1)
        error_grid[reaction_rate_index, nabla_temp_index] += error

# Plotting the error grid
# The x and y axes will represent the temperature gradient and reaction rate, respectively
plt.figure(figsize=(8, 6))
plt.imshow(error_grid, extent=(0, 1, 0, 1), origin='lower', interpolation='nearest', aspect='auto')
plt.colorbar(label='Aggregated Error')
plt.xlabel('Temperature Gradient (Normalized)')
plt.ylabel('Reaction Rate (Normalized)')
plt.title('Error Distribution Across Temperature Gradient and Reaction Rate')
plt.show()