import numpy as np
import matplotlib.pyplot as plt

# Assuming `training_data` is a list of tuples, where each tuple contains (reaction_rate, nabla_temp)
# For demonstration, creating a random list of such tuples
np.random.seed(0)  # For reproducible results
training_data = [(np.random.rand(), np.random.rand()) for _ in range(1000)]

# Define the resolution and range for reaction rate and nabla temp
resolution = 0.01
max_value = 1.0
grid_points = int(max_value / resolution) + 1

# Initialize the training density grid
training_density_grid = np.zeros((grid_points, grid_points))

# Populate the training density grid based on the training data points
for reaction_rate, nabla_temp in training_data:
    # Calculate the grid indices for the current data point
    reaction_rate_index = int(reaction_rate / resolution)
    nabla_temp_index = int(nabla_temp / resolution)
    
    # Increment the cell in the grid corresponding to the data point
    training_density_grid[reaction_rate_index, nabla_temp_index] += 1

# Normalize the training density grid by dividing by the maximum to get a density between 0 and 1
training_density_grid /= np.max(training_density_grid)

# Plotting the training density grid
plt.figure(figsize=(8, 6))
plt.imshow(training_density_grid, origin='lower', extent=(0, max_value, 0, max_value), aspect='auto')
plt.colorbar(label='Training Data Density')
plt.xlabel('Temperature Gradient (Normalized)')
plt.ylabel('Reaction Rate (Normalized)')
plt.title('Density of Training Data Across Temperature Gradient and Reaction Rate')
plt.show()



















import numpy as np
import matplotlib.pyplot as plt

# Constants for the grid
resolution = 0.01
max_value = 1.0
grid_points = int(max_value / resolution) + 1

# Initialize a grid to hold the density of the training data
training_data_density = np.zeros((grid_points, grid_points))

# Dummy training data arrays for reaction rate and nabla temp
# Replace these with the actual arrays from your training dataset
reaction_rate_values = np.random.rand(10000) * max_value
nabla_temp_values = np.random.rand(10000) * max_value

# Process the training data and populate the density grid
for rr, nt in zip(reaction_rate_values, nabla_temp_values):
    # Map the reaction rate and nabla temp values to grid indices
    rr_idx = min(int(rr / resolution), grid_points - 1)
    nt_idx = min(int(nt / resolution), grid_points - 1)
    
    # Increment the cell in the grid corresponding to the current values
    training_data_density[nt_idx, rr_idx] += 1

# Normalize the training data density grid
training_data_density /= np.max(training_data_density)

# Generate a mesh grid for plotting
x = np.linspace(0, max_value, grid_points)
y = np.linspace(0, max_value, grid_points)
X, Y = np.meshgrid(x, y)

# Plot the training data density using a filled contour plot
plt.figure(figsize=(8, 6))
contour_plot = plt.contourf(X, Y, training_data_density, levels=100, cmap='viridis', norm=plt.LogNorm())
plt.colorbar(contour_plot, label='Training Data Density')
plt.xlabel('Nabla Temp (Normalized)')
plt.ylabel('Reaction Rate (Normalized)')
plt.title('Training Data Density in Nabla Temp vs Reaction Rate Space')
plt.show()

