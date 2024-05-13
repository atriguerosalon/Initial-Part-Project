import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

# Filter size
filter_size = 0.5
# Load the phi_NN_theoretical.npy array
phi_NN = np.load(f'./Symbolic_Regression_for_NN/phi_NN_theoretical_{filter_size}.npy')

# Load the characteristic direction and normalize it to a unit vector
principal_direction = np.load(f'./Symbolic_Regression_for_NN/characteristic_direction_{filter_size}.npy')
norm = np.linalg.norm(principal_direction)
principal_direction_unit = principal_direction / norm

# Determine the angle (theta) based on the normalized direction
theta = np.arctan2(principal_direction_unit[1], principal_direction_unit[0])
print(f"Principal Direction Angle in degrees(Theta): {np.rad2deg(theta)}")    
max_index = phi_NN.shape[0] - 1  # Assuming a square array
num_points = 2000  # Increase the number of points to sample more densely

# Determine the starting point
x_start, y_start = max_index, 0  # Lower-right corner (1,0)

# Calculate the maximum projected distance along the characteristic direction
horizontal_projection = max_index / abs(np.cos(theta)) if np.cos(theta) != 0 else float('inf')
vertical_projection = max_index / abs(np.sin(theta)) if np.sin(theta) != 0 else float('inf')
max_projected_distance = min(horizontal_projection, vertical_projection)

# Create evenly spaced values along the characteristic direction
s_values = np.linspace(0, max_projected_distance, num_points)

# Compute the coordinates along the characteristic direction using the angle
x_line = np.clip(x_start - s_values * np.cos(theta), 0, max_index)
y_line = np.clip(y_start + s_values * np.sin(theta), 0, max_index)

# Use map_coordinates to sample values along the characteristic line
values_along_line = map_coordinates(phi_NN, [y_line, x_line], order=1)

# Normalize the s_values to be between 0 and 1
s_values_normalized = s_values / s_values[-1]

# Save the extracted values to a file
np.save(f'./Symbolic_Regression_for_NN/values_along_characteristic_line_phi_NN_{filter_size}.npy', [s_values_normalized, values_along_line])

# Plot the extracted values
plt.plot(s_values_normalized, values_along_line)
plt.title('Values Along the Characteristic Direction')
plt.xlabel('Normalized Distance (s)')
plt.ylabel('Phi Values')
plt.grid(True)
plt.show()
