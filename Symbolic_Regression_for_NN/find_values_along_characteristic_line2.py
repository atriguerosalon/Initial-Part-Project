import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
# Filter size
filter_size = 1
# Load the phi_NN_theoretical array
phi_NN = np.load(f'./Symbolic_Regression_for_NN/phi_NN_theoretical_{filter_size}.npy')

# Load the characteristic direction and normalize it
principal_direction = np.load(f'./Symbolic_Regression_for_NN/characteristic_direction_{filter_size}.npy')
norm = np.linalg.norm(principal_direction)
principal_direction_unit = -principal_direction / norm

# Start from the lower-right corner (1,0) in normalized coordinates
max_index = phi_NN.shape[0] - 1
x_start, y_start = max_index, 0

# Calculate the angle based on the characteristic direction
theta = np.arctan2(principal_direction_unit[1], principal_direction_unit[0])
print(f"Angle (in degrees): {np.degrees(theta)}")

# Determine the number of points to sample and initialize the path coordinates
num_points = 1000
if filter_size == 0.5:
    s_values = np.linspace(0, (max_index - 68)  * np.sqrt(2), num_points) # -10.1 for the "" filter
if filter_size == '':
    s_values = np.linspace(0, (max_index - 10.1)  * np.sqrt(2), num_points) # -10.1 for the "" filter
if filter_size == 0.75:
    s_values = np.linspace(0, (max_index - 65)  * np.sqrt(2), num_points) # -10.1 for the "" filter
if filter_size == 1:
    s_values = np.linspace(0, (max_index - 62)  * np.sqrt(2), num_points) # -10.1 for the "" filter	
# Create coordinates along the characteristic direction
x_line = x_start + s_values * np.cos(theta)
y_line = y_start + s_values * np.sin(theta)

# Ensure coordinates remain within valid bounds
x_line = np.clip(x_line, 0, max_index)
y_line = np.clip(y_line, 0, max_index)

# Extract values along this path
values_along_line = map_coordinates(phi_NN, [y_line, x_line], order=1)

# Normalize the s_values to be between 0 and 1
s_values_normalized = s_values / s_values[-1]

# Save the extracted values to a file
np.save(f'./Symbolic_Regression_for_NN/values_along_characteristic_line_phi_NN_final_{filter_size}.npy', [[x_line/100, y_line/100], [s_values, values_along_line]])

# Plot the characteristic line on the same plot
plt.plot(x_line, y_line, color='red', label='Characteristic Line')
plt.legend()
plt.show()


# Plot the extracted values
plt.plot(s_values_normalized, values_along_line)
plt.title('Values Along the Characteristic Direction')
plt.xlabel('Normalized Distance (s)')
plt.ylabel('Phi Values')
plt.grid(True)
plt.show()
