import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Filter size list
filter_size = 0.5

# Load the theoretical neural network data
phi_NN = np.load(f'./Symbolic_Regression_for_NN/phi_NN_theoretical_{filter_size}.npy')

# Calculate gradients (differences) along both x and y directions
grad_y, grad_x = np.gradient(phi_NN)

# Reshape gradients for PCA analysis
grad_vectors = np.vstack((grad_x.ravel(), grad_y.ravel())).T

# Apply PCA to the gradients to identify principal directions
pca = PCA(n_components=2)
pca.fit(grad_vectors)
principal_direction = pca.components_[0]
orthogonal_direction = pca.components_[1]

# Get angle in degrees of the principal direction
angle = np.arctan2(principal_direction[1], principal_direction[0]) * 180 / np.pi

# Obtain percentage of variance explained by the principal direction
variance_explained = pca.explained_variance_ratio_[0] * 100
print(f"Variance Explained by Principal Direction: {variance_explained:.2f}%")
# Print the characteristic direction
print(f"Principal Gradient Direction: {principal_direction}")
print(f"Orthogonal (Characteristic) Direction: {orthogonal_direction}")

# Plot the original data with characteristic direction
extent = [0, 1, 0, 1]  # Adjust based on original data dimensions if needed
plt.imshow(phi_NN, extent=extent, origin='lower', aspect='auto', cmap='viridis')
plt.quiver(
    [0.5], [0.5],  # Starting point of the arrows (center of plot)
    [principal_direction[0]], [principal_direction[1]], color='red',
    angles='xy', scale_units='xy', scale=0.2, label=f'Principal Gradient Direction, angle: {angle:.2f}°'
)
plt.quiver(
    [0.5], [0.5],  # Starting point of the arrows (center of plot)
    [orthogonal_direction[0]], [orthogonal_direction[1]], color='blue',
    angles='xy', scale_units='xy', scale=0.2, label='Orthogonal (Characteristic) Direction'
)

# Add the perpendicular line at (0.71, 0.34)
#plt.plot([0.71, 0.71 + orthogonal_direction[0]], [0.34, 0.34 + orthogonal_direction[1]], 'g-', label='Perpendicular Line')
#Save the principal direction to a file
np.save(f'./Symbolic_Regression_for_NN/characteristic_direction_{filter_size}.npy', principal_direction)

plt.colorbar(label='NN Prediction')
plt.xlabel('Normalized X-axis (Reaction Rate)')
plt.ylabel('Normalized Y-axis (Temperature Gradient)')

plt.legend()

            
plt.title(f'Characteristic Direction Identification, filter size: {filter_size}')
# Save the plot 
plt.savefig(f'./Symbolic_Regression_for_NN/characteristic_direction_{filter_size}.pdf')

plt.show()