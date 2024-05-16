import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Label size
label_size = 16
# Filter size list
filter_sizes = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
# Principal direction list
principal_directions = []
# Variance explained list
variance_explained_list = []

for filter_size in filter_sizes:
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
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

    # Append the principal direction to the list
    principal_directions.append(angle)

    # Append the variance explained to the list
    variance_explained_list.append(variance_explained)

    # Print the characteristic direction
    print(f"Principal Gradient Direction: {principal_direction}")
    print(f"Orthogonal (Characteristic) Direction: {orthogonal_direction}")

    # Plot the original data with characteristic direction
    extent = [0, 1, 0, 1]  # Adjust based on original data dimensions if needed
    plt.imshow(phi_NN, extent=extent, origin='lower', aspect='auto', cmap='viridis')
    plt.quiver(
        [0.5], [0.5],  # Starting point of the arrows (center of plot)
        [principal_direction[0]], [principal_direction[1]], color='red',
        angles='xy', scale_units='xy', scale=0.2, label=f'Principal Gradient Direction'#, angle: {angle:.2f}°', 
    )
    plt.quiver(
        [0.5], [0.5],  # Starting point of the arrows (center of plot)
        [orthogonal_direction[0]], [orthogonal_direction[1]], color='blue',
        angles='xy', scale_units='xy', scale=0.2, label='Orthogonal (Characteristic) Direction'
    )

    # Add text label with variance explained to the legend, pointing in the principal direction
    #plt.quiver(0.5, 0.5, principal_direction[0]/10, principal_direction[1]/10, color='white', scale=1, label=f'Variance Explained: {variance_explained:.2f}%')

    # Add the perpendicular line at (0.71, 0.34)
    #plt.plot([0.71, 0.71 + orthogonal_direction[0]], [0.34, 0.34 + orthogonal_direction[1]], 'g-', label='Perpendicular Line')
    #Save the principal direction to a file
    np.save(f'./Symbolic_Regression_for_NN/characteristic_direction_{filter_size}.npy', principal_direction)

    # Properly capture the colorbar object
    cbar = plt.colorbar()
    #cbar.set_label(label="$\\overline{\\Phi}_{NN}$", size=label_size)

    # Change tick size of the colorbar
    cbar.ax.tick_params(labelsize=label_size)  # Set tick label size


    plt.xlabel('$\\overline{\\omega}_{c_{T}}^+$', fontsize=label_size)
    plt.ylabel('$| \\nabla \\tilde{c}_{T}|^+$', fontsize=label_size)
    plt.xticks(fontsize=label_size)	
    plt.yticks(fontsize=label_size)
    plt.legend(loc='best', fontsize=label_size-1.5, edgecolor='black', fancybox=False).get_frame().set_linewidth(1)

                
    #plt.title(f'Characteristic Direction Identification, filter size: {filter_size}')
    # Save the plot plt.savefig('filename.png', bbox_inches='tight', pad_inches=0)

    plt.savefig(f'./Symbolic_Regression_for_NN/characteristic_direction_{filter_size}_loop.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    #plt.show()
plt.close()
# Plot the angles of the principal directions
plt.figure()

plt.plot(filter_sizes, principal_directions, marker='o', c='black')


# Set ticks to match the font size
plt.xticks(fontsize=label_size)	
plt.yticks(fontsize=label_size)

# Have faint grey dashed lines at the filter sizes
for filter_size in filter_sizes:
    plt.axvline(filter_size, color='grey', linestyle='--', linewidth=1, alpha=0.5)
# Set the ticks to match the filter sizes in numbers
plt.xticks(filter_sizes)

# Make the y-axis ticks go from 0 to 180	
#plt.yticks(np.arange(-90, 0, 10))
plt.xlabel('$\\Delta /\\delta_{th}$', fontsize=label_size)
plt.ylabel('$\\Theta$', fontsize=label_size)
#plt.title('Principal Direction Angle vs. Filter Size')
plt.savefig(f'./Symbolic_Regression_for_NN/principal_direction_angles.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()
# Plot the variance explained by the principal directions
plt.figure()
plt.plot(filter_sizes, variance_explained_list, marker='o')
plt.xticks(fontsize=label_size)	
plt.yticks(fontsize=label_size)
plt.xlabel('Filter Size')
plt.ylabel('Variance Explained (%)')
#plt.title('Variance Explained by Principal Direction vs. Filter Size')
plt.savefig(f'./Symbolic_Regression_for_NN/variance_explained.pdf')
