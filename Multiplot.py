import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from data_preparation import filename_to_field, create_custom_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable



# Define the standard deviation for the Gaussian filter
exclude_boundary = 4
sigma_value = 5  # Example sigma value for Gaussian filter, adjust as needed
data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
data_path_reaction = 'wtemp-slice-B1-0000080000.raw'

# Load data, calculate and normalize fields, and calculate phi
wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, exclude_boundary)

#WHY IS THIS A THING!!!
phi = phi.T
phi = np.flipud(phi)

wcr_field_star = wcr_field_star.T
wcr_field_star = np.flipud(wcr_field_star)

ct_field_star = ct_field_star.T
ct_field_star = np.flipud(ct_field_star)

# Apply Gaussian filters
wcr_field_filtered = gaussian_filter(wcr_field_star, sigma=sigma_value)
ct_field_filtered = gaussian_filter(ct_field_star, sigma=sigma_value)
phi_res_filtered = gaussian_filter(phi, sigma=sigma_value)

#Create custom color map
white_jet = create_custom_cmap()

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # Adjust the figure size as needed

# Set common extent for all plots
extent_mm = [0, 10, 0, 10]  # [left, right, bottom, top] in mm

# Plotting the original fields
axs[0, 0].imshow(wcr_field_star, cmap=white_jet, extent=extent_mm)
axs[0, 0].set_title('(a) ωcT*')

axs[0, 1].imshow(ct_field_star, cmap=white_jet, extent=extent_mm)
axs[0, 1].set_title('(b) |∇cT|*')

axs[0, 2].imshow(phi, cmap=white_jet, extent=extent_mm)
axs[0, 2].set_title('(c) Φres')

# Plotting the filtered fields
axs[1, 0].imshow(wcr_field_filtered, cmap=white_jet, extent=extent_mm)
axs[1, 0].set_title('(d) ωcT* filtered')

axs[1, 1].imshow(ct_field_filtered, cmap=white_jet, extent=extent_mm)
axs[1, 1].set_title('(e) |∇cT|* filtered')

axs[1, 2].imshow(phi_res_filtered, cmap=white_jet, extent=extent_mm)
axs[1, 2].set_title('(f) Φres filtered')

# Set the labels and titles as per your requirement
for ax in axs.flat:
    ax.set(xlabel='x (mm)', ylabel='y (mm)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# Adjust layout to create space for the colorbar
fig.subplots_adjust(right=0.8)

# Create an axis for the colorbar on the right side of the figure
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

# Plot one of the images to use for the colorbar
# Since all images have the same colormap, it doesn't matter which one we choose
im = axs[0, 0].imshow(wcr_field_star, cmap=white_jet, extent=extent_mm)

# Create the colorbar
cbar = fig.colorbar(im, cax=cbar_ax)

# Set colorbar label
cbar.set_label('Normalized Value', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()
