import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from data_preparation import filename_to_field, create_custom_cmap
import os

# Add desired font settings
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# Define the standard deviation for the Gaussian filter

# filter size in cell number - 25 means length of 25 cells
fwidth_n = np.array([25, 37, 49, 62, 74, 86, 99])
filter_size = fwidth_n[2]

# std. deviation
sigma_value = np.sqrt(filter_size ** 2 / 12.0)
print(f'sigma_value: {sigma_value}')
# Exclusion boundaries
base_exclusion_left = 0
base_exclusion_right = 0
additional_exclusion = 0.5 * filter_size  # Adjust according to cell size if needed

left_exclusion = base_exclusion_left + additional_exclusion
right_exclusion = base_exclusion_right + additional_exclusion
exclude_boundary = int(left_exclusion), int(right_exclusion)

# Data paths
data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
data_path_reaction = 'wtemp-slice-B1-0000080000.raw'

# Load data, calculate and normalize fields, and calculate phi
wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, exclude_boundary)

# WHYYYYYYY
phi = np.flipud(phi)

# Apply Gaussian filters
wcr_field_filtered = gaussian_filter(wcr_field_star, sigma=sigma_value)
ct_field_filtered = gaussian_filter(ct_field_star, sigma=sigma_value)
phi_res_filtered = gaussian_filter(phi, sigma=sigma_value)

#Create custom color map
white_jet = create_custom_cmap()

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(10, 6))  # Adjust the figure size as needed

# Set common extent for all plots
original_extent_mm = [0, 10, 0, 10]  # [left, right, bottom, top] in mm
new_horizontal_extent_start_mm = 10 * left_exclusion / 384
new_horizontal_extent_end_mm = 10 - 10 * right_exclusion / 384
    
extent_mm = [new_horizontal_extent_start_mm, new_horizontal_extent_end_mm, 0, 10]

# Plotting the original fields
axs[0, 0].imshow(wcr_field_star, cmap=white_jet, extent=extent_mm)
#axs[0, 0].set_title('(a) ωcT*')

axs[0, 1].imshow(ct_field_star, cmap=white_jet, extent=extent_mm)
#axs[0, 1].set_title('(b) |∇cT|*')

axs[0, 2].imshow(phi, cmap=white_jet, extent=extent_mm)
#axs[0, 2].set_title('(c) Φres')

# Plotting the filtered fields
axs[1, 0].imshow(wcr_field_filtered, cmap=white_jet, extent=extent_mm)
#axs[1, 0].set_title('(d) ωcT* filtered')

axs[1, 1].imshow(ct_field_filtered, cmap=white_jet, extent=extent_mm)
#axs[1, 1].set_title('(e) |∇cT|* filtered')

axs[1, 2].imshow(phi_res_filtered, cmap=white_jet, extent=extent_mm)
#axs[1, 2].set_title('(f) Φres filtered')

# Set the labels and titles as per your requirement
for ax in axs.flat:
    #x and y in latex format font
    ax.set_xlabel(r'$x$ (mm)', fontsize=15)
    ax.set_ylabel(r'$y$ (mm)', fontsize=15)

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.label_outer()

subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

# Label the top left of each subplot
for i, ax in enumerate(axs.flat):
    if i == 0 or i == 3:
        offset = -0.2
    else:
        offset = -0.1

    # Position the text in the top left of the subplot
    ax.text(offset, 1.1, subplot_labels[i], transform=ax.transAxes, fontsize=16, 
            fontweight='normal', va='top', ha='left', color='black')
    
# Adjust the subplots to make space for the colorbar
fig.subplots_adjust(left=0.05, right=0.85, bottom=0.10, top=0.95, wspace=0.1, hspace=0.1)

# Add the colorbar to the right of the subplots
cbar_ax = fig.add_axes([0.87, 0.1, 0.025, 0.85])

# Choose any image for creating the colorbar since all images use the same colormap and range
im = axs[0, 2].imshow(phi, cmap=white_jet, extent=extent_mm)

# Create the colorbar
cbar =fig.colorbar(im, cax=cbar_ax)

#Set colorbar font size
cbar.ax.tick_params(labelsize=12)

#Save figure as pdf
if not os.path.exists('final_figs'):
    os.makedirs('final_figs')
plt.savefig('final_figs/' + f'Unfiltered_Filtered_plots_sigma_{sigma_value}_new.pdf', dpi=300)
plt.show()