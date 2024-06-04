import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from data_preparation import filename_to_field, create_custom_cmap, f_exclude_boundary, filename_to_0th_order_fields, load_data
import os

# MISSING: Add the path to the folder containing the data files
# MAKE SURE TO CITE THE ADDED INFORMATION FOR THE 0th ORDER FILTERING
# Label size
label_size = 18

# Add desired font settings
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# filter size in cell number - 25 means length of 25 cells
fwidth_n = np.array([0, 25, 37, 49, 62, 74, 86, 99])
filter_sizes = [0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

idx = 3
filter_size = filter_sizes[idx]
# std. deviation
sigma_value = np.sqrt(fwidth_n[idx] ** 2 / 12.0)
print(f'sigma_value: {sigma_value}')
#hi Job
# Exclusion boundaries
'''	
base_exclusion_left = 
base_exclusion_right = 0
additional_exclusion = 0.5 * filter_size  # Adjust according to cell size if needed

left_exclusion = base_exclusion_left + additional_exclusion
right_exclusion = base_exclusion_right + additional_exclusion
'''
filter
exclude_boundary = f_exclude_boundary(filter_size)
left_exclusion, right_exclusion = exclude_boundary
print(f'left_exclusion: {left_exclusion}, right_exclusion: {right_exclusion}')
# Data paths
data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
data_path_reaction = 'wtemp-slice-B1-0000080000.raw'

data_path_temp_filtered = "Ref_0th_Fields\\tilde-nablatemp-slice-B1-0000080000-049.raw"
data_path_reaction_filtered ="Data_new_NN\\dataset_slice_B1_TS80\\bar-wtemp-slice-B1-0000080000-049.raw"
# Load data, calculate and normalize fields, and calculate phi
wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, exclude_boundary)
wcr_field_filtered, ct_field_filtered, phi_res_filtered = filename_to_0th_order_fields(data_path_temp_filtered, data_path_reaction_filtered, filter_size)
#wcr_field_filtered, ct_field_filtered = load_data(data_path_temp_filtered, data_path_reaction_filtered, exclude_boundary)
phi_res_filtered = gaussian_filter(phi, sigma=sigma_value)

#Create custom color map
white_jet = create_custom_cmap()

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(8, 6))  # Adjust the figure size as needed

# Set common extent for all plots
nx_original = 384
ny_original = 384
original_extent_mm = [0, 10, 0, 10]  # [left, right, bottom, top] in mm
new_horizontal_extent_start_mm = original_extent_mm[1] * left_exclusion / nx_original
new_horizontal_extent_end_mm = original_extent_mm[1] - original_extent_mm[1] * right_exclusion / ny_original
    
extent_mm = [new_horizontal_extent_start_mm, new_horizontal_extent_end_mm, 0, 10]

# Plotting the original fields
axs[0, 0].imshow(wcr_field_star, cmap=white_jet, extent=extent_mm, vmin=0, vmax=1)
# Maximum value in original wcr_field_star
max_wcr = np.max(wcr_field_star)
#axs[0, 0].set_title('(a) ωcT*')

axs[0, 1].imshow(ct_field_star, cmap=white_jet, extent=extent_mm, vmin=0, vmax=1)
# Maximum value in ct_field_star
max_ct = np.max(ct_field_star)
#axs[0, 1].set_title('(b) |∇cT|*')

axs[0, 2].imshow(phi, cmap=white_jet, extent=extent_mm, vmin=0, vmax=1)
# Maximum value in phi
max_phi = np.max(phi)
#axs[0, 2].set_title('(c) Φres')

#maximum value in filtered fields
max_filtered_wcr = np.max(wcr_field_filtered)
max_filtered_ct = np.max(ct_field_filtered)
max_filtered_phi = np.max(phi_res_filtered)
print(f"Maximum value in filtered fields: {max_filtered_wcr}, {max_filtered_ct}, {max_filtered_phi}")
# Plotting the filtered fields
axs[1, 0].imshow(wcr_field_filtered, cmap=white_jet, extent=extent_mm, vmin=0, vmax=1)
#axs[1, 0].set_title('(d) ωcT* filtered')

axs[1, 1].imshow(ct_field_filtered, cmap=white_jet, extent=extent_mm, vmin=0, vmax=1)
#axs[1, 1].set_title('(e) |∇cT|* filtered')

axs[1, 2].imshow(phi_res_filtered, cmap=white_jet, extent=extent_mm, vmin=0, vmax=1)
#axs[1, 2].set_title('(f) Φres filtered')

# Set the labels and titles as per your requirement
for ax in axs.flat:
    #x and y in latex format font
    ax.set_xlabel(r'$x$ (mm)', fontsize=label_size+2)
    ax.set_ylabel(r'$y$ (mm)', fontsize=label_size+2)

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=label_size)
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
fig.subplots_adjust(left=0.09, right=0.90, bottom=0.10, top=0.95, wspace=0.05, hspace=0.15)

# Add the colorbar to the right of the subplots
cbar_ax = fig.add_axes([0.92, 0.1, 0.025, 0.85])

# Choose any image for creating the colorbar since all images use the same colormap and range
# im = axs[0, 2].imshow(phi, cmap=white_jet, extent=extent_mm)
im = axs[0, 2].imshow(phi, cmap=white_jet, extent=extent_mm)

# Create the colorbar
cbar =fig.colorbar(im, cax=cbar_ax)

#Set colorbar font size
cbar.ax.tick_params(labelsize=label_size)

#Save figure as pdf
if not os.path.exists('final_figs'):
    os.makedirs('final_figs')
#Save fig with sigma value up to two decimal places
plt.savefig('final_figs/' + f'Final_Unfiltered_Filtered_plots_fsize_{filter_size}.pdf', dpi=300)
plt.show()