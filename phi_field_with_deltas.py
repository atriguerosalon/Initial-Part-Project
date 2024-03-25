import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
import os
from data_preparation import filename_to_field, create_custom_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable #never used this before but it gets the job done


if __name__ == '__main__':
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    
    # filter size in cell number - 25 means length of 25 cells
    fwidth_n = np.array([25, 37, 49, 62, 74, 86, 99])
    #filter_size = fwidth_n[0]
    filter_size = 0
    # std. deviation: NOT USED IN THIS SCRIPT
    sigma_value = np.sqrt(filter_size ** 2 / 12.0)
    
    # Calculate the exclusion boundary based on the filter size
    base_exclusion_left = 0
    base_exclusion_right = 0
    additional_exclusion = 0.5 * filter_size  # Adjust according to cell size if needed

    left_exclusion = base_exclusion_left + additional_exclusion
    right_exclusion = base_exclusion_right + additional_exclusion
    exclude_boundary = int(left_exclusion), int(right_exclusion)
    
    data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
    data_path_reaction = 'wtemp-slice-B1-0000080000.raw'

    # Load data, calculate and normalize fields, and calculate phi
    wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, exclude_boundary)
    wcr_field_star = np.clip(wcr_field_star, None, 1) #Naughty line, ask Alejandro
    # Adjust the size of the plots
    nx_original = 384
    ny_original = 384
    original_extent_mm = [0, 10, 0, 10]  # [left, right, bottom, top] in mm
    new_horizontal_extent_start_mm = original_extent_mm[1] * left_exclusion / nx_original
    new_horizontal_extent_end_mm = original_extent_mm[1] - original_extent_mm[1] * right_exclusion / ny_original
        
    extent_mm = [new_horizontal_extent_start_mm, new_horizontal_extent_end_mm, 0, 10]  # [left, right, bottom, top] in mm
    white_jet = create_custom_cmap()
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the wcr field
    im = ax.imshow(wcr_field_star, cmap=white_jet, extent=extent_mm)

    #ax.set_title('Reaction Rate Field with Phi Contour')
    #Set label font size
    ax.set_xlabel('$x$ (mm)', fontsize=15)
    ax.set_ylabel('$y$ (mm)', fontsize=15)

    # Overlay the black contour of the phi field (not the filtered one)
    # Use levels=[0.5] to draw the contour at the middle of the phi range (0 and 1)
    ax.contour(phi, levels=[0], colors='black', extent=extent_mm, origin = 'upper')

    #Get the even indexes of the filter sizes
    dashed_line_positions = fwidth_n[::2]

    # Convert the filter sizes to mm
    dashed_line_positions_mm = [original_extent_mm[1] * dashed_line_positions[i] / nx_original for i in range(len(dashed_line_positions))]
    
    #Base exclusion left and right in mm
    base_exclusion_left = 10
    base_exclusion_right = 0
    base_exclusion_left_mm = original_extent_mm[1] * base_exclusion_left / nx_original
    base_exclusion_right_mm = original_extent_mm[1] * base_exclusion_right / nx_original
    for pos in dashed_line_positions_mm:
        # Left side dashed line
        base_exclusion_left_mm = original_extent_mm[1] * base_exclusion_left / nx_original
        ax.axvline(x=original_extent_mm[0] + pos + base_exclusion_left_mm, color='black', linestyle='--', linewidth=2)
        # Right side dashed line
        ax.axvline(x=original_extent_mm[1] - pos - base_exclusion_right_mm, color='black', linestyle='--', linewidth=2)

    #Change font of the numbers on the axes
    ax.tick_params(axis='both', which='major', labelsize=15)

    # Create a new axes on the right which will be used for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    #Set colorbar font size
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    # Save the figure
    if not os.path.exists('final_figs'):
        os.makedirs('final_figs')
    plt.savefig('final_figs/phi_contour_with_deltas.pdf', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
