import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
import os
from data_preparation import filename_to_field, create_custom_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Constants and function definitions as provided earlier ...

if __name__ == '__main__':
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    exclude_boundary = 5
    sigma_value = 5  # Adjust sigma value for Gaussian filter as needed
    data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
    data_path_reaction = 'wtemp-slice-B1-0000080000.raw'

    # Load data, calculate and normalize fields, and calculate phi
    wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, exclude_boundary)

    # Apply transformations if needed
    phi = phi.T

    wcr_field_star = wcr_field_star.T
    wcr_field_star = np.flipud(wcr_field_star)

    # Apply Gaussian filter to phi
    
    extent_mm = [0, 10, 0, 10]  # [left, right, bottom, top] in mm
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
    ax.contour(phi, levels=[0], colors='black', extent=extent_mm)

    #Change font of the numbers on the axes
    ax.tick_params(axis='both', which='major', labelsize=15)

    # Create a new axes on the right which will be used for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    #Set colorbar font size
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    #Color bar "Phi res" in greek letters
    #cbar.set_label('$\u03A6_{res}$', fontsize=15)

    # Save the figure
    if not os.path.exists('final_figs'):
        os.makedirs('final_figs')
    plt.savefig('final_figs/phi_contour.pdf', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    