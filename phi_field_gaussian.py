import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
import os
from data_preparation import filename_to_field, create_custom_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Constants and function definitions as provided earlier ...

if __name__ == '__main__':
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
    
    # Apply Gaussian filter to phi
    phi_filtered = gaussian_filter(phi, sigma=sigma_value)
    
    extent_mm = [0, 10, 0, 10]  # [left, right, bottom, top] in mm

    white_jet = create_custom_cmap()
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original phi field
    im0 = axs[0].imshow(wcr_field_star, cmap=white_jet, extent=extent_mm)
    axs[0].set_title('Reaction Rate Field')
    axs[0].set_xlabel('X (mm)')
    axs[0].set_ylabel('Y (mm)')

    # Create a new axes on the right which will be used for the colorbar
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im0, cax=cax)

    # Plot the filtered phi field
    im1 = axs[1].imshow(phi_filtered, cmap=white_jet, extent=extent_mm)
    axs[1].set_title(f'Filtered Phi Field (σ={sigma_value})')
    axs[1].set_xlabel('X (mm)')
    axs[1].set_ylabel('Y (mm)')

    # Create a new axes on the right which will be used for the colorbar
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)

    plt.tight_layout()
    plt.show()

    # Optionally, save the plots as files
    if not os.path.exists('figs'):
        os.makedirs('figs')
        
    plt.savefig('figs/filtered_phi_field_sigma_{sigma_value}.pdf', dpi=300)
