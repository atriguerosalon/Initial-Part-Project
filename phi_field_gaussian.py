import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
import os
from data_preparation import filename_to_field
# Constants and function definitions as provided earlier ...

if __name__ == '__main__':
    exclude_boundary = 0
    sigma_value = 2  # Example sigma value for Gaussian filter, adjust as needed
    data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
    data_path_reaction = 'wtemp-slice-B1-0000080000.raw'
    
    # Load data, calculate and normalize fields, and calculate phi
    wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, exclude_boundary)
    
    # Apply Gaussian filter to phi
    phi_filtered = apply_gaussian_filter(phi, sigma=sigma_value)
    
    # Plot original and filtered phi fields
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(phi, cmap='jet')
    ax[0].set_title('Original Phi Field')
    ax[1].imshow(phi_filtered, cmap='jet')
    ax[1].set_title(f'Filtered Phi Field (σ={sigma_value})')
    plt.show()

    # Optionally, save the plots as files
    if not os.path.exists('figs'):
        os.makedirs('figs')
    plt.savefig('figs/original_phi_field.png', dpi=300)
    plt.savefig('figs/filtered_phi_field_sigma_{sigma_value}.png', dpi=300)
