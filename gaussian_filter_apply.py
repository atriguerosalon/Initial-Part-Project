from scipy.ndimage import gaussian_filter
from data_preparation import filename_to_field
from matplotlib import pyplot as plt
import numpy as np
import os

# Define the Gaussian filter size (standard deviation), for 1 to 10, in steps of 0.5
data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
data_path_reaction = 'wtemp-slice-B1-0000080000.raw'

exclude_boundaries = (5,5)
wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, exclude_boundaries)
sigma_steps = np.arange(1, 2, 0.5)
#Show size of wcr_field_star
print(f'size of wcr_field_star: {wcr_field_star.shape}')

#Now rotate around the y=x axis
wcr_field_star = wcr_field_star.T
ct_field_star = ct_field_star.T
phi = phi.T

# Apply Gaussian filter to the fields
def apply_gaussian(sigma, exclude_boundaries):
    data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
    data_path_reaction = 'wtemp-slice-B1-0000080000.raw'

    wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, exclude_boundaries)

    wcr_field_filtered = gaussian_filter(wcr_field_star, sigma=sigma)
    ct_field_filtered = gaussian_filter(ct_field_star, sigma=sigma)
    phi_filtered = gaussian_filter(phi, sigma=sigma)

    return wcr_field_filtered, ct_field_filtered, phi_filtered

def run_apply_gaussian():
    for sigma in sigma_steps:
        # Apply Gaussian filter to the wcr_field
        wcr_field_filtered = gaussian_filter(wcr_field_star, sigma=sigma)
        # Now, visualize the filtered field
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(wcr_field_star, cmap='hot', origin='lower')
        plt.title('Original ωcT Field')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(wcr_field_filtered, cmap='hot', origin='lower')
        plt.title(f'Filtered ωcT Field (σ={sigma})')
        plt.colorbar()

        plt.tight_layout()
        
        if not os.path.exists('figs'):
            os.makedirs('figs')
        plt.savefig('figs/' + f'wcr_field_filtered_sigma_{sigma}_new.pdf', dpi=300)
        #Save figure as pdf with sigma value on the filename
        #plt.savefig(f'wcr_field_filtered_sigma_{sigma}.pdf')