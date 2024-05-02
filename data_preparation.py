import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

# Constants
#LOOK INTO MAKING THSE AS INPUT VARIABLES IN THE FUTURE!!!
nx, ny = 384, 384
lx, ly = 0.01, 0.01  # [m]
dx, dy = lx / (nx - 1), ly / (ny - 1)
x = np.arange(0, lx + dx, dx)
y = np.arange(0, ly + dy, dy)
TU = 1500.000  # K (reactant temperature)
TB = 1623.47  # K (burn temperature)
MAX_WCR = 1996.8891
MAX_CT = 3931.0113

filter_sizes=[0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00]
fwidth_n = np.array([0, 25, 37, 49, 62, 74, 86, 99])

# Function definitions
def exclude_boundaries(field, left_exclusion, right_exclusion):
    # Transpose and flip operations to match your data formatting needs
    field = np.flipud(field.T)
    # Only modify the second dimension (columns) of the array.
    if left_exclusion > 0 and right_exclusion > 0:
        # Print shape of the field before exclusion
        #print(f"Shape of field before exclusion: {field.shape}")
        cropped_field = field[:, left_exclusion:-right_exclusion or None]
        # Print shape of the field after exclusion
        #print(f"Shape of field after exclusion: {cropped_field.shape}")
        return cropped_field
    elif left_exclusion > 0:
        return field[:, left_exclusion:]
    elif right_exclusion > 0:
        return field[:, :-right_exclusion or None]
    else:
        return field

def f_exclude_boundary(filter_size):

  #Get the index of the filter_size
  index = filter_sizes.index(filter_size)
  actual_filter_size = fwidth_n[index]
  # Exclusion boundaries
  base_exclusion_left = 25
  base_exclusion_right = 0
  additional_exclusion = 0.5 * actual_filter_size  # Adjust according to cell size if needed

  left_exclusion = base_exclusion_left + additional_exclusion
  right_exclusion = base_exclusion_right + additional_exclusion
  return int(left_exclusion), int(right_exclusion)

def load_data(data_path_temp, data_path_reaction, exclude_boundary=(0,0)):
    # Unpack the exclude_boundary tuple
    left_exclude, right_exclude = exclude_boundary
    
    # Load the data and reshape
    data_temp = np.fromfile(data_path_temp, count=-1, dtype=np.float64).reshape(nx, ny)
    data_reaction = np.fromfile(data_path_reaction, count=-1, dtype=np.float64).reshape(nx, ny)
    
    # Apply the exclusion boundaries
    data_temp = exclude_boundaries(data_temp, left_exclude, right_exclude)
    data_reaction = exclude_boundaries(data_reaction, left_exclude, right_exclude)
    
    return data_temp, data_reaction
    
def sigma_value(filter_size):
  # Get the index of the filter_size
  index = filter_sizes.index(filter_size)
  actual_filter_size = fwidth_n[index]
  return np.sqrt(actual_filter_size ** 2 / 12.0)

def calculate_fields(data_temp, data_reaction, TB, TU):
    wcr_field = data_reaction / (TB - TU)
    ct_field = data_temp / (TB - TU)
    return wcr_field, ct_field

def normalize_fields(wcr_field, ct_field, max_wcr, max_ct):
    wcr_field_star = wcr_field / max_wcr
    ct_field_star = ct_field / max_ct
    return wcr_field_star, ct_field_star

def calculate_phi(wcr_field_star, ct_field_star):
    phi = np.zeros_like(wcr_field_star)
    phi[(wcr_field_star > 0.4) & (ct_field_star < 0.2)] = 1
    return phi

def calculate_phi_0th_order(wcr_field_star, ct_field_star, filter_size):
    #Run Gaussian filter before calculating phi


    if filter_size == 0:
        return calculate_phi(wcr_field_star, ct_field_star)
    
    elif filter_size not in filter_sizes:
        #Raise an error if the filter size is not in the list
        raise ValueError(f"Filter size {filter_size} is not in the list of available filter sizes: {filter_sizes}")
    
    index = filter_sizes.index(filter_size)
    actual_filter_size = fwidth_n[index]

    sigma_value = np.sqrt(actual_filter_size ** 2 / 12.0)

    wcr_field_star = gaussian_filter(wcr_field_star, sigma=sigma_value)
    ct_field_star = gaussian_filter(ct_field_star, sigma=sigma_value)
    phi = np.zeros_like(wcr_field_star)
    phi[(wcr_field_star > 0.4) & (ct_field_star < 0.2)] = 1
    return phi

def filename_to_field(data_path_temp, data_path_reaction, exclude_boundaries=(0,0)):
    data_temp, data_reaction = load_data(data_path_temp, data_path_reaction, exclude_boundaries)
    wcr_field, ct_field = calculate_fields(data_temp, data_reaction, TB, TU)
    wcr_field_star, ct_field_star = normalize_fields(wcr_field, ct_field, MAX_WCR, MAX_CT)
    phi = calculate_phi(wcr_field_star, ct_field_star)
    return wcr_field_star, ct_field_star, phi

def create_custom_cmap():
    res = 1024
    starting_val = 0.2
    jet = plt.cm.get_cmap('jet', res)

    # Modify the colormap to set values below 0.2 to white
    newcolors = jet(np.linspace(0, 1, res))
    newcolors[:int(res*starting_val), :] = np.array([1, 1, 1, 1])  # RGBA for white color
    new_jet = mcolors.LinearSegmentedColormap.from_list('white_jet', newcolors)
    return new_jet

def create_custom_hot_cmap():
    hot = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#4B006E'),
    (1e-20, '#4B006E'),
    (0.03, '#4169E1'),
    (0.15, '#adff5a'),
    (0.3, '#ffff5a'),
    (0.39, '#ff9932'),
    (0.6, '#D22B2B'),
    (1, '#D22B2B'),
], N=256)
    return hot

def plot_fields(wcr_field_star, ct_field_star, phi):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(wcr_field_star)
    ax[0].set_title('wcr field star')
    ax[1].imshow(ct_field_star)
    ax[1].set_title('ct field star')
    ax[2].imshow(phi)
    ax[2].set_title('phi field')
    plt.show()

def overlay_fields(phi, img_path, x, y, filename='overlay.pdf'):
    # Load the reference image
    img = mpimg.imread(img_path)
    white_jet = create_custom_cmap()
    phi_flipped = np.fliplr(phi)

    # Transpose the data if it is rotated 90 degrees
    phi_flipped = phi_flipped.T
    
    # Plot reference image
    fig, ax = plt.subplots()
    ax.imshow(phi_flipped, extent=[x.min(), x.max(), y.min(), y.max()], alpha=1, cmap=white_jet, aspect='auto')
    # Overlay the phi field with transparency
    cax = ax.imshow(img, extent=[x.min(), x.max(), y.min(), y.max()], alpha= 0.7)

    # Adding a colorbar for the overlay
    fig.colorbar(cax, ax=ax, orientation='vertical')

    # Save the figure to the folder named 'figs'
    if not os.path.exists('figs'):
        os.makedirs('figs')
    plt.savefig('figs/' + filename, dpi=300)

    plt.show()

if __name__ == '__main__':
    # Usage of functions

    # plot_fields(wcr_field_star, ct_field_star, phi)  # Uncomment to plot fields
    #overlay_fields(phi, 'figs/figure7.png', x, y)  # Adjust 'figure7.png' to your image's path
    #plot phi 0th order
    """
    phi_0th_order = calculate_phi_0th_order(wcr_field_star, ct_field_star, 1.0)
    #Use colorbar white_jet
    white_jet = create_custom_cmap()
    #Plot Phi 0th order side by side with Phi
    fig, ax = plt.subplots(1,2, figsize=(10, 10))
    ax[0].imshow(phi, cmap=white_jet)
    ax[1].imshow(phi_0th_order, cmap=white_jet)
    plt.show()
    """
    
    data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
    data_path_reaction = 'wtemp-slice-B1-0000080000.raw'
    #Plot phi zeroth from filter size 0.5 to 2.0, in the same figure
    filter_sizes=[0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00]
    fig, ax = plt.subplots(1, len(filter_sizes), figsize=(20, 10))

    for i, filter_size in enumerate(filter_sizes):
        #Apply gaussian filter to phi_0th_order
        sigma_val = sigma_value(filter_size)

        #Exclude boundaries
        left_exclusion, right_exclusion = f_exclude_boundary(filter_size)

        wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, (left_exclusion, right_exclusion))
        phi_0th_order = calculate_phi_0th_order(wcr_field_star, ct_field_star, filter_size)
        #Apply Gaussian filter
        phi_0th_order = gaussian_filter(phi_0th_order, sigma=sigma_val)

        # Plot the phi 0th order
        ax[i].imshow(phi_0th_order, cmap=create_custom_hot_cmap())
        ax[i].set_title(f"Filter size: {filter_size}")
        #Save the figure to the folder named 'figs'
        if not os.path.exists('figs'):
            os.makedirs('figs')
        plt.savefig('figs/phi_0th_order.pdf', dpi=300)
        
    plt.show()
