import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import matplotlib.colors as mcolors

# Constants
#LOOK INTO MAKING THSE AS INPUT VARIABLES IN THE FUTURE!!!
nx, ny = 384, 384
lx, ly = 0.01, 0.01  # [m]
dx, dy = lx / (nx - 1), ly / (ny - 1)
x = np.arange(0, lx + dx, dx)
y = np.arange(0, ly + dy, dy)
TU = 1500.000  # K
TB = 1623.47  # K
MAX_WCR = 1996.8891
MAX_CT = 3931.0113

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
    return np.flipud(phi)

def filename_to_field(data_path_temp, data_path_reaction, exclude_boundaries=(0,0)):
    data_temp, data_reaction = load_data(data_path_temp, data_path_reaction, exclude_boundaries)
    wcr_field, ct_field = calculate_fields(data_temp, data_reaction, TB, TU)
    wcr_field_star, ct_field_star = normalize_fields(wcr_field, ct_field, MAX_WCR, MAX_CT)
    phi = calculate_phi(wcr_field_star, ct_field_star)
    return wcr_field_star, ct_field_star, phi

def create_custom_cmap():
    jet = plt.cm.get_cmap('jet', 256)

    # Modify the colormap to set values below 0.2 to white
    newcolors = jet(np.linspace(0, 1, 256))
    newcolors[:int(256*0.2), :] = np.array([1, 1, 1, 1])  # RGBA for white color
    new_jet = mcolors.LinearSegmentedColormap.from_list('white_jet', newcolors)
    return new_jet

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
    exclude_boundary = (5,5)
    data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
    data_path_reaction = 'wtemp-slice-B1-0000080000.raw'
    wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, exclude_boundary)
    # plot_fields(wcr_field_star, ct_field_star, phi)  # Uncomment to plot fields
    overlay_fields(phi, 'figs/figure7.png', x, y)  # Adjust 'figure7.png' to your image's path
