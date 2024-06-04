import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import torch
import os

# Constants
#LOOK INTO MAKING THSE AS INPUT VARIABLES IN THE FUTURE!!!
nx, ny = 384, 384
lx, ly = 0.01, 0.01  # [m]
dx, dy = lx / (nx - 1), ly / (ny - 1)
x = np.arange(0, lx + dx, dx)
y = np.arange(0, ly + dy, dy)
TU = 1500.000  # K (reactant temperature)
TB = 1623.47  # K (burn temperature)
MAX_WCT = 1996.8891
MAX_CT = 3931.0113
DTH=0.0012904903  # m
DU=0.2219636  # kg/m^3
SL= 1.6585735551  # m/s
WCT_NORM_FACTOR = DU*SL/DTH
NCT_NORM_FACTOR = 1/DTH
filter_sizes=[0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00]
fwidth_n = np.array([0, 25, 37, 49, 62, 74, 86, 99])

data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
data_path_reaction = 'wtemp-slice-B1-0000080000.raw'

#need to define the function for the nn
class MyNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, n_hidden3, output_size, dropout_prob=0.0003):
        super(MyNeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, n_hidden1, dtype=torch.float64)
        self.dropout1 = torch.nn.Dropout(p=dropout_prob)  # Dropout layer after the first hidden layer
        self.fc2 = torch.nn.Linear(n_hidden1, n_hidden2,dtype=torch.float64)
        self.dropout2 = torch.nn.Dropout(p=dropout_prob)  # Dropout layer after the second hidden layer
        self.fc3 = torch.nn.Linear(n_hidden2, n_hidden3, dtype=torch.float64)
        self.dropout3 = torch.nn.Dropout(p=dropout_prob)  # Dropout layer after the third hidden layer
        self.fc4 = torch.nn.Linear(n_hidden3, output_size, dtype=torch.float64)
        self.sigmoid = torch.nn.Sigmoid()  # Sigmoid activation function (can be replaced with other activations)

    def forward(self, x, training=True):
        out = self.fc1(x)
        out = self.dropout1(out) if training else out  # Apply dropout only during training
        out = self.sigmoid(out)

        out = self.fc2(out)
        out = self.dropout2(out) if training else out
        out = self.sigmoid(out)

        out = self.fc3(out)
        out = self.dropout3(out) if training else out
        out = self.sigmoid(out)

        out = self.fc4(out)
        return out

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

def normalize_fields(wcr_field, ct_field, max_wct, max_ct):
    wcr_field_star = wcr_field / max_wct
    ct_field_star = ct_field / max_ct
    return wcr_field_star, ct_field_star

def calculate_phi(wcr_field_star, ct_field_star):
    phi = np.zeros_like(wcr_field_star)
    phi[(wcr_field_star > 0.4) & (ct_field_star < 0.2)] = 1
    return phi

def calculate_phi_0th_order(wcr_field_star, ct_field_star, filter_size):
    #Run Gaussian filter before calculating phi
    # Filter sizes 0(DNS) 0.5dth 0.75dth 1dth 1.25dth 1.5dth 1.75dth 2dth
    B1_MAX_WCT = [1996.8891465, 1526.9898031, 1291.2278508, 1109.90621188, 1000.3838916, 
                  956.39128432, 907.51922871, 859.85215504]
    
    B1_MAX_NCT = [3931.0113, 1337.4787422, 915.33373068, 672.59978870, 517.83026364,
                  422.29734194, 352.81715368, 297.16564086]

    B1_DTH = 0.001290490
    B1_DU = 0.2219636
    B1_SL = 1.658573555
    B1_TU = 1500.000 
    B1_TB = 1623.47

    
    if filter_size == 0:
        return calculate_phi(wcr_field_star, ct_field_star)
    
    elif filter_size not in filter_sizes:
        #Raise an error if the filter size is not in the list
        raise ValueError(f"Filter size {filter_size} is not in the list of available filter sizes: {filter_sizes}")
    
    index = filter_sizes.index(filter_size)
    actual_filter_size = fwidth_n[index]

    sigma_value = np.sqrt(actual_filter_size ** 2 / 12.0)

    wcr_field_star_filtered = gaussian_filter(wcr_field_star, sigma=sigma_value)
    ct_field_star_filtered = gaussian_filter(ct_field_star, sigma=sigma_value)
    phi_0th = np.zeros_like(wcr_field_star_filtered)
    phi_0th[(wcr_field_star_filtered > 0.4) & (ct_field_star_filtered < 0.2)] = 1
    phi_0th = gaussian_filter(phi_0th, sigma=sigma_value)
    return phi_0th

def filename_to_0th_order_field(data_path_temp, data_path_reaction, filter_size):
    filter_sizes=[0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00]
    idx = filter_sizes.index(filter_size)
    # Filter sizes 0(DNS) 0.5dth 0.75dth 1dth 1.25dth 1.5dth 1.75dth 2dth
    B1_MAX_WCT = [1996.8891465, 1526.9898031, 1291.2278508, 1109.90621188, 1000.3838916, 
                  956.39128432, 907.51922871, 859.85215504]
    
    B1_MAX_NCT = [3931.0113, 1337.4787422, 915.33373068, 672.59978870, 517.83026364,
                  422.29734194, 352.81715368, 297.16564086]

    B1_DTH = 0.001290490
    B1_DU = 0.2219636
    B1_SL = 1.658573555
    B1_TU = 1500.000 
    B1_TB = 1623.47

    MAX_WCT = B1_MAX_WCT[idx]
    MAX_NCT = B1_MAX_NCT[idx]
    data_temp, data_reaction = load_data(data_path_temp, data_path_reaction, f_exclude_boundary(filter_size))
    wcr_field, ct_field = calculate_fields(data_temp, data_reaction, B1_TB, B1_TU)
    wcr_field_star, ct_field_star = normalize_fields(wcr_field, ct_field, MAX_WCT, MAX_NCT)
    phi_0th_order_unfiltered = calculate_phi(wcr_field_star, ct_field_star)
    phi_0th_order = gaussian_filter(phi_0th_order_unfiltered, sigma=sigma_value(filter_size))
    return phi_0th_order

def filename_to_field(data_path_temp, data_path_reaction, exclude_boundaries=(0,0), MAX_WCT=MAX_WCT, MAX_CT=MAX_CT, TB=TB, TU=TU):
    data_temp, data_reaction = load_data(data_path_temp, data_path_reaction, exclude_boundaries)
    wcr_field, ct_field = calculate_fields(data_temp, data_reaction, TB, TU)
    wcr_field_star, ct_field_star = normalize_fields(wcr_field, ct_field, MAX_WCT, MAX_CT)
    phi = calculate_phi(wcr_field_star, ct_field_star)
    return wcr_field_star, ct_field_star, phi

def get_non_dimensionalized_fields(data_path_temp, data_path_reaction, filter_size, TB=TB, TU=TU, WCT_NORM_FACTOR=WCT_NORM_FACTOR, NCT_NORM_FACTOR=NCT_NORM_FACTOR):
    data_temp, data_reaction = load_data(data_path_temp, data_path_reaction, f_exclude_boundary(filter_size))
    wcr_field, ct_field=calculate_fields(data_temp, data_reaction, TB, TU)
    wcr_bar_field = gaussian_filter(wcr_field, sigma=sigma_value(filter_size), mode='reflect')/WCT_NORM_FACTOR
    ct_bar_field = gaussian_filter(ct_field, sigma=sigma_value(filter_size), mode='reflect')/NCT_NORM_FACTOR
    return wcr_bar_field, ct_bar_field

def create_custom_cmap(res=1024, starting_val=0.2):
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


def load_phi_field_NN_new(epoch):
  filter_sizes=[0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00]
          #load NN
  model_path=r"Models_During_Training\\new_model_discretized{}.pt".format(epoch)
  model = torch.load(model_path)

  #save values for all fields in B1
  for i in range(len(filter_sizes)):
      wcr_bar_field, ct_bar_field=get_non_dimensionalized_fields(data_path_temp, data_path_reaction, filter_sizes[i])
      wcr_bar_plus_flat=wcr_bar_field.flatten()
      ct_bar_plus_flat=ct_bar_field.flatten()
      phi_NN_field_flat=[]
      for j in range(len(wcr_bar_plus_flat)):
          inputs=[wcr_bar_plus_flat[j], ct_bar_plus_flat[j], filter_sizes[i]/2]
          inputs_tensor=torch.tensor(inputs, dtype=torch.float64)
          phi_pred=model(inputs_tensor,training=False)
          phi_NN_field_flat.append(phi_pred.detach().numpy())
      phi_NN_field = np.array(phi_NN_field_flat).reshape(wcr_bar_field.shape)

      folder_path = f"Prediction Plots\\Epoch{epoch}"  # Replace 'path/to/your/new_folder' with the desired folder path

# Check if the folder already exists
      if not os.path.exists(folder_path):
        # If it doesn't exist, create the folder
        os.makedirs(folder_path)
      
      plt.imshow(phi_NN_field, cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()])
      plt.colorbar()
      plt.savefig(f"Prediction Plots\\Epoch{epoch}\\Filter{filter_sizes[i]}.png")
      plt.close()
      np.save(f"NewNNFields\\Field_Filter_{filter_sizes[i]}", phi_NN_field)

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

    load_phi_field_NN_new(40)
    
    """    
    
    #Plot phi zeroth from filter size 0.5 to 2.0, in the same figure

    fig, ax = plt.subplots(1, len(filter_sizes), figsize=(20, 10))

    for i, filter_size in enumerate(filter_sizes):
        #Apply gaussian filter to phi_0th_order
        sigma_val = sigma_value(filter_size)

        #Exclude boundaries
        left_exclusion, right_exclusion = f_exclude_boundary(filter_size)

        wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, (left_exclusion, right_exclusion))
        phi_0th_order = calculate_phi_0th_order(wcr_field_star, ct_field_star, filter_size)
        #Apply Gaussian filter
        #phi_0th_order = gaussian_filter(phi_0th_order, sigma=sigma_val)
        #Changed function definition to include gaussian within the function
        # Plot the phi 0th order
        ax[i].imshow(phi_0th_order, cmap=create_custom_hot_cmap())
        ax[i].set_title(f"Filter size: {filter_size}")
        #Save the figure to the folder named 'figs'
        if not os.path.exists('figs'):
            os.makedirs('figs')
        plt.savefig('figs/phi_0th_order.pdf', dpi=300)
        
    plt.show()
    """