import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LinearSegmentedColormap
import scipy as sp
from data_preparation import create_custom_cmap, filename_to_field
import datashader as ds
from datashader.mpl_ext import dsshow
import pandas as pd
from scipy.ndimage import gaussian_filter


#import NN from data_preparation
#import DNS
# spatial constants
filter_sizes=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00]


# Data paths
data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
data_path_reaction = 'wtemp-slice-B1-0000080000.raw'

fwidth_n = np.array([25, 37, 49, 62, 74, 86, 99])

#Create function that returns sigma value based on filter size index
def sigma_value(filter_size):
  # Get the index of the filter_size
  index = filter_sizes.index(filter_size)
  actual_filter_size = fwidth_n[index]
  return np.sqrt(actual_filter_size ** 2 / 12.0)

def exclude_boundary(filter_size):
  #Get the index of the filter_size
  index = filter_sizes.index(filter_size)
  actual_filter_size = fwidth_n[index]
  # Exclusion boundaries
  base_exclusion_left = 0
  base_exclusion_right = 0
  additional_exclusion = 0.5 * actual_filter_size  # Adjust according to cell size if needed

  left_exclusion = base_exclusion_left + additional_exclusion
  right_exclusion = base_exclusion_right + additional_exclusion
  return int(left_exclusion), int(right_exclusion)

def get_boundaries(filter_size):
  exclude_boundaries_L, exclude_boundaries_R = exclude_boundary(filter_size)
  nx, ny = 384-(exclude_boundaries_L+exclude_boundaries_R), 384
  lx, ly = 0.01, 0.01 #[m]
  dx, dy = lx/(nx-1), ly/(ny-1)
  x = np.arange(dx*exclude_boundaries_L,lx+dx,dx)
  y = np.arange(0,ly+dy,dy)
  return x,y

def phi_field_res(filter_size):
  phi = filename_to_field(data_path_temp, data_path_reaction, exclude_boundary(filter_size))[2]
  phi_res = gaussian_filter(phi, sigma=sigma_value(filter_size))
  return phi_res

def phi_field_NN(filter_size):
  exclude_boundaries_L, exclude_boundaries_R = exclude_boundary(filter_size)
  phi_NN = np.load(f"Phi_NN_data/Phi_NN_{filter_size}.npy")
  phi_NN_bound=phi_NN[:][exclude_boundaries_L-1:len(phi_NN[0])-exclude_boundaries_R-1].T
  return phi_NN_bound

hot = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#4B006E'),
    (0.03, '#4169E1'),
    (0.15, '#adff5a'),
    (0.3, '#ffff5a'),
    (0.39, '#ff9932'),
    (0.6, '#D22B2B'),
    (1, '#D22B2B'),
], N=256)


def using_datashader(ax, filter_size):
    #print(phi_field_NN(filter_size))
    df = pd.DataFrame(dict(x=phi_field_res(filter_size).flatten(), y=phi_field_NN(filter_size).flatten()))
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        vmin=0,
        vmax=35,
        norm="linear",
        cmap=hot,
        ax=ax,
    )
fig, ax = plt.subplots()
using_datashader(ax, 1.0)
plt.plot([0,1], [0,1], linestyle='--', marker='', c='black', lw=0.8)
plt.xlim(0,1)
plt.ylim(0,1)
plt.ylabel("$\\overline{\\Phi}_{NN}$")
plt.xlabel("$\\overline{\\Phi}_{res}$")
plt.show()

# scatter_plot_run1(1.0)
white_jet = create_custom_cmap()
def plot_comparison_graphs():
  for i in range(len(filter_sizes)):
    if i==0:
      plt.subplot(2,len(filter_sizes),i+1).text(-171, 171, "$\\overline{\\Phi}_{res}$", fontsize=16, fontfamily='serif')
      plt.subplot(2,len(filter_sizes),i+1+len(filter_sizes)).text(-171, 171, "$\\overline{\\Phi}_{NN}$", fontsize=16, fontfamily='serif')
    plt.subplot(2,len(filter_sizes),i+1).axes.get_xaxis().set_visible(False)
    plt.subplot(2,len(filter_sizes),i+1).axes.get_yaxis().set_visible(False)
    plt.pcolor(phi_field_res(filter_sizes[i]), cmap='jet')
    plt.colorbar(location='top').set_ticks([0, np.floor(phi_field_res(filter_sizes[i]).max()*10)/10])
    plt.subplot(2,len(filter_sizes),i+1+len(filter_sizes)).axes.get_xaxis().set_visible(False)
    plt.subplot(2,len(filter_sizes),i+1+len(filter_sizes)).axes.get_yaxis().set_visible(False)
    plt.subplot(2,len(filter_sizes),i+1+len(filter_sizes)).set_title(str(filter_sizes[i]), y=-0.15)
    plt.pcolor(phi_field_NN(filter_sizes[i]), cmap='jet')
  plt.suptitle("$\\Delta /\\delta_{th}$", y=0.04)
  plt.show()
plot_comparison_graphs()

def calculate_pearson_r(filter_size):
  beep = [val for sublist in phi_field_res(filter_size) for val in sublist]
  boop = [val for sublist in phi_field_NN(filter_size) for val in sublist]
  pearson_r = sp.stats.pearsonr(beep, boop)[0]
  return pearson_r

def calculate_MSE(filter_size):
  beep = [val for sublist in  phi_field_res(filter_size) for val in sublist]
  boop = [val for sublist in phi_field_NN(filter_size) for val in sublist]
  MSE = mean_squared_error(beep, boop)
  return MSE

def comparison_plot(MSE_or_Pearson):
  y=[]
  if MSE_or_Pearson=="Pearson":
    for i in filter_sizes:
      y.append(calculate_pearson_r(i))
  elif MSE_or_Pearson=="MSE":
    for i in filter_sizes:
      y.append(calculate_MSE(i))
  else:
    print("comparison_plot only takes \'MSE\' or \'Pearson\'")
  plt.plot(filter_sizes,y, 'k', marker='o')
  plt.vlines(filter_sizes[0:-1], 0, 1.05*max(y), colors='gray', linestyles='dashed', alpha=0.3)
  plt.xlim(0.47, 2.03)
  plt.ylim(0.95*min(y), 1.05*max(y))
  plt.tick_params(axis='both', which='major', direction='in', top=True, right=True)
  plt.xlabel("$\\Delta/\\delta_{th,norm}$")
  if MSE_or_Pearson=="MSE":
    plt.ylabel("$\\epsilon_{MSE}$")
  else:
    plt.ylabel("$r_{p}$")
  plt.show()
comparison_plot('MSE')
'''
MSE_vals=map(calculate_MSE(filter_sizes), filter_sizes)
pearson_r_vals = map(calculate_pearson_r(filter_sizes), filter_sizes)

# Plot MSE
plt.plot(filter_sizes, MSE_vals)
plt.show()

# Plot Pearson R
plt.plot(filter_sizes, pearson_r_vals)
plt.show()

# reaction rate plot NN
plt.pcolor(x, y, np.moveaxis(wcr_field_NN, (0,1), (1,0)), cmap = 'hot')
plt.colorbar()
plt.show()

# error plot
sexy='hot'
wronk=np.subtract(wcr_field_res,wcr_field_NN)
plt.pcolor(x, y, np.moveaxis(wronk, (0,1), (1,0)), cmap =sexy)
plt.colorbar()
plt.show()

# data plot
plt.pcolor(x, y, np.moveaxis(wcr_field_res, (0,1), (1,0)), cmap='hot')
plt.colorbar()
plt.show()
'''