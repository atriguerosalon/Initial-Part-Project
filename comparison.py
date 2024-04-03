import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LinearSegmentedColormap
import scipy as sp
from data_preparation import create_custom_cmap, filename_to_field, calculate_phi_0th_order, sigma_value
import datashader as ds
from datashader.mpl_ext import dsshow
import pandas as pd
from scipy.ndimage import gaussian_filter

# Add desired font settings
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


#import NN from data_preparation
#import DNS
# spatial constants
filter_sizes=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00]
grid_size_mm=10

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
  lx, ly = 0.01*nx/384, 0.01 #[m]
  dx, dy = lx/(nx), ly/(ny)
  x = np.arange(exclude_boundaries_L,385-exclude_boundaries_R,1)*10/384 #[mm]
  y = np.arange(0,ly+dy,dy)*1000 #[mm] - Ik this is messy but idk why it dont work if I just change lx=10
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

def phi_field_0th(filter_size):
  exclude_boundaries_L, exclude_boundaries_R = exclude_boundary(filter_size)
  wcr_field_star, ct_field_star, phi = filename_to_field(data_path_temp, data_path_reaction, (exclude_boundaries_L, exclude_boundaries_R))
  unfiltered_0th= calculate_phi_0th_order(wcr_field_star, ct_field_star, filter_size)
  unfiltered_0th = np.flipud(unfiltered_0th)
  return gaussian_filter(unfiltered_0th, sigma=sigma_value(filter_size))


hot = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-40, '#4B006E'),
    (0.006, '#4169E1'),
    (0.01, '#adff5a'),
    (0.03, '#ffff5a'),
    (0.4, '#ff9932'),
    (0.9, '#D22B2B'),
    (1, '#D22B2B'),
], N=256)


def using_datashader(ax, filter_size):
    #print(phi_field_NN(filter_size))
    x=phi_field_res(filter_size)[::-1].flatten()
    y=phi_field_NN(filter_size).flatten()

    max_x_val=np.max(x)
    globals()["divisions"]=5000
    x_vals=np.arange(0, max_x_val-1/divisions*max_x_val, 1/divisions*max_x_val)
    y_vals=np.array([])
    for i in range(divisions-1):
      globals()[f"interval_indices{i}"]=np.array([])
      globals()[f"interval_y_vals{i}"]=np.array([])
    for i in range(len(x)):
      interval_val=int((x[i]*divisions))
      globals()[f"interval_indices{interval_val}"]=np.append(globals()[f"interval_indices{interval_val}"], i)
    for i in range(divisions-1):
      for j in globals()[f"interval_indices{i}"]:
        if j!=None:
          globals()[f"interval_y_vals{i}"]=np.append(globals()[f"interval_y_vals{i}"], y[int(j)])
        else:
          globals()[f"interval_y_vals{i}"]=np.append(globals()[f"interval_y_vals{i}"], 0)
      if globals()[f"interval_y_vals{i}"].size==0:
        y_vals=np.append(y_vals,0)
      else:
        y_vals=np.append(y_vals,np.mean(globals()[f"interval_y_vals{i}"]))
    df = pd.DataFrame({'x':x_vals, 'y':y_vals})
    """
    df=pd.DataFrame({'x':x, 'y':y})
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        width_scale=0.15,
        height_scale=0.15,
        norm="linear",
        cmap=hot,
        ax=ax,
    )
    """
    plt.scatter(x_vals,y_vals, s=0.05)

fig, ax = plt.subplots()
using_datashader(ax, 1.0)
plt.plot([0,1], [0,1], linestyle='--', marker='', c='black', lw=0.8)
plt.xlim(0,1)
plt.ylim(0,1)
plt.ylabel("$\\overline{\\Phi}_{NN}$")
plt.xlabel("$\\overline{\\Phi}_{res}$")
plt.savefig(f"NN_vs._res_Scatterplot_Averaged_{divisions}.png")
plt.show()

#remake scatterplots

def plot_comparison_graphs():
  height_ratios=[0.1,2,2,2]
  width_ratios=[]
  for i in range(len(filter_sizes)):
     width_ratios.append(len(get_boundaries(filter_sizes[i])[0]))
  fig, axs=plt.subplots(4,len(filter_sizes), gridspec_kw={'height_ratios': height_ratios, 'width_ratios': width_ratios})
  row_titles=["$\\overline{\\Phi}_{res}$","$\\overline{\\Phi}_{NN}$","$\\overline{\\Phi}_{0th}$"]
  #setting y-axis labels and row titles (type of analysis conducted)
  for i in range(3):
    axs[i+1,0].text(-4.5, 4.5, row_titles[i], fontsize=18, fontfamily='serif')
    axs[i+1,0].axes.set_ylabel('y (mm)', labelpad=-4, fontsize=14)
  #hiding unnecessary axes
    for j in range(0,len(filter_sizes)):
      if i!=2:
        axs[i+1, j].get_xaxis().set_visible(False)
      if j!=0:
        axs[i+1,j].axes.get_yaxis().set_visible(False)
  
  for i in range(len(filter_sizes)):
    #setting up x-axis and column titles (filter size used)
    axs[3,i].text(s=str(filter_sizes[i]), x=4.25, y=-3.25, fontsize=14)
    x,y =get_boundaries(filter_sizes[i])
    axs[3,i].axes.set_xticks([2,4,6,8])
    axs[3,i].axes.set_xlabel('x (mm)',labelpad=0.5,fontsize=14)

    #plotting actual graphs

    axs[1,i].imshow(phi_field_res(filter_sizes[i]), cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()])
    axs[2,i].imshow(np.flipud(phi_field_NN(filter_sizes[i])), cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()])
    axs[3,i].imshow(np.flipud(phi_field_0th(filter_sizes[i])), cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()])

    #colorbar tings
    tick_pos=np.arange(0, np.floor(phi_field_res(filter_sizes[i]).max()*10)/10+0.05, 0.05)
    tick_labels=["" for i in tick_pos]
    tick_labels[0]=str(0)
    tick_labels[-1]=str(np.floor(phi_field_res(filter_sizes[i]).max()*10)/10) #idk why this dont work
    plt.colorbar(mappable=axs[1,i].imshow(phi_field_res(filter_sizes[i]), cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()]), cax=axs[0,i], orientation='horizontal')
    axs[0,i].set_xticks(tick_pos, labels=tick_labels, minor=False)
  
  #title, fig saving, and showing
  fig.suptitle("$\\Delta /\\delta_{th}$",x=0.52, y=0.03, fontsize=18)
  plt.savefig("Graph Comparison")
  plt.show()

#plot_comparison_graphs()

def calculate_pearson_r(filter_size, NN_or_0th):
  beep = [val for sublist in phi_field_res(filter_size)[::-1] for val in sublist]
  boop = [val for sublist in phi_field_NN(filter_size) for val in sublist]
  bob = [val for sublist in phi_field_0th(filter_size) for val in sublist]
  if NN_or_0th=='NN':
    pearson_r = sp.stats.pearsonr(beep, boop)[0]
  elif NN_or_0th=='0th':
    pearson_r = sp.stats.pearsonr(beep, bob)[0]
  else:
    print(f"calculate_pearson_r only takes \'NN\' or \'0th\' as inputs, not {NN_or_0th}.")
    return 
  return pearson_r

def calculate_MSE(filter_size, NN_or_0th):
  beep = np.array([val for sublist in  phi_field_res(filter_size)[::-1] for val in sublist])
  boop = np.array([val for sublist in phi_field_NN(filter_size) for val in sublist])
  bob = np.array([val for sublist in phi_field_0th(filter_size) for val in sublist])
  MSE = mean_squared_error(beep, boop)
  if NN_or_0th=='NN':
    MSE = mean_squared_error(beep, boop)
  elif NN_or_0th=='0th':
    MSE = mean_squared_error(beep, bob)
  else:
    print(f"calculate_MSE only takes \'NN\' or \'0th\' as inputs, not {NN_or_0th}.")
    return 
  return MSE

def comparison_plot(MSE_or_Pearson):
  y_NN=[]
  y_0th=[]
  if MSE_or_Pearson=="Pearson":
    for i in filter_sizes:
      y_NN.append(calculate_pearson_r(i, 'NN'))
      y_0th.append(calculate_pearson_r(i, '0th'))
  elif MSE_or_Pearson=="MSE":
    for i in filter_sizes:
      y_NN.append(calculate_MSE(i, 'NN'))
      y_0th.append(calculate_MSE(i, '0th'))
  else:
    print(f"comparison_plot only takes \'MSE\' or \'Pearson\', not {MSE_or_Pearson}")
  plt.plot(filter_sizes,y_NN, 'k', marker='o',label="NN vs. DNS")
  plt.plot(filter_sizes, y_0th, marker='o', label='0th vs. DNS')
  plt.vlines(filter_sizes, 0, 1.05*max(y_NN), colors='gray', linestyles='dashed', alpha=0.3)
  plt.xlim(0.47, 2.03)
  plt.ylim(0.95*min(min(y_NN), min(y_0th)), 1.05*max(y_NN))
  plt.xticks([i/4 for i in range(2,9)])
  plt.tick_params(axis='both', which='major', direction='in', top=True, right=True)
  plt.xlabel("$\\Delta/\\delta_{th,norm}$", fontsize=16)
  plt.legend()
  if MSE_or_Pearson=="MSE":
    plt.ylabel("$\\epsilon_{MSE}$", fontsize=16)
  else:
    plt.ylabel("$r_{p}$", fontsize=16)
  plt.show()
#comparison_plot('MSE')