import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter
from data_preparation import filename_to_field, calculate_phi_0th_order, sigma_value
import scipy as sp

#may or may not use:
import mpl_scatter_density
import datashader as ds
from datashader.mpl_ext import dsshow
import pandas as pd


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


def scatterplots(filter_size, ax1=None, ax2=None):
  x=phi_field_res(filter_size)[::-1].flatten()
  y=phi_field_NN(filter_size).flatten()
  empty_intervals=[]
  max_x_val=np.max(x)
  globals()["divisions"]=5000
  x_vals=np.linspace(0, 1, divisions)
  y_vals=np.array([])
  all_intervals=list(range(divisions-1))
  
  for i in range(divisions-1):
    #creating arrays for every interval
    globals()[f"interval_indices{i}"]=np.array([])
    globals()[f"interval_y_vals{i}"]=np.array([])
  
  for i in range(len(x)):
    #assigning every x index to some interval
    interval_val=int((x[i]*divisions))
    globals()[f"interval_indices{interval_val}"]=np.append(globals()[f"interval_indices{interval_val}"], i)
  for i in range(divisions-1):
    #finding empty intervals
    if globals()[f"interval_indices{i}"].size==0:
      empty_intervals.append(i)
  
  mask = np.array(empty_intervals)
  all_intervals = np.array(all_intervals)
  # Create a mask covering all indices in all_intervals
  full_mask = np.zeros(len(all_intervals), dtype=bool)
  full_mask[mask] = True

  # Use the full mask to exclude intervals
  valid_intervals = all_intervals[~full_mask]
  for i in valid_intervals:
    #mapping every y value in valid intervals
    for j in globals()[f"interval_indices{i}"]:
        globals()[f"interval_y_vals{i}"] = np.append(globals()[f"interval_y_vals{i}"], y[int(j)]) 
    #finding the y average in the interval
    y_vals=np.append(y_vals, np.mean(globals()[f"interval_y_vals{i}"]))
  
  #only keeping x_vals of valid intervals
  x_vals=np.array([x_vals[i] for i in valid_intervals])

  #datashader bs - not really much of a point
  """
    #df=pd.DataFrame({'x':x, 'y':y})
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        width_scale=1,
        height_scale=1,
        norm="linear",
        cmap='inferno',
        ax=ax,
    )
    """
  scatter_s=0.05
  lw=0.8
  label_size=14
  #plotting averaged
  if ax1 is None:
      fig1, ax1 = plt.subplots()
  else:
      fig1 = ax1.get_figure()  # Retrieve the figure associated with the provided axis
  ax1.scatter(x_vals, y_vals, s=scatter_s, color='k')
  ax1.plot([0, 1], [0, 1], linestyle='--', marker='', c='blue', lw=lw)
  ax1.set_xlim(0, 1)
  ax1.set_ylim(0, 1)
  ax1.set_ylabel("$\\overline{\\Phi}_{NN}$", fontsize=label_size)
  ax1.set_xlabel("$\\overline{\\Phi}_{res}$", fontsize=label_size)
  if ax1 is None:
      fig1.savefig(f"C:\\Users\\Equipo\\Initial-Part-Project\\Scatter_Varying_Divisions\\Scatterplot_Averaged_{filter_size}.png")
      plt.close(fig1)  # Close the figure if created within the function

  #trying Savitz-Golay filtering
  """
  if ax1 is None:
    fig1, ax1 = plt.subplots()
  else:
    fig1 = ax1.get_figure()  # Retrieve the figure associated with the provided axis
  y_filtered=sp.signal.savgol_filter(y, 5001, 7)
  ax1.scatter(x, y_filtered, s=scatter_s, color='k')
  ax1.plot([0, 1], [0, 1], linestyle='--', marker='', c='blue', lw=lw)
  ax1.set_xlim(0, 1)
  ax1.set_ylim(0, 1)
  ax1.set_ylabel("$\\overline{\\Phi}_{NN}$", fontsize=label_size)
  ax1.set_xlabel("$\\overline{\\Phi}_{res}$", fontsize=label_size)
  if ax1 is None:
      fig1.savefig(f"C:\\Users\\Equipo\\Initial-Part-Project\\Scatter_Varying_Divisions\\Scatterplot_Filtered_{filter_size}.png")
      plt.close(fig1)  # Close the figure if created within the function
  """
      
  #plotting non-averaged
  if ax2 is None:
      fig2, ax2 = plt.subplots()
  else:
      fig2 = ax2.get_figure()  # Retrieve the figure associated with the provided axis
  ax2.scatter(x, y, s=scatter_s, color='k')
  ax2.plot([0, 1], [0, 1], linestyle='--', marker='', c='blue', lw=lw)
  ax2.set_xlim(0, 1)
  ax2.set_ylim(0, 1)
  ax2.set_ylabel("$\\overline{\\Phi}_{NN}$", fontsize=label_size)
  ax2.set_xlabel("$\\overline{\\Phi}_{res}$", fontsize=label_size)
  if ax2 is None:
      fig1.savefig(f"C:\\Users\\Equipo\\Initial-Part-Project-3\\Scatter_Varying_Divisions\\Scatterplot_Reg_{filter_size}.png")
      plt.close(fig1)  # Close the figure if created within the function
  return fig1, ax1, fig2, ax2

def compare_filter_sizes():
  row_text_size=12
  filters_for_scatter=[0.5,1.0,1.5,2.0]
  fig, axs= plt.subplots(2, 4)
  axs=axs.ravel()
  for i, filter_size in enumerate(filters_for_scatter):
    scatterplots(filter_size, ax2=axs[i], ax1=axs[i+len(filters_for_scatter)])
  row_titles=["Unfiltered", "Filtered"]
  #setting y-axis labels and row titles (type of analysis conducted)
  for i in range(len(row_titles)):
    axs[i*len(filters_for_scatter)].text(-0.52, 0.5, row_titles[i], fontsize=row_text_size, fontfamily='serif')
  #hiding unnecessary axes
    for j in range(len(filters_for_scatter)):
      if i!=(len(row_titles)-1):
        axs[i*len(filters_for_scatter)+j].get_xaxis().set_visible(False)
      if j!=0:
        axs[i*len(filters_for_scatter)+j].axes.get_yaxis().set_visible(False)
  
  for i in range(len(filters_for_scatter)):
    #setting up x-axis and column titles (filter size used)
    axs[len(filters_for_scatter)+i].text(s=str(filters_for_scatter[i]), x=.45, y=-.28, fontsize=12)
  fig.suptitle("$\\Delta /\\delta_{th}$",x=0.52, y=0.038, fontsize=18)
  plt.tight_layout()  # Adjust layout for better spacing
  plt.savefig("Filtering w. Diff Filter Sizes.png")
  plt.show()

#compare_filter_sizes()

def plot_comparison_graphs():
  rc_textsize=18
  height_ratios=[0.1,2,2,2]
  width_ratios=[]
  for i in range(len(filter_sizes)):
     width_ratios.append(len(get_boundaries(filter_sizes[i])[0]))
  fig, axs=plt.subplots(4,len(filter_sizes), gridspec_kw={'height_ratios': height_ratios, 'width_ratios': width_ratios})
  row_titles=["$\\overline{\\Phi}_{res}$","$\\overline{\\Phi}_{NN}$","$\\overline{\\Phi}_{0th}$"]
  #setting y-axis labels and row titles (type of analysis conducted)
  for i in range(len(row_titles)):
    axs[i+1,0].text(-7, 4.5, row_titles[i], fontsize=22, fontfamily='serif')
    axs[i+1,0].axes.set_yticks([0,2,4,6,8,10], labels=[0,2,4,6,8,10], fontsize=14)
    axs[i+1,0].axes.set_ylabel('y (mm)', labelpad=-0.75, fontsize=rc_textsize)
  #hiding unnecessary axes
    for j in range(len(filter_sizes)):
      if i!=2:
        axs[i+1, j].get_xaxis().set_visible(False)
      if j!=0:
        axs[i+1,j].axes.get_yaxis().set_visible(False)
  
  for i in range(len(filter_sizes)):
    #setting up x-axis and column titles (filter size used)
    axs[len(row_titles),i].text(s=str(filter_sizes[i]), x=4.25, y=-4.9, fontsize=rc_textsize)
    x,y =get_boundaries(filter_sizes[i])
    axs[len(row_titles),i].axes.set_xticks([2,4,6,8], labels=[2,4,6,8], fontsize=14)
    axs[len(row_titles),i].axes.set_xlabel('x (mm)',labelpad=0.5,fontsize=rc_textsize)

    #plotting actual graphs

    axs[1,i].imshow(phi_field_res(filter_sizes[i]), cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()])
    axs[2,i].imshow(np.flipud(phi_field_NN(filter_sizes[i])), cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()])
    axs[3,i].imshow(np.flipud(phi_field_0th(filter_sizes[i])), cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()])

    #colorbar tings
    tick_pos=np.arange(0, np.floor(phi_field_res(filter_sizes[i]).max()*10)/10+0.05, 0.05)
    tick_labels=["" for i in tick_pos]
    tick_labels[0]=str(0)
    tick_labels[-1]=str(np.floor(phi_field_res(filter_sizes[i]).max()*10)/10) #idk why this dont work
    plt.colorbar(mappable=axs[1,i].imshow(phi_field_res(filter_sizes[i]), cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()]), cax=axs[0,i], orientation='horizontal', fraction=0.047*len(x)/384)
    axs[0,i].set_xticks(tick_pos, labels=tick_labels, minor=False, fontsize=16)
    axs[0,i].xaxis.set_ticks_position('top')
  
  #title, fig saving, and showing
  fig.suptitle("$\\Delta /\\delta_{th}$",x=0.535, y=0.045, fontsize=22)
  plt.savefig("Graph Comparison")
  plt.show()

#plot_comparison_graphs()
def plot_demo_graph():
  tick_size = 16
  rc_textsize = 18
  filter_index = 0
  height_ratios = [.2, 2]
  width_ratios = [1]

  # Create the figure and axes
  fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': height_ratios, 'width_ratios': width_ratios})
  x,y =get_boundaries(filter_sizes[filter_index])
  # Plot the image
  image = axs[1].imshow(phi_field_res(filter_sizes[filter_index]), cmap='jet', extent=[x.min(), x.max(), y.min(), y.max()])

  # Customize the image axes
  axs[1].set_yticks([0, 2, 4, 6, 8, 10], labels=[0, 2, 4, 6, 8, 10], fontsize=tick_size)
  axs[1].set_ylabel('y (mm)', labelpad=-0.75, fontsize=rc_textsize)
  axs[1].set_xticks([2, 4, 6, 8], labels=[2, 4, 6, 8], fontsize=tick_size)
  axs[1].set_xlabel('x (mm)', labelpad=0.5, fontsize=rc_textsize)

  # Create the colorbar
  cbar = plt.colorbar(mappable=image, cax=axs[0], orientation='horizontal', fraction=0.04)  # Adjust fraction here

  # Customize the colorbar ticks and labels
  tick_pos = np.arange(0, np.floor(phi_field_res(filter_sizes[filter_index]).max() * 10) / 10 + 0.05, 0.05)
  tick_labels = ["" for _ in tick_pos]
  tick_labels[0] = str(0)
  tick_labels[-1] = str(np.floor(phi_field_res(filter_sizes[filter_index]).max() * 10) / 10)

  axs[0].set_xticks(tick_pos)
  axs[0].set_xticklabels(tick_labels)
  axs[0].tick_params(axis='x', which='major', labelsize=tick_size)
  axs[0].xaxis.set_ticks_position('top')

  # Adjust the aspect ratio of the colorbar to match the image
  cbar.ax.set_aspect(0.106)  # For 2.00, this is 0.063. For 0.5, this is 0.106

  plt.show()
plot_demo_graph()

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
  beep = np.array([boo for bee in  phi_field_res(filter_size)[::-1] for boo in bee])
  boop = np.array([boo for bee in phi_field_NN(filter_size) for boo in bee])
  bob = np.array([boo for bee in phi_field_0th(filter_size) for boo in bee])
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
  if MSE_or_Pearson=='MSE':
    plt.ylim(0, 1.05*max(y_NN))
  elif MSE_or_Pearson == 'Pearson':
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
comparison_plot('Pearson')