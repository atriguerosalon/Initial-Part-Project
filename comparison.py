import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter
from data_preparation import filename_to_field, calculate_phi_0th_order, sigma_value, exclude_boundaries, filename_to_0th_order_fields
import scipy as sp
import torch


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

data_path_temp_filtered = "Ref_0th_Fields\\tilde-nablatemp-slice-B1-0000080000-049.raw"
data_path_reaction_filtered ="Data_new_NN\\dataset_slice_B1_TS80\\bar-wtemp-slice-B1-0000080000-049.raw"

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
  base_exclusion_left = 25
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

def phi_field_NN_old(filter_size):
  exclude_boundaries_L, exclude_boundaries_R = exclude_boundary(filter_size)
  phi_NN = np.flipud(np.load(f"Phi_NN_data/Phi_NN_{filter_size}.npy")).T
  phi_NN_bound=exclude_boundaries(phi_NN,exclude_boundaries_L, exclude_boundaries_R)
  return phi_NN_bound

def phi_field_NN_new(filter_size):
   return np.flipud(np.load(f"NewNNFields\\Field_Filter_{filter_size}.npy"))

def phi_field_0th(filter_size):
  _, _, phi_field_0th = filename_to_0th_order_fields(data_path_temp_filtered, data_path_reaction_filtered, filter_size)
  unfiltered_0th = np.flipud(phi_field_0th)
  return unfiltered_0th

plt.rcParams['axes.linewidth'] = 1.5

def scatterplots(filter_size, ax2=None):
    x = phi_field_res(filter_size)[::-1].flatten()
    y = phi_field_NN_old(filter_size).flatten()
    
    empty_intervals = []
    max_x_val = np.max(x)
    divisions = 20000
    x_vals = np.linspace(0, 1, divisions)
    y_vals = np.array([])
    all_intervals = list(range(divisions - 1))

    for i in range(divisions - 1):
        globals()[f"interval_indices{i}"] = np.array([])
        globals()[f"interval_y_vals{i}"] = np.array([])

    for i in range(len(x)):
        interval_val = int((x[i] * divisions))
        globals()[f"interval_indices{interval_val}"] = np.append(globals()[f"interval_indices{interval_val}"], i)
    for i in range(divisions - 1):
        if globals()[f"interval_indices{i}"].size == 0:
            empty_intervals.append(i)

    mask = np.array(empty_intervals)
    all_intervals = np.array(all_intervals)
    full_mask = np.zeros(len(all_intervals), dtype=bool)
    full_mask[mask] = True

    valid_intervals = all_intervals[~full_mask]
    for i in valid_intervals:
        for j in globals()[f"interval_indices{i}"]:
            globals()[f"interval_y_vals{i}"] = np.append(globals()[f"interval_y_vals{i}"], y[int(j)])
        y_vals = np.append(y_vals, np.mean(globals()[f"interval_y_vals{i}"]))

    x_vals = np.array([x_vals[i] for i in valid_intervals])

    scatter_s = 0.05
    lw = 1.5  # Increased linewidth
    label_size = 25

    if ax2 is None:
        fig2, ax2 = plt.subplots()
    else:
        fig2 = ax2.get_figure()
    print("Made it here")
    # Rasterized=True to reduce file size, 
    #Increase rasterized resolution to 300 dpi

    ax2.scatter(x, y, s=scatter_s, color='k', alpha=0.15, rasterized=True)
    print("Made it here, post scatter")
    ax2.plot([0, 1.5], [0, 1.5], linestyle='--', marker='', c='midnightblue', lw=lw)
    ax2.set_xlim(0, max(x)*1.06)
    ax2.set_ylim(0, max(y)*1.06)
    ax2.set_ylabel("$\\overline{\\Phi}_{NN}$", fontsize=label_size)
    ax2.set_xlabel("$\\overline{\\Phi}_{res}$", fontsize=label_size)
    ax2.tick_params(axis='both', labelsize=18)  # Increase tick label size
    if ax2 is None:
        fig2.savefig(f"C:\\Users\\Equipo\\Initial-Part-Project-3\\Scatter_Varying_Divisions\\Scatterplot_Reg_{filter_size}.png")
        plt.close(fig2)
    return fig2, ax2

def compare_filter_sizes():
    row_text_size = 22
    filters_for_scatter = [0.5, 1.0, 1.5, 2.0]
    fig, axs = plt.subplots(1, 4, figsize=(19, 6.3), linewidth=2) 
    axs = axs.ravel()

    for i, filter_size in enumerate(filters_for_scatter):
        fig2, ax2 = scatterplots(filter_size, ax2=axs[i])  # Capture fig and ax2 returned by scatterplots function
        ax2.yaxis.set_major_locator(plt.MaxNLocator(5))  # Ensure y-axis ticks are present for all subplots
        if i != 0:
            ax2.set_ylabel('')  # Remove y-label for all subplots except the first one

    #for j in range(1,len(filters_for_scatter)):
            #axs[j].axes.get_yaxis().set_visible(False)

    for i in range(len(filters_for_scatter)):
        xlim = axs[i].get_xlim()
        ylim = axs[i].get_ylim()
        x_center = (xlim[0] + xlim[1]) / 2  # Calculate the center of the x-axis
        y_bottom = ylim[0] - (ylim[1] - ylim[0]) * 0.3  # Position slightly below the bottom of the y-axis
        axs[i].text(s=str(filters_for_scatter[i]), x=x_center, y=y_bottom, ha='center', fontsize=row_text_size)

    fig.suptitle("$\\Delta /\\delta_{th}$", x=0.52, y=0.075, fontsize=25)

    plt.tight_layout()

    #Set maximum size of the figure in pdf to 10MB
    plt.savefig("Filtering w. Diff Filter Sizes rast.pdf", dpi=500)
    plt.close()

def plot_comparison_graphs():
  rc_textsize=18
  height_ratios=[0.1,2,2,2]
  width_ratios=[]
  for i in range(len(filter_sizes)):
     width_ratios.append(len(get_boundaries(filter_sizes[i])[0]))
  fig, axs=plt.subplots(4,len(filter_sizes), gridspec_kw={'height_ratios': height_ratios, 'width_ratios': width_ratios})
  row_titles=["$\\overline{\\Phi}_{res}$","$\\overline{\\Phi}_{NN, new}$","$\\overline{\\Phi}_{NN,lit}$"]
  #setting y-axis labels and row titles (type of analysis conducted)
  for i in range(len(row_titles)):
    axs[i+1,0].text(-10, 4.5, row_titles[i], fontsize=22, fontfamily='serif')
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
    vmax_val=np.array([phi_field_res(filter_sizes[i]), phi_field_NN_new(filter_sizes[i]), phi_field_NN_old(filter_sizes[i])]).max()
    print(phi_field_res(filter_sizes[i]).max(), phi_field_NN_new(filter_sizes[i]).max(), phi_field_NN_old(filter_sizes[i]).max())
    axs[1,i].imshow(phi_field_res(filter_sizes[i]), cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()], vmax=vmax_val)
    axs[2,i].imshow(np.flipud(phi_field_NN_old(filter_sizes[i])), cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()], vmax=vmax_val)
    axs[3,i].imshow(np.flipud(phi_field_0th(filter_sizes[i])), cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()], vmax=vmax_val)

    #colorbar tings
    tick_pos=np.arange(0, np.floor(phi_field_res(filter_sizes[i]).max()*10)/10+0.05, 0.05)
    tick_labels=["" for i in tick_pos]
    tick_labels[0]=str(0)
    tick_labels[-1]=str(np.floor(phi_field_res(filter_sizes[i]).max()*10)/10) #idk why this dont work
    plt.colorbar(mappable=axs[1,i].imshow(phi_field_res(filter_sizes[i]), cmap='jet', extent =[x.min(), x.max(), y.min(), y.max()]), cax=axs[0,i], orientation='horizontal', fraction=0.046*len(x)/384, pad=0.4)
    axs[0,i].set_xticks(tick_pos, labels=tick_labels, minor=False, fontsize=16)
    axs[0,i].xaxis.set_ticks_position('top')

  #title, fig saving, and showing
  fig.suptitle("$\\Delta /\\delta_{th}$",x=0.535, y=0.045, fontsize=22)
  plt.savefig("Graph Comparison.pdf")
  plt.show()


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
#plot_demo_graph()

def calculate_pearson_r(filter_size, data_to_compare):
  beep = np.array([boo for bee in  phi_field_res(filter_size)[::-1] for boo in bee])
  boop = np.array([boo for bee in phi_field_NN_old(filter_size) for boo in bee])
  bob = np.array([boo for bee in phi_field_0th(filter_size) for boo in bee])
  bub = np.array([boo for bee in phi_field_NN_new(filter_size) for boo in bee])
  if data_to_compare=='NN':
    pearson_r = sp.stats.pearsonr(beep, boop)[0]
  elif data_to_compare=='0th':
    pearson_r = sp.stats.pearsonr(beep, bob)[0]
  elif data_to_compare=='newNN':
    pearson_r = sp.stats.pearsonr(beep, bub)[0]
  else:
    print(f"calculate_pearson_r only takes \'NN\' or \'0th\' as inputs, not {data_to_compare}.")
    return 
  return pearson_r

def calculate_MSE(filter_size, data_to_compare):
  beep = np.array([boo for bee in  phi_field_res(filter_size)[::-1] for boo in bee])
  
  if data_to_compare=='NN':
    boop = np.array([boo for bee in phi_field_NN_old(filter_size) for boo in bee])
    MSE = mean_squared_error(beep, boop)
  elif data_to_compare=='0th':
    bob = np.array([boo for bee in phi_field_0th(filter_size) for boo in bee])
    MSE = mean_squared_error(beep, bob)
  elif data_to_compare=='newNN':
    bub = np.array([boo for bee in phi_field_NN_new(filter_size) for boo in bee])
    MSE = mean_squared_error(beep, bub)
  else:
    print(f"calculate_MSE only takes \'NN\' or \'0th\' as inputs, not {data_to_compare}.")
    return 
  return MSE

plt.rcParams['axes.linewidth'] = 1.5

def comparison_plot(MSE_or_Pearson):
  y_NN=[]
  y_0th=[]
  y_newNN=[]
  if MSE_or_Pearson=="Pearson":
    for i in filter_sizes:
      y_NN.append(calculate_pearson_r(i, 'NN'))
      y_0th.append(calculate_pearson_r(i, '0th'))
      y_newNN.append(calculate_pearson_r(i, 'newNN'))
  elif MSE_or_Pearson=="MSE":
    for i in filter_sizes:
      y_NN.append(calculate_MSE(i, 'NN'))
      y_0th.append(calculate_MSE(i, '0th'))
      y_newNN.append(calculate_MSE(i, 'newNN'))
  else:
    print(f"comparison_plot only takes \'MSE\' or \'Pearson\', not {MSE_or_Pearson}")
  plt.plot(filter_sizes,y_NN, 'k', marker='o', markersize=8,label="$NN_{lit}$/res", linewidth=1.5, rasterized=True)
  plt.plot(filter_sizes, y_newNN, 'r', marker='o', markersize=8, label='$NN_{new}$/res', linewidth=1.5, rasterized=True)
  plt.plot(filter_sizes, y_0th, marker='o', markersize=8, label='0th/res', linewidth=1.5, rasterized=True)
  plt.vlines(filter_sizes, 0, 1.05*max(max(y_NN), max(y_0th), max(y_newNN)), colors='gray', linestyles='dashed', alpha=0.3, linewidth=1.5)
  plt.xlim(0.35, 2.15)
  if MSE_or_Pearson=='MSE':
    plt.ylim(0, 1.2*max(max(y_NN), max(y_0th), max(y_newNN)))
  elif MSE_or_Pearson == 'Pearson':
    plt.ylim(0, 1.05*max(max(y_NN), max(y_0th), max(y_newNN)))
  plt.xticks([i/4 for i in range(2,9)])
  plt.tick_params(axis='both', which='major', direction='in', right=True, labelsize=15)
  plt.xlabel("$\\Delta/\\delta_{th,norm}$", fontsize=18)
  plt.legend(loc='best', fontsize=18, edgecolor='black', fancybox=False).get_frame().set_linewidth(1.5)
  if MSE_or_Pearson=="MSE":
    plt.ylabel("$\\epsilon_{MSE}$", fontsize=18)
  else:
    plt.ylabel("$r_{p}$", fontsize=18)
  plt.savefig(f"{MSE_or_Pearson}_Plot.pdf")
  plt.show()
  plt.close()

def plot_theoretical_change_new_NN(filter_size1, filter_size2):
  if filter_size1 not in filter_sizes:
     print(f"{filter_size1} not a valid filter size")
     return
  if filter_size2 not in filter_sizes:
     print(f"{filter_size2} not a valid filter size")
     return
  
  phi_theo1=np.load("NewNNPredTheoretical\Array0.5.npy")
  phi_theo2=np.load("NewNNPredTheoretical\Array2.0.npy")
  diff=phi_theo2-phi_theo1
  plt.pcolor(diff)
  plt.colorbar()
  plt.suptitle("$\phi_{NN}$("+str(filter_size2)+")-$\phi_{NN}$("+str(filter_size1)+")")
  plt.show()

#run the functions here:

#compare_filter_sizes() #plot here

#remember that before applying this, you need to run data_prep
plot_comparison_graphs() #plot here

#comparison_plot('MSE') #plot here
#comparison_plot("Pearson")
#plot_theoretical_change_new_NN(0.5, 2)
