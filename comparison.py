import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
import scipy as sp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from data_preparation import create_custom_cmap
from gaussian_filter_apply import apply_gaussian
import datashader as ds
from datashader.mpl_ext import dsshow
import pandas as pd
#import NN from data_preparation
#import DNS
# spatial constants
exclude_boundaries = 0
nx, ny = 384-2*exclude_boundaries, 384-2*exclude_boundaries
lx, ly = 0.01, 0.01 #[m]
dx, dy = lx/(nx-1), ly/(ny-1)
x = np.arange(0,lx+dx,dx)
y = np.arange(0,ly+dy,dy)
filter_sizes=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00]

# data load
def phi_field_res(filter_size):
  phi_res = apply_gaussian(filter_size*10, exclude_boundaries)[2].T
  return phi_res

def phi_field_NN(filter_size):
  phi_NN = np.load(f"Phi_NN_data/Phi_NN_{filter_size}.npy")
  return phi_NN

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
'''
plt.subplot(1,2,1)
plt.pcolor(x,y, phi_field_NN(1.0))
plt.colorbar(location='above')
plt.subplot(1,2,2)
plt.pcolor(x,y, phi_field_res(1.0))
plt.colorbar()
plt.show()
'''

#This is really bad timewise lmaoooo:
def scatter_plot_run1(filter_size):
  nbins=500
  fig = plt.figure(figsize=(12, 10)) #potentially wronk
  #plt.scatter(phi_field_res(filter_size), phi_field_NN(filter_size), s=0.0001)
  #beep = [val for sublist in phi_field_res(filter_size) for val in sublist]
  #boop = [val for sublist in phi_field_NN(filter_size) for val in sublist]
  beep = phi_field_res(filter_size).flatten()
  boop = phi_field_NN(filter_size).flatten()
  k=sp.stats.gaussian_kde([beep, boop])
  print("k generated")
  #xi, yi = np.mgrid[min(beep):max(beep):nbins*1j, min(boop):max(boop):nbins*1j]
  xi, yi = np.mgrid[beep.min():beep.max():nbins*1j, boop.min():boop.max():nbins*1j]
  print("chill, be patient (esp you Owen), its 300k points with O(n) time complexity - usually takes about 10 mins")
  zi = k(np.vstack([xi.flatten(), yi.flatten()]))
  # Make the plot
  plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=hot)
  
  """
  density = ax.scatter_density(phi_field_res(filter_size), phi_field_NN(filter_size), cmap=hot)
  plt.plot([0,1], [0,1], linestyle='--', marker='', c='black', lw=0.8)
  plt.ylabel("$\\bar{\\Phi}_{c,NN}^{+}$")
  plt.xlabel("$\\bar{\\Phi}_{c,res}^{+}$")
  cbaxes = inset_axes(ax, width="3%", height="30%", loc=4)
  fig.colorbar(density, cax=cbaxes, ticks=[], orientation='vertical')
  """
  plt.show()
  
def using_datashader(ax, filter_size):

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

    plt.colorbar(dsartist)

fig, ax = plt.subplots()
using_datashader(ax, 1.0)
plt.show()

#scatter_plot_run1(1.0)
white_jet = create_custom_cmap()
def plot_comparison_graphs():
  for i in range(len(filter_sizes)):
    if i==0:
      plt.subplot(2,len(filter_sizes),i+1).text(-171, 171, "$\\Phi_{res}$", fontsize=16""", fontfamily='serif'""")
      plt.subplot(2,len(filter_sizes),i+1+len(filter_sizes)).text(-171, 171, "$\\Phi_{NN}$", fontsize=16""", fontfamily='serif'""")
    plt.subplot(2,len(filter_sizes),i+1).axes.get_xaxis().set_visible(False)
    plt.subplot(2,len(filter_sizes),i+1).axes.get_yaxis().set_visible(False)
    plt.pcolor(phi_field_res(filter_sizes[i]), cmap='jet')
    plt.colorbar(location='top')
    plt.subplot(2,len(filter_sizes),i+1+len(filter_sizes)).axes.get_xaxis().set_visible(False)
    plt.subplot(2,len(filter_sizes),i+1+len(filter_sizes)).axes.get_yaxis().set_visible(False)
    plt.subplot(2,len(filter_sizes),i+1+len(filter_sizes)).set_title(str(filter_sizes[i]), y=-0.15)
    plt.pcolor(phi_field_NN(filter_sizes[i]), cmap='jet')
  plt.suptitle("$\\Delta /\\delta t_{th}$", y=0.04)
  plt.show()
plot_comparison_graphs()
'''

filter_sizes=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00]

def calculate_MSE(filter_size):
  beep = [val for sublist in apply_gaussian(filter_size, exclude_boundaries)[2] for val in sublist]
  boop = [val for sublist in phi_NN for val in sublist]
  MSE = mean_squared_error(beep, boop)
  return MSE

def calculate_pearson_r(filter_size):
  beep = [val for sublist in apply_gaussian(filter_size, exclude_boundaries)[2] for val in sublist]
  boop = [val for sublist in phi_NN for val in sublist]
  pearson_r = sp.stats.pearsonr(beep, boop)
  return pearson_r

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