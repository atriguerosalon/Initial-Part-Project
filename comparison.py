# library
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
from sklearn.metrics import mean_squared_error
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
import scipy as sp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

drive.mount('/content/drive')
plt.rcParams.update(plt.rcParamsDefault)

# spatial constants
nx, ny = 384, 384
lx, ly = 0.01, 0.01 #[m]
dx, dy = lx/(nx-1), ly/(ny-1)
x = np.arange(0,lx+dx,dx)
y = np.arange(0,ly+dy,dy)

# flame related constants
TU = 1500.000 #K
TB = 1623.47 #K

# data load
data_path = '/content/drive/My Drive/RawData/wtemp-slice-B1-0000080000.raw'
data = np.fromfile( data_path, count=-1, dtype=np.float64).reshape(nx,ny)

# data plot
plt.pcolor(x, y, np.moveaxis(data, (0,1), (1,0)), cmap = 'hot')
plt.colorbar()

# Data Analysis
NN=np.empty(1)
DNS=np.empty(1)
print(NN)
print(DNS)
if len(NN)!=len(DNS):
  print("Data is not of the same size. NN has a size "+str(len(NN))+"while DNS has a size"+str(len(DNS)))


# data load
data_path1 = '/content/drive/My Drive/RawData/nablatemp-slice-B1-0000080000.raw'
data1 = np.fromfile( data_path1, count=-1, dtype=np.float64).reshape(nx,ny)

# data plot
plt.pcolor(x, y, np.moveaxis(data1, (0,1), (1,0)), cmap = 'hot')
plt.colorbar()
plt.show()

sexy='hot'
error=np.subtract(data,data1)
plt.pcolor(x, y, np.moveaxis(error, (0,1), (1,0)), cmap =sexy)
plt.colorbar()
plt.show()


bing=abs(np.subtract(data,data1))
plt.pcolor(x, y, np.moveaxis(abserror, (0,1), (1,0)), cmap ='hot')
plt.colorbar()
plt.show()


hot = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#4B006E'),
    (0.03, '#4169E1'),
    (0.08, '#adff5a'),
    (0.2, '#ffff5a'),
    (0.3, '#ff9932'),
    (0.5, '#D22B2B'),
    (1, '#D22B2B'),
], N=256)



fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(data, data1, cmap=hot)
plt.plot([0,250000], [0,250000], linestyle='--', marker='', c='black', lw=0.8)
plt.ylabel("$\\bar{\\omega}_{c,NN}^{+}$")
plt.xlabel("$\\bar{\\omega}_{c,res}^{+}$")
plt.ylim((0,250000))
plt.xlim((0,250000))
cbaxes = inset_axes(ax, width="3%", height="30%", loc=4)
fig.colorbar(density, cax=cbaxes, ticks=[], orientation='vertical')
plt.show()


filter_sizes=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00]

beep = [val for sublist in data for val in sublist]
boop = [val for sublist in data1 for val in sublist]


MSE = mean_squared_error(beep, boop)
print(MSE)

pearson_r = sp.stats.pearsonr(beep, boop)
print(pearson_r.statistic**2)