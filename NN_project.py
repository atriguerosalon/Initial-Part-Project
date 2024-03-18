import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from keras.models import load_model

# spatial constants
nx, ny = 384, 384
lx, ly = 0.01, 0.01  # [m]
dx, dy = lx / (nx - 1), ly / (ny - 1)
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)

# flame related constants
TU = 1500.000  # K
TB = 1623.47  # K
DTH = 0.0012904903  # m
DU = 0.2219636  # kg/m^3
SL = 1.6585735551  # m/s

# normalization
CT_NORM = TB - TU
WCT_NORM = DU * SL / DTH
NCT_NORM = 1.0 / DTH

# filter size in cell number - 25 means length of 25 cells
fwidth_n = np.array([25, 37, 49, 62, 74, 86, 99])

# std. deviation
sig = np.sqrt(fwidth_n[0] ** 2 / 12.0)

# Data loading and processing placeholder - Replace with your actual data loading
# Example for data_1
data_1_path = "wtemp-slice-B1-0000080000.raw" # Update this path
data_1 = np.fromfile(data_1_path, count=-1, dtype=np.float64)
data_1 = data_1.reshape(ny, nx)
data_1_filtered = scipy.ndimage.gaussian_filter(data_1, sigma=sig)

# Example for data_2
data_2_path = "nablatemp-slice-B1-0000080000.raw"  # Update this path
data_2 = np.fromfile(data_2_path, count=-1, dtype=np.float64)
data_2 = data_2.reshape(ny, nx)
data_2_filtered = scipy.ndimage.gaussian_filter(data_2, sigma=sig)

# Filter size variable
data_3 = np.full((ny, nx), 0.5 * DTH)

# Non-dimensionalization
data_1_filtered = data_1_filtered / CT_NORM / WCT_NORM
data_2_filtered = data_2_filtered / CT_NORM / NCT_NORM
data_3 = data_3 / (2.0 * DTH)

# Make X
X = np.stack((data_1_filtered.flatten(), data_2_filtered.flatten(), data_3.flatten()), axis=1)

# Model load
model = load_model('phi_nn_epoch_00240.h5')

# Prediction
phi_nn = model.predict(X, batch_size=16, verbose=2)
phi_nn = phi_nn.reshape(ny, nx)

# Data plot for phi_nn
plt.figure()
plt.pcolor(x, y, phi_nn, shading='auto')
plt.colorbar()
plt.title('Predicted phi_nn')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()