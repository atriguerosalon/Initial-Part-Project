# library
import numpy as np
import matplotlib.pyplot as plt
import scipy
from keras.models import load_model

# spatial constants
nx, ny = 384, 384
lx, ly = 0.01, 0.01 #[m]
dx, dy = lx/(nx-1), ly/(ny-1)
x = np.arange(0,lx+dx,dx)
y = np.arange(0,ly+dy,dy)

# flame related constants
TU = 1500.000 # K
TB = 1623.47 # K
DTH = 0.0012904903 # m
DU = 0.2219636 # kg/m^3
SL = 1.6585735551 # m/s

# normalization
CT_NORM = TB - TU
WCT_NORM = DU * SL / DTH
NCT_NORM = 1.0 / DTH

# data load
data_1_path = 'bar-wtemp-slice-B1-0000080000-049.raw'
data_1 = np.fromfile( data_1_path, count=-1, dtype=np.float64).reshape(-1,1)

data_2_path = 'tilde-nablatemp-slice-B1-0000080000-049.raw'
data_2 = np.fromfile( data_2_path, count=-1, dtype=np.float64).reshape(-1,1)

#Filter size variable
data_3 = np.zeros(np.shape(data_1))
data_3[:,0] = 0.5 * DTH

#non-dimensionalization
data_1 = data_1 / CT_NORM / WCT_NORM
data_2 = data_2 / CT_NORM / NCT_NORM
data_3 = data_3 / (2.0 * DTH)

# make X
X = np.concatenate((data_1.reshape(-1,1), data_2.reshape(-1,1), data_3.reshape(-1,1)), axis=1)

# model load
model = load_model('phi_nn_epoch_00240.h5')

# prediction
phi_nn = np.zeros((np.shape(X)[0]))
phi_nn = model.predict(X, batch_size=16, verbose=2).reshape(nx, ny)

# data plot
plt.pcolor(x, y, np.moveaxis(phi_nn, (0,1), (1,0)))
plt.colorbar()