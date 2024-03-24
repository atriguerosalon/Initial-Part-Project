"""∇cT """
import scipy
import numpy as np
import tensorflow as tf
from scipy import ndimage
import matplotlib.pyplot as plt


nx, ny = 384, 384
lx, ly = 0.01, 0.01  # [m]
dx, dy = lx / (nx - 1), ly / (ny - 1)
x = np.arange(0, lx + dx, dx)
y = np.arange(0, ly + dy, dy)

# flame related constants
TU = 1500.000 # K
TB = 1623.47 # K
DTH = 0.0012904903 # m
DU = 0.2219636 # kg/m^3
SL = 1.6585735551 # m/s

#data_path1 
data_path = "C:\\Users\\danie\\Documents\\delft university of technology\\test analysis and simulation\\mission1\\mission1\\nablatemp-slice-B1-0000080000.raw"
#PLS MAKE THE DATA PATH WITH RESPECT TO THE WORKING DIRECTORY, SO THAT IT CAN BE EASILY RUN ON ANY MACHINE

ctdata = np.fromfile(data_path, count=-1, dtype=np.float64).reshape(nx, ny)
ctdata /= 123.47
ctdata = ctdata / (1 / DTH)
ctdata =  scipy.ndimage.gaussian_filter(ctdata, 10)

# Data plot1
plt.figure(figsize=(8, 6))  # Optional: Adjust figure size
plt.pcolor(x, y, np.moveaxis(ctdata, (0, 1), (1, 0)), shading='auto')  # Use shading='auto' for smooth color transitions.
plt.colorbar()  # Show color scale
plt.xlabel('X Position (mm)')  # Label the x-axis
plt.ylabel('Y Position (mm)')  # Label the y-axis
plt.title('Temperature Gradient Distribution')  # Add a title to the plot
plt.show()




# data_path2
d_p = "C:\\Users\\danie\\Documents\\delft university of technology\\test analysis and simulation\\mission1\\mission1\\wtemp-slice-B1-0000080000.raw"

# load the data
wctdata = np.fromfile(d_p, count=-1, dtype=np.float64).reshape(nx, ny)
wctdata /= 123.47
# filter size in cell number - 25 means length of 25 cells
fwidth_n = np.array([25, 37, 49, 62, 74, 86, 99])

# std. deviation - this part not sure :( - please figure out and teach me :P
#sig = np.sqrt( fwidth_n[0] ** 2 /12.0 )
sig = 0.5*DTH

wctdata = wctdata / (DU * SL) / DTH
wctdata = scipy.ndimage.gaussian_filter(wctdata, sigma=sig, mode='reflect', radius=100)

# data plot
plt.pcolor(x, y, np.moveaxis(wctdata, (0,1), (1,0)))
plt.colorbar()
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('Reaction rate')
plt.show()



# Function to load the neural network model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to prepare data for the model
def prepare_data(ctdata, wctdata):
    # Flatten the data if your model expects 1D input per sample
    ctdata_flat = ctdata.flatten()
    wctdata_flat = wctdata.flatten()
    
    # The model expects two inputs; for models expecting a single concatenated input, you'd concatenate these arrays
    # For example: input_data = np.concatenate((ctdata_flat, wctdata_flat))
    # Here, we prepare them as separate inputs
    return [np.array([wctdata_flat]), np.array([ctdata_flat])]  # Adding batch dimension

# Assuming `ctdata` and `wctdata` are defined in your previous code block

# Load the neural network model
# Update this path to where your model is stored
model_path = "C:\\Users\\danie\\Documents\\delft university of technology\\test analysis and simulation\\mission2\\mission2\\phi_nn_epoch_00240.h5"
model = load_model(model_path)

# Prepare the `ctdata` and `wctdata` for the model
input_data = prepare_data(ctdata, wctdata)

# Predict using the model
# If the model expects two separate inputs, we provide them as a list
predictions = model.predict(input_data)

# Process `predictions` as needed for your application
print(predictions)
