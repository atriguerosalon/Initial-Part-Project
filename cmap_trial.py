import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Define the original jet colormap
jet = plt.cm.get_cmap('jet', 256)

# Modify the colormap to set values below 0.2 to white
newcolors = jet(np.linspace(0, 1, 256))
newcolors[:int(256*0.2), :] = np.array([1, 1, 1, 1])  # RGBA for white color
new_jet = mcolors.LinearSegmentedColormap.from_list('white_jet', newcolors)

# Test the new colormap with a sample image
img = np.random.rand(100,100)
plt.imshow(img, cmap=new_jet)
plt.colorbar()
plt.show()
