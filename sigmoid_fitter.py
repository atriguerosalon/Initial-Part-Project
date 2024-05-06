import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Example sigmoid function
def sigmoid(x, beta, x0):
    return 1 / (1 + np.exp(-beta * (x - x0)))

# Example points from NN (replace these with actual data)
#x_data = np.linspace(0, 1.41, 100)  # Example data
#y_data = sigmoid(x_data, 1, 0)  # Example data
# Add some noise to simulate realistic data
#noise = np.random.normal(0, 0.05, size=x_data.shape)
#y_data = y_data + noise

# Load the actual data here, 1st column is x_data, 2nd column is y_data
x_data, y_data = np.load('sigmoid_data.npy').T
print(y_data)
# Fit the sigmoid to the data
popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[1, 0])  # Initial guesses for beta and x0
beta_opt, x0_opt = popt

# Generate the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 200)
y_fit = sigmoid(x_fit, beta_opt, x0_opt)

# Save the fitted curve values
np.save('sigmoid_fit_values.npy', [x_fit, y_fit])

# Plotting
plt.scatter(x_data, y_data, label='Neural Network Data', color='blue')
plt.plot(x_fit, y_fit, label=f'Fitted Sigmoid\nbeta={beta_opt:.2f}, x0={x0_opt:.2f}', color='red')
plt.legend()
plt.xlabel('Input Feature (Aligned Axis)')
plt.ylabel('Sigmoid Output')
plt.show()
