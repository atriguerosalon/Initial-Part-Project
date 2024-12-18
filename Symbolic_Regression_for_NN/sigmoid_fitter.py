import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Filter size
filter_size = ''
# Example sigmoid function
def sigmoid(x, beta, x0):
    return 1 / (1 + np.exp(-beta * (x - x0)))

# Example points from NN (replace these with actual data)
#x_data = np.linspace(0, 1.41, 100)  # Example data
#y_data = sigmoid(x_data, 1, 0)  # Example data
# Add some noise to simulate realistic data
#noise = np.random.normal(0, 0.05, size=x_data.shape)
#y_data = y_data + noise
# Load dummy data
#x_data, y_data = np.load('./Symbolic_Regression_for_NN/sigmoid_data.npy').T
# Load the actual data here, 1st column is x_data, 2nd column is y_data
# Print the shape of np.load('./Symbolic_Regression_for_NN/values_along_characteristic_line_phi_NN_final.npy')[1]
#print(np.load(f'./Symbolic_Regression_for_NN/values_along_characteristic_line_phi_NN_final_{filter_size}.npy').shape)
# Print the shape of the np
#print(np.load(f'./Symbolic_Regression_for_NN/values_along_characteristic_line_phi_NN_final_{filter_size}.npy').shape)
x_data, y_data = np.load(f'./Symbolic_Regression_for_NN/values_along_characteristic_line_phi_NN_final_{filter_size}.npy')
# Print the shape of np.load('./Symbolic_Regression_for_NN/values_along_characteristic_line_phi_NN_final.npy')[1]
#print(np.load(f'./Symbolic_Regression_for_NN/values_along_characteristic_line_phi_NN_{filter_size}.npy').shape)
real_x_data, real_y_data = np.load(f'./Symbolic_Regression_for_NN/values_along_characteristic_line_phi_NN_final_{filter_size}.npy')[0]

# Plot x_data and y_data, next to real_x_data and real_y_data
plt.scatter(x_data, y_data, label='Neural Network Data', color='blue', s=5)
plt.scatter(real_x_data, real_y_data, label='Real Data', color='green', s=5)
plt.legend()
plt.xlabel('Input Feature (Aligned Axis)')
plt.ylabel('Sigmoid Output')
plt.show()

# Fit the sigmoid to the data
popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[1, 0])  # Initial guesses for beta and x0
beta_opt, x0_opt = popt

# Generate the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = sigmoid(x_fit, beta_opt, x0_opt)

# Add rsq using sklearn
r_squared = r2_score(y_data, sigmoid(x_data, *popt))

#Save the parameters of the sigmoid function
np.save(f'./Symbolic_Regression_for_NN/char_line_sigmoid_parameters_{filter_size}.npy', popt)
# Save the fitted curve values
np.save(f'./Symbolic_Regression_for_NN/char_line_sigmoid_fitted_values.npy_{filter_size}', [x_fit, y_fit])

# Save the corresponding x and y real values
np.save(f'./Symbolic_Regression_for_NN/char_line_xy_values_{filter_size}.npy', [real_x_data, real_y_data])

# Plot real_x and real_y
plt.plot(real_x_data, real_y_data, label='Real Data', color='green', linewidth=2)
plt.show()	

plt.scatter(x_data, y_data, label='Neural Network Data', color='blue', s=5)
plt.plot(x_fit, y_fit, label=f'Fitted Sigmoid\nbeta={beta_opt:.2f}, x0={x0_opt:.2f}', color='red')
# Add rsq to the plot
plt.text(0.5, 0.5, f'R² = {r_squared:.2f}', fontsize=12, ha='center')
plt.legend()
plt.xlabel('Input Feature (Aligned Axis)')
plt.ylabel('Sigmoid Output')
plt.show()
