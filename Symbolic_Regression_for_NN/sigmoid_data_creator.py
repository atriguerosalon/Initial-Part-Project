import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the sigmoid-like function
def custom_sigmoid(x):
    k = -15  # Adjust the steepness
    x_0 = 0.5071  # Center of transition
    return 1 / (1 + np.exp(k * (-x + x_0)))

# Generate x values from 0 to 1.41
x_values = np.linspace(0, 1.41, num=1000)
y_values = custom_sigmoid(x_values)

# Optional: Plot to visualize the results
plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylabel('Sigmoid-like function')
plt.title('Sigmoid-like function starting at 0')
plt.show()

# Output data array
sigmoid_data = np.column_stack((x_values, y_values))

# Save the data to an array
np.save('./Symbolic_Regression_for_NN/sigmoid_data.npy', sigmoid_data)

