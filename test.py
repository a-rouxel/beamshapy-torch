import matplotlib.pyplot as plt
import numpy as np
import torch
from cost_functions import generalized_sigmoid_function




x_values = torch.arange(-10, 10, 0.01)
y_values = generalized_sigmoid_function(x_values, steepness=7, num_terms=3, spacing=5)

plt.figure(figsize=(8, 6))
plt.plot(x_values.numpy(), y_values.numpy(), label="34terms")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Generalized Sigmoid Function")
plt.legend()
plt.grid(True)
plt.show()

# Plot the generalized sigmoid function for different numbers of terms
