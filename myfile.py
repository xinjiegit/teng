import numpy as np

data = np.load('./heat_equation_2d_spectral_fourier/T_2.npy')

print(type(data))          # Check the type of the data (e.g., numpy array)
print(data.shape)          # Print the shape of the array
print(data.dtype)          # Print the data type of the array
