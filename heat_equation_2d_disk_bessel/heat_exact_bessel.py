import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import os

# Parameters
alpha = 0.1                 # Diffusion coefficient
N_r = 20                   # Number of radial grid points
N_theta = 20               # Number of angular grid points
T_final = 2.0               # Final time
M = 1000                      # Number of time steps
dt = T_final / M            # Time step size

# Radial and angular grids
r_vals = np.linspace(1/N_r, 1, N_r)
theta_vals = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)

# Bessel zeros
bessel_zeros_m0 = sp.jn_zeros(0, 4)
bessel_zeros_m1 = sp.jn_zeros(1, 4)
bessel_zeros_m2 = sp.jn_zeros(2, 1)
bessel_zeros_m3 = sp.jn_zeros(3, 1)
bessel_zeros_m4 = sp.jn_zeros(4, 1)

# Define disk harmonics function
def Z_mn(m, n, r, theta):
    if m == 0:
        lambdamn = bessel_zeros_m0[n-1]
    elif m == 1:
        lambdamn = bessel_zeros_m1[n-1]
    elif m == 2:
        lambdamn = bessel_zeros_m2[n-1]
    elif m == 3:
        lambdamn = bessel_zeros_m3[n-1]
    elif m == 4:
        lambdamn = bessel_zeros_m4[n-1]
    else:
        raise ValueError("Only m values up to 4 are supported in this example.")
    return sp.jv(m, lambdamn * r) * np.cos(m * theta), lambdamn

# Define coefficients from initial condition
coefficients = [
    (1/4, 0, 1),
    (-1/16, 0, 2),
    (1/64, 0, 3),
    (-1/256, 0, 4),
    (1/4, 1, 1),
    (-1/8, 1, 2),
    (1/16, 1, 3),
    (-1/32, 1, 4),
    (1/4, 2, 1),
    (1/4, 3, 1),
    (1/4, 4, 1),
]

# Initialize solution array
u_time = np.zeros((M, N_r, N_theta))

output_dir = "heat_solution_files"
os.makedirs(output_dir, exist_ok=True)


# Time evolution loop
for time_step in range(M):
    t = time_step * dt
    u_t = np.zeros((N_r, N_theta), dtype=np.complex128)
    for c, m, n in coefficients:
        Zmn, lambdamn = Z_mn(m, n, r_vals[:, None], theta_vals)
        u_t += c * Zmn * np.exp(-alpha * lambdamn**2 * t)

    filename = os.path.join(output_dir, f"T_{t:.3f}".rstrip("0").rstrip(".") + ".npy")
    np.save(filename, u_t)
    print(f"Saved {filename}")

# Plot the solution at final time
R, Theta = np.meshgrid(r_vals, theta_vals, indexing='ij')
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, u_time[-1, :, :], cmap="viridis")
plt.colorbar(label="u(r, θ, T_final)")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Heat Equation Solution at t = {T_final}")
plt.axis("equal")
plt.savefig("heat_solution.png")
plt.show()
