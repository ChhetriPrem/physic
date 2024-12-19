import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, epsilon_0, m_e, hbar

# Constants (in SI units)
a_0 = 5.29177e-11  # Bohr radius in meters
e2_over_4pi_epsilon = e**2 / (4 * np.pi * epsilon_0)  # e^2 / (4 * pi * epsilon_0)

# Define the potential function for the Coulomb potential (in SI units)
def potential(r):
    return - e2_over_4pi_epsilon / r

# Discretizing the space
r_min = 1e-10  # Minimum radius (in meters)
r_max = 20e-10  # Maximum radius (in meters)
num_points = 1000  # Number of points in the grid

r = np.linspace(r_min, r_max, num_points)  # Radial distances
dr = r[1] - r[0]  # Step size

# Kinetic energy term (finite difference approximation)
T = -0.5 * hbar**2 / m_e * (np.diag(np.ones(num_points-1), -1) - 2 * np.diag(np.ones(num_points)) + np.diag(np.ones(num_points-1), 1)) / dr**2

# Potential energy term (diagonal matrix for Coulomb potential)
V = np.diag(potential(r))

# Total Hamiltonian matrix
H = T + V

# Solve the eigenvalue problem
eigvals, eigvecs = np.linalg.eigh(H)

# Ground state energy and corresponding wavefunction
ground_state_energy = eigvals[0]  # First eigenvalue corresponds to the ground state
ground_state_wavefunction = eigvecs[:, 0]

# Normalize the wavefunction
normalization = np.sqrt(np.trapz(ground_state_wavefunction**2, r))  # Integrate and normalize
ground_state_wavefunction /= normalization

# Convert energy to eV for better interpretation
energy_ev = ground_state_energy / (1.6e-19)  # Convert from Joules to eV

# Plotting the results (ground state)
plt.figure(figsize=(8, 6))
plt.plot(r * 1e10, ground_state_wavefunction**2, label=f"Ground State Energy = {energy_ev:.3f} eV")
plt.title('Radial Probability Density for Ground State')
plt.xlabel('Radial Distance (Angstroms)')
plt.ylabel('Probability Density (|R(r)|^2)')
plt.grid(True)
plt.legend()
plt.xlim(0, 20)  # Adjust the x-axis to match the range from the image
plt.show()

# Print the energy in eV (convert from Joules)
print(f"Ground State Energy: {energy_ev:.3f} eV")
