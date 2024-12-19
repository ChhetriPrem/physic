import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Constants (in atomic units)
hcross = 1973  # eVÅ
e = 3.795  # eVÅ
m = 0.511 * 10**6  # eV/c^2
r_min, r_max = 1e-5, 20  # range for r (avoiding singularity at r=0)

# Potential function V(r)
def potential(r):
    return -e**2 / r

# Schrödinger equation as a second-order differential equation
def schrodinger(r, y, E):
    y_val, dydr = y
    d2ydr2 = 2 * m / hcross**2 * (potential(r) - E) * y_val
    return [dydr, d2ydr2]

# Boundary conditions
def boundary_conditions(E):
    # Solve using initial guess near r_min
    sol = solve_ivp(
        schrodinger,
        [r_min, r_max],
        [0, 1e-5],  # Initial values: y(r_min)=0, dy/dr(r_min)=small value
        args=(E,),
        dense_output=True
    )
    y_val = sol.y[0]
    return y_val[-1]  # Value of y at r_max (should approach 0 for bound states)

# Finding eigenvalues using the shooting method
def find_eigenvalue(E1, E2):
    return brentq(boundary_conditions, E1, E2)

# Find ground state and first excited state energy
E_ground = find_eigenvalue(-14, -13)
E_excited = find_eigenvalue(-3.5, -3.4)

print(f"Ground state energy: {E_ground:.4f} eV")
print(f"First excited state energy: {E_excited:.4f} eV")

# Solve for wave functions
def solve_wave_function(E):
    sol = solve_ivp(
        schrodinger,
        [r_min, r_max],
        [0, 1e-5],
        args=(E,),
        dense_output=True
    )
    return sol

sol_ground = solve_wave_function(E_ground)
sol_excited = solve_wave_function(E_excited)

# Normalize wave functions
def normalize_wave_function(sol):
    r = np.linspace(r_min, r_max, 1000)
    y_val = sol.sol(r)[0]
    norm = np.sqrt(np.trapz(y_val**2, r))
    return r, y_val / norm

r_ground, y_ground = normalize_wave_function(sol_ground)
r_excited, y_excited = normalize_wave_function(sol_excited)

# Output eigenvalues and arrays
print(f"Eigenvalue for Ground State: {E_ground:.4f} eV")
print(f"Eigenvalue for First Excited State: {E_excited:.4f} eV")

# Print wavefunction arrays (first 10 values for brevity)
print("\nWavefunction for Ground State (first 10 values):")
print(y_ground[:10])

print("\nWavefunction for First Excited State (first 10 values):")
print(y_excited[:10])

# Plot wave functions
plt.figure(figsize=(10, 6))
plt.plot(r_ground, y_ground, label="Ground State")
plt.plot(r_excited, y_excited, label="First Excited State")
plt.title("Wave Functions of Hydrogen Atom")
plt.xlabel("r (Å)")
plt.ylabel("y(r)")
plt.legend()
plt.grid()
plt.show()
