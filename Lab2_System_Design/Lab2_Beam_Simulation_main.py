import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize


#    1. BEAM PRODUCTION 

# Constants
q = 1.602e-19  # proton charge [C]
m = 1.673e-27  # proton mass [kg]
m_Mev = 938.27231 # proton mass [MeV/c^2]
c = 3e8        # speed of light [m/s]

# Cyclotron specs
E_MeV = 200                 # final energy [MeV]
E = E_MeV * 1e6 * q         # energy in joules
R_max = 0.9                 # maximum radius [m]
t_acc = 0.9e-3              # acceleration time [s]



v_rel = c*np.sqrt(1 - (1 +E_MeV/m_Mev)**(-2) ) # relativistic velocity
gamma = 1 / np.sqrt(1 - (v_rel/c)**2)
m_rel = m * gamma                           # relativistic mass


# Magnetic field at extraction
B_extr = m_rel * v_rel / (q * R_max)

# Cyclotron frequency
f_cyclotron = q * B_extr / (2 * np.pi * m_rel)
T_rev = 1 / f_cyclotron
n_circles = int(t_acc / T_rev)


# High voltage per gap (4 gaps per turn)
V_per_gap = E / (4 * n_circles * q)

#frequency of high voltage
f_high_voltage = f_cyclotron/2

B_center = f_cyclotron * 2 * np.pi * m / q #use restmass for the center, frequency is constant



print("==== Beam Production Calculations for Isochronous Cyclotron ====")
print(f"1. Magnetic field at the center: {B_center:.3f} T")
print(f"2. Magnetic field at extraction (90 cm radius, 200 MeV energy): {B_extr:.3f} T")
print(f"3. Cyclotron frequency: {f_cyclotron / 1e6:.3f} MHz")
print(f"4. Frequency of high voltage: {f_high_voltage / 1e6:.3f} MHz")
print(f"4. Number of revolutions: {n_circles}")
print(f"5. Required high voltage amplitude per gap: {V_per_gap / 1e3:.2f} kV")


#   2. BEAM TRANSPORT

# Constants
sigma_x0 = 3e-3   # Initial beam size [m]
sigma_xp0 = 1e-3  # Initial divergence [rad]
sigma_y0 = 3e-3  
sigma_yp0 = 1e-3
tube_limit = 25e-3  # vacuum tube radius (The full beam size (3 sigma) must always stay smaller than it)


# Initial beam matrix with N particles
def initial_matrix(N):
    initial_matrix = np.zeros((N, 4))
    initial_matrix[:, 0] = np.random.normal(0, sigma_x0, N)   # x
    initial_matrix[:, 1] = np.random.normal(0, sigma_xp0, N)  # x'
    initial_matrix[:, 2] = np.random.normal(0, sigma_y0, N)   # y
    initial_matrix[:, 3] = np.random.normal(0, sigma_yp0, N)  # y'
    return initial_matrix

initial_beam = initial_matrix(5000)

# Transfer matrices (4x4)
def drift_matrix(d):
    return np.array([[1, d, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, d],
                     [0, 0, 0, 1]])

# de-focus matrix 
def focus_matrix(f):
    return np.array([[1, 0, 0, 0],
                     [-1/f, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 1/f, 1]])


# Propagate step-by-step and store intermediate beam sizes and divergences
def propagate_stepwise(initial_beam, d1, d2, d3, d4, f1, f2, f3, f4):

    # define the matrices 
    M1 = focus_matrix(f1)
    M2 = drift_matrix(d1)
    M3 = focus_matrix(f2)
    M4 = drift_matrix(d2)
    M5 = focus_matrix(f3)
    M6 = drift_matrix(d3)
    M7 = focus_matrix(f4)
    M8 = drift_matrix(d4)
    sequence = [M1, M2, M3, M4, M5, M6, M7, M8] # store all matrices
    beam = initial_beam.copy()
    steps = []
    # we calculate the dimensions of the beam after each step
    for M in sequence:
        beam = beam @ M.T 
        steps.append(beam.copy()) # save dimensions of each step
    return steps

# Cost function with symmetry, final size
def cost_function(params):
    d1, d2, d3, d4, f1, f2, f3, f4 = params
    steps = propagate_stepwise(initial_beam, d1, d2, d3, d4, f1, f2, f3, f4)

    # Final beam
    final = steps[-1]
    sigma_x_end = np.std(final[:, 0])
    sigma_y_end = np.std(final[:, 2])
    # The beam must be symmetrical at the isocenter 
    symmetry_penalty = abs(sigma_x_end - sigma_y_end)
    # The beam size (1 sigma) must be a maximum of 3 mm at the isocenter.
    final_size_penalty = (max(0, sigma_x_end - 3e-3) + max(0, sigma_y_end - 3e-3))**2

    return symmetry_penalty + 10000*final_size_penalty 

# Constraint: total length = 7 m
def constraint_total_length(params):
    d1, d2, d3, d4 = params[0:4]
    return 7.0 - (d1 + d2 + d3 + d4)

# the beam cannot touch the tube walls
def constraint_tube(params):
    d1, d2, d3, d4, f1, f2, f3, f4 = params
    steps = propagate_stepwise(initial_beam, d1, d2, d3, d4, f1, f2, f3, f4)  # propagate the beam
    for step in steps:  
        sigma_x = np.std(step[:, 0]) # save all sd along x direction 
        sigma_y = np.std(step[:, 2])
        exit=[]    # save all the distance from the tube
        exit.append(tube_limit- 3 * sigma_x ) # want it >0
        exit.append(tube_limit- 3 * sigma_y ) # want it >0
    return exit

# Optimization setup
bounds = [(0.5, 3)] * 4 + [(-10, 10)] * 4
constraints = [{'type': 'eq', 'fun': constraint_total_length}, 
               {'type': 'ineq', 'fun': constraint_tube}]
initial_guess = [1.5, 1.5, 2.0, 2.0, 2, -2, 2, -2]

# Run optimization
print(f"Beginning optimization...")
sigma_x_final = 0
sigma_y_final = 1
while not math.isclose(sigma_y_final, sigma_x_final, abs_tol=1e-4):   #bc the optimization does not take long and the final beam size was not alwyas symmetrical
    result = minimize(cost_function, initial_guess, bounds=bounds, constraints=constraints)
    # Extract optimal values
    d1, d2, d3, d4, f1, f2, f3, f4 = result.x
    final_rays = propagate_stepwise(initial_beam, d1, d2, d3, d4, f1, f2, f3, f4)[-1]
    sigma_x_final = np.std(final_rays[:, 0])
    sigma_y_final = np.std(final_rays[:, 2])


# Display results
print("==== Optimized Beamline with Intermediate Constraint ====")
print(f"d1 = {d1:.2f} m, d2 = {d2:.2f} m, d3 = {d3:.2f} m, d4 = {d4:.2f} m")
print(f"f1 = {f1:.2f} m, f2 = {f2:.2f} m, f3 = {f3:.2f} m, f4 = {f4:.2f} m")
print(f"Final beam size at isocenter: {sigma_x_final*1e3:.2f} mm (X), {sigma_y_final*1e3:.2f} mm (Y)")


beam = initial_matrix(5000)
steps = propagate_stepwise(beam, d1, d2, d3, d4, f1, f2, f3, f4)

# Plot phase-space at key locations
labels = ["After Q1", "After D1", "After Q2", "After D2", "After Q3", "After D3", "After Q4", "After D4"]
fig, axes = plt.subplots(2, 8, figsize=(24, 8))  # 8 columns to match number of steps

for i, step in enumerate(steps):
    ax_x = axes[0, i]
    ax_y = axes[1, i]

    # Horizontal (x - x')
    ax_x.scatter(step[:, 0]*1e3, step[:, 1]*1e3, s=1, alpha=0.1, color='red')
    ax_x.set_title(f"{labels[i]}")
    ax_x.set_xlabel("x [mm]")
    ax_x.set_ylabel("x' [mrad]")
    ax_x.grid(True)

    # Vertical (y - y')
    ax_y.scatter(step[:, 2]*1e3, step[:, 3]*1e3, s=1, alpha=0.1, color='blue')
    ax_y.set_title(f"{labels[i]}")
    ax_y.set_xlabel("y [mm]")
    ax_y.set_ylabel("y' [mrad]")
    ax_y.grid(True)

plt.tight_layout()
plt.show()


#   3. BEAM DELIVERY

# === Parameters ===
spot_sigma = 2  # [mm] 1σ of Gaussian spot
s_size = int(5 * spot_sigma + 1)  # Size of the dose spot matrix
prescribed_dose = 1.0  # [Gy]
spacing = 2  # [mm] spacing between pencil beams
target_size_mm = 60  # Target is a 60x60 mm square
grid_size_mm = 100  # Dose grid size in mm
resolution = 1  # mm per pixel
grid_size_px = int(grid_size_mm / resolution) # number of pixel per grid size

# === Create Gaussian Spot Template ===
x = np.arange(s_size) - (s_size - 1) / 2
gauss1D = np.exp(-0.5 * (x / spot_sigma)**2)
spot = np.outer(gauss1D, gauss1D)

# === Create Dose Grid ===
dose = np.zeros((grid_size_px, grid_size_px))

# === Create 34x34 Pencil Beam Grid ===
num_spots = 34
start_pos = (grid_size_px - (num_spots - 1) * spacing) // 2  # the amount of space from the border of the grid to the first spot

# === Add Dose from Each Pencil Beam ===
for i in range(num_spots):
    for j in range(num_spots):
        x_center = start_pos + i * spacing # update the position of the centre of the beam
        y_center = start_pos + j * spacing
        x0 = int(x_center - s_size // 2)  # distance of the spot from the left side of the grid
        y0 = int(y_center - s_size // 2)  # distance of the spot from the top of the grid

        if 0 <= x0 < grid_size_px - s_size and 0 <= y0 < grid_size_px - s_size:  # check if the spot is within the grid
            dose[x0:x0 + s_size, y0:y0 + s_size] += spot    # add the spot to the dose grid

# ====  Normalize Total Dose So Average Dose in Target = Prescribed Dose  ====
# Define target area mask
target_start = (grid_size_px - target_size_mm) // 2  # grid position where the target starts (20)
target_end = target_start + target_size_mm  # grid position where the target ends (80)
mask = np.zeros_like(dose)
mask[target_start:target_end, target_start:target_end] = 1 # mask that has 1 where the target is, 0 otherwise

mean_dose_in_target = np.sum(dose * mask) / np.sum(mask) # calculate the average dose for each pixel of the target 
dose *= (prescribed_dose / mean_dose_in_target) # Normalize Total Dose so mean_dose_in_target =1


# === Plot Dose Distribution ===
plt.figure(figsize=(6, 5))
plt.imshow(dose, cmap='jet', origin='lower')
plt.colorbar(label='Dose [Gy]')
plt.title('Static Dose Delivery (No Motion)')
plt.xlabel('X [pixels]')
plt.ylabel('Y [pixels]')
plt.tight_layout()
plt.show()


#   ===== MOTION =====
# Parameters
motion_amplitude = 10  # mm
motion_period = 4  # seconds
delta_t = 0.022  # Time between beams (s)
time = np.linspace(0, 8, 200)  # Simulate two full breathing cycles (0–8 s)


# Updated triangle wave function (starts at -amp)
def triangle_wave(t, amp=motion_amplitude, T=motion_period):
    t = t % T
    return -2 * amp * abs((t / T)-0.5) + amp/2

# Compute motion over time
motion = triangle_wave(time) 


# === Function to Simulate Dose Delivery with Motion ===
def simulate_interplay(parallel=True):
    dose = np.zeros((grid_size_px, grid_size_px))
    start_pos = (grid_size_px - (num_spots - 1) * spacing) // 2
    spot_index = 0


    for i in range(num_spots):
        for j in range(num_spots):
            t = spot_index * delta_t # based on the spot index I find at what time instant I am
            offset = triangle_wave(t)# calculate the relative displacement of the body based on the triangular curve at current time instant

            if parallel:
                # Motion in X direction
                x_center = start_pos + i * spacing - offset  #if target moves to the right, the spot moves to the left
                y_center = start_pos + j * spacing
            else:
                # Motion in Y direction
                x_center = start_pos + i * spacing
                y_center = start_pos + j * spacing - offset

            x0 = int(x_center - s_size // 2)
            y0 = int(y_center - s_size // 2)

            if 0 <= x0 < grid_size_px - s_size and 0 <= y0 < grid_size_px - s_size:  # check if the spot is within the grid
                dose[x0:x0 + s_size, y0:y0 + s_size] += spot # add the spot to the dose grid

            spot_index += 1 

    # Normalize dose (come prima)
    target_start = (grid_size_px - target_size_mm) // 2
    target_end = target_start + target_size_mm
    mask = np.zeros_like(dose)
    mask[target_start:target_end, target_start:target_end] = 1   # select the target area
    mean_dose_in_target = np.sum(dose * mask) / np.sum(mask)
    dose *= (prescribed_dose / mean_dose_in_target) #scale such that the mean dose in the target is equal to the prescribed dose
    return dose

# === Simulate both directions ===
dose_parallel = simulate_interplay(parallel=True)
dose_orthogonal = simulate_interplay(parallel=False)

# === Plot both dose maps ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].imshow(dose_parallel, cmap='jet', origin='lower')
axs[0].set_title("Dose Map: Parallel to Motion")
axs[0].set_xlabel("X [pixels]")
axs[0].set_ylabel("Y [pixels]")
axs[0].grid(False)

axs[1].imshow(dose_orthogonal, cmap='jet', origin='lower')
axs[1].set_title("Dose Map: Orthogonal to Motion")
axs[1].set_xlabel("X [pixels]")
axs[1].set_ylabel("Y [pixels]")
axs[1].grid(False)

plt.tight_layout()
plt.show()


# === Compute and display coverage statistics ===

# assess whether at least 95% of the target receives >0.9 Gy
def evaluate_coverage(dose_map, label):
    target_start = (grid_size_px - target_size_mm) // 2 # 20
    target_end = target_start + target_size_mm #80
    target_dose = dose_map[target_start:target_end, target_start:target_end] # only take the part of the grid where the target is
    covered_voxels = np.sum(target_dose >= 0.9 * prescribed_dose) # sum the number of pixels that received a dose >0.9
    total_voxels = target_dose.size 
    coverage_percentage = (covered_voxels / total_voxels) * 100 # % of voxel that has >0.9 Gy
    print(f"{label}: {coverage_percentage:.2f}% of target receives ≥ 90% of prescribed dose")

print('almeno il 95 perc del target ha raggiunto almeno >0.9 Gy ?')
evaluate_coverage(dose_parallel, "Parallel scan")
evaluate_coverage(dose_orthogonal, "Orthogonal scan")
print('per il movimento orizzontale no')
print('---- implementiamo interpolazione -----')
# the objective is not reached for horizontal motion, we have to plan the beam delivery in a way that the target is reached

#  ==== interplay reduction strategy ====
# Implement: wait for alignment before firing the beam
def simulate_adaptive_spot_triggering(tolerance_mm=2):
    dose = np.zeros((grid_size_px, grid_size_px))
    start_pos = (grid_size_px - (num_spots - 1) * spacing) // 2 # 17

    spot_index = 0
    t = 0.0  # Start simulation time
    for i in range(num_spots):
        for j in range(num_spots):
            spot_position_x = start_pos + i * spacing
            spot_position_y = start_pos + j * spacing
              
            # Wait until tumor aligns with beam spot (within tolerance) 
            while True:
                offset = triangle_wave(t) # displacement at the current t
                tumor_x = offset
                if abs(tumor_x) <= tolerance_mm: # if the displacement is lower than the tollerance
                    break # I'm quite aligned, I can fire the spot
                t += delta_t  # Wait for next possible time slot

            # Fire the beam spot now that tumor is aligned
            x_center = spot_position_x - offset  # motion-compensated delivery 
            y_center = spot_position_y

            x0 = int(x_center - s_size // 2)
            y0 = int(y_center - s_size // 2)

            if 0 <= x0 < grid_size_px - s_size and 0 <= y0 < grid_size_px - s_size:
                dose[x0:x0 + s_size, y0:y0 + s_size] += spot

            spot_index += 1
            t += delta_t  # Move to next delivery slot

    # Normalize dose (come prima)
    target_start = (grid_size_px - target_size_mm) // 2
    target_end = target_start + target_size_mm
    mask = np.zeros_like(dose)
    mask[target_start:target_end, target_start:target_end] = 1
    mean_dose_in_target = np.sum(dose * mask) / np.sum(mask)
    dose *= (prescribed_dose / mean_dose_in_target)
    return dose

# Run adaptive spot triggering simulation
list_tolerance = [0.5, 1.0, 1.5, 2.0, 2.5]
for tolerance in list_tolerance:
    dose_adaptive = simulate_adaptive_spot_triggering(tolerance_mm=tolerance) 
    evaluate_coverage(dose_adaptive, f"Adaptive Spot Triggering (Tolerance {tolerance} mm)") 
    if tolerance == 2.0: 
        break



# Plot dose map
plt.figure(figsize=(6, 5))
plt.imshow(dose_adaptive, cmap='jet', origin='lower')
plt.title("Adaptive Spot Triggering (Strategy 6)")
plt.xlabel("X [pixels]")
plt.ylabel("Y [pixels]")
plt.colorbar(label="Dose [Gy]")
plt.tight_layout()
plt.show()


