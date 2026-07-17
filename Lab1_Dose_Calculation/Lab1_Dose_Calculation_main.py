
# template for Monte Carlo lab (GBIO2070)

import matplotlib.pyplot as plt
import numpy as np
from get_voxel_index import *
from compute_distance_to_interface import *
from scipy.io import loadmat
import math
from cdirect import *

# Load a .mat file
#mio
reference_question2 = loadmat("MCsquare_results/dose_MCsquare_EM_and_nuclear.mat")
dose_data = reference_question2["dose_MCsquare"]
#mio

####### Initialization #######
'''Check units of all variables and functions --> work in mm and MeV'''
# Generate stopping power database
SP_water = np.loadtxt('data/SP_water.txt', 'float', '#', None, None, 8)
SP_database = []
SP_database.append([]) # index 0 is empty to be consistant with matlab code
SP_database.append(SP_water) # index 1 = water
# simulation geometry
grid_size = np.array([100, 100, 400])  # size of the voxel grid
voxel_size = np.array([1., 1., 1.])    # (mm)
offset = np.array([-50., -50., 0.])    # coordinates of the first voxel (mm)
density_map = np.ones(grid_size) * 1.0  *1e-3 # (g/cm3)  changed it to mm^3 from cm^3
material_map = np.ones(grid_size) * 1  # 1 = water
scoring_grid = np.zeros(grid_size)     # initialize the dose distribution grid

# simulation parameters
Beam_position = np.array([0., 0., 0.])    # (mm)
Beam_direction = np.array([0., 0., 1.])
Beam_mean_energy = 200.        # (MeV)
Spot_size = np.array([1., 3.]) # standard deviation (mm)
Beam_energy_spread = 1.        # (MeV)
step_length = 1.               # (mm)
Num_particles = 5000


####### Simulation algorithm #######


# initialize a new particle
position = Beam_position	# position of the current particle (mm)
direction = Beam_direction	# direction of the current particle
energy = Beam_mean_energy	# energy of the current particle (MeV)
    
# simulation of particle transport

##################################
#### IMPLEMENT YOUR CODE HERE ####
##################################

#some constants
m_e = 0.511 # MeV, electron mass
m_p = 938.272 # MeV, proton mass
m_o=  14914.6 # MeV, oxygen mass
r_e = 2.818e-12 # mm, classical electron radius
T_min = 1 # MeV, minimum energy transfer (can choose any value)

E_s = 12 # MeV, stated in the paper (for water)
X_w = 36.0863  # mm, radiation length of water
density_water = 1 *1e-3 # g/cm^3, density of water  (set to mm^3)
#c= 29979245800 #cm/s
c= 1
n_e = density_water*6.022e23*(2/18 + 8/18) # number of electrons per atom (electron density)

# calcoliamo la RADIATION LENGTH del materiale X0(rho)
def radiation_length(density): # X0(rho)  rho should be in g/cm^3

    density = density * 1e3 #convert to cm^3
    def f_x0(rho): 
        """Compute f_x0 based on the given piecewise function."""
        #rho is in g/cm^3
        if rho >= 0.9:
            return 1.19 + 0.44 * np.log(rho - 0.44)
        elif 0.26 <= rho < 0.9:
            return 1.0446 - 0.2180 * rho
        else:  # rho <= 0.26
            return 0.9857 + 0.0085 * rho
    return density_water* 1e3 /density * X_w/f_x0(density)


def actual_step_length(position, direction, voxel_size, offset):
    computed_distance = compute_distance_to_interface(position, direction, voxel_size, offset)
    return min(computed_distance, step_length)

#Question 2: NUCLEAR INTERACTION

#Macroscopic cross sections from the paper (fitted)

def macroscopic_cross_section_pp(energy):
    if energy <= 10 or energy >= 300:
        return 0
    return 0.315 * energy**(-1.126) + 3.78e-6 * energy

def macroscopic_cross_section_pO_elastic(energy):
    if energy <= 50 or energy >= 250:
        return 0
    return 1.88 / energy + 4e-5 * energy - 0.01475

def macroscopic_cross_section_pO_inelastic(energy):
    if energy <=7 or energy >= 250:
        return 0
    cross_section = 0.001 * (1.64 * (energy - 7.9) * np.exp(-0.064 * energy + 7.85 / energy) + 9.86)
    #below 7 MeV, the cross section is 0 (see paper), otherwise it is negative
    return cross_section
def dz_nucl(mu_pp, mu_pO, mu_pO_inelastic):
    mu_tot = mu_pp + mu_pO + mu_pO_inelastic
    if mu_tot == 0:
        return 100000000000000000   #make the path length very big, equivalent to having no interaction
    lambda_tot=1/(mu_pp + mu_pO + mu_pO_inelastic)
    xi = np.random.uniform(0, 1)
    return -lambda_tot*math.log(xi)*10

def sample_interaction_type(mu_pp, mu_pO, mu_pO_inelastic): # sample the TYPE of interaction
    # valuta qual'è la probabilità (xi) e in base a quella estrae la cross section dell'interazione
    mu_total = mu_pp + mu_pO + mu_pO_inelastic
    xi = np.random.uniform(0, 1)

    if 0 < xi <= mu_pp / mu_total: # p vs p elastic, the incident proton gives up energy to the other proton and both protons change direction.
        return 'pp'
    elif mu_pp / mu_total < xi <= (mu_pp + mu_pO) / mu_total: # p vs oxygen elastic, the incident proton gives up energy to the other proton and both protons change direction.
        return 'po_e'  
    else:
        return 'po_i' # inelastic, so energy changes


#This function is used to determine the angle at which the scattered proton moves in the lab frame, given its angle in the CM frame (see lides)
def CM_to_labframe(M1,M2,energy,cos_thetaCM):
    betaCM=math.sqrt(energy*(energy+2*M1*c**2))/(energy+M1*c**2+M2*c**2)
    gammaCM=1/math.sqrt(1-betaCM**2)
    tau=math.sqrt(((M1/M2)**2)*(1-betaCM**2)+betaCM**2)
    cos_theta=(cos_thetaCM+tau)/math.sqrt((cos_thetaCM+tau)**2+(1/(gammaCM**2))*(math.sin(math.acos(cos_thetaCM)))**2)
    return cos_theta

#samples the energy transfer to the oxygen nucleus in the elastic interaction
def To_sample(energy):
    beta = np.sqrt(1 - (m_p/(energy+m_p))**2)
    gamma = (energy+m_p)/m_p   
    T_0_max = 2*m_o*(beta*gamma)**2/(1+2*gamma*m_o/m_p+(m_o/m_p)**2)
    T_0_mean = 0.65*math.exp(-0.0013*energy)-0.71*math.exp(-0.0177*energy)

    # Scaling parameter of the exponential distribution
    scale = T_0_mean
    if scale < 0:
        scale = 0
    # Function to sample from a truncated exponential distribution
    def sample_truncated_exponential(scale, T_0_max):
        while True:
            # Sample from the exponential distribution
            #sample = np.random.exponential(scale)
            sample = -scale * np.log(np.random.uniform()*scale)   #CHANGED THIS LINE
            # If the sample is less than T_max, return it
            if sample <= T_0_max:
                return sample

    # Sampling
    T_0_sample = sample_truncated_exponential(scale, T_0_max)
    return T_0_sample

#deflection angle for oxygen in the elastic interaction, not converted to lab frame
def deflectionAngle_po_e(energy, T_0,m_o,c):
    #M1 = m_p
    #M2 = m_o
    #betaCM=math.sqrt(energy*(energy+2*M1*c**2))/(energy+M1*c**2+M2*c**2)
    #gammaCM=1/math.sqrt(1-betaCM**2)  
    #cos_thetaCM=1-(T_0/((betaCM**2)*(gammaCM**2)*m_o*c**2))
    #tau=math.sqrt(((M1/M2)**2)*(1-betaCM**2)+betaCM**2)
    #cos_theta=(cos_thetaCM+tau)/math.sqrt((cos_thetaCM+tau)**2+(1/(gammaCM**2))*(math.sin(math.acos(cos_thetaCM)))**2)

    # WE DON'T HAVE TO USE THE CM FORMULAS FROM THE SLIDES?
    beta = np.sqrt(1 - (m_p/(energy+m_p))**2)
    gamma = (energy+m_p)/m_p   
    cos_thetaCM=1-(T_0/((beta**2)*(gamma**2)*m_o*c**2))
    return cos_thetaCM
#compute the CSDA
def CSDA(energy =0,direction=0,position=0, initialize= True, central_axis=False):

    # initialize a new particle
    # 2. Sample initial particle position
    if initialize:
        #position = Beam_position + [np.random.normal(scale= Spot_size[0]), np.random.normal(scale= Spot_size[1]), 0]	# position of the current particle (mm)
        if central_axis:
            position = Beam_position.copy()   #extra question
        else:
            position = Beam_position + [np.random.normal()*Spot_size[0], np.random.normal()*Spot_size[1], 0]	# position of the current particle (mm)
        direction = Beam_direction	# direction of the current particle
        # 3. Sample initial particle energy
        #energy = Beam_mean_energy + np.random.normal(scale = Beam_energy_spread)	# energy of the current particle (MeV)
        energy = Beam_mean_energy + np.random.normal()*Beam_energy_spread	# energy of the current particle (MeV)
    
    voxel_index = get_voxel_index(position, voxel_size, offset)  

    # simulation of particle transport
    while energy > 0:
            
        # 2. simulate the random sampling of the initial position
        # get the material index of the current voxel
        voxel_index = get_voxel_index(position, voxel_size, offset)
        # check if the particle is still inside the grid
        if np.any(voxel_index < 0) or np.any(voxel_index >= grid_size):
            break
        # Find the closest energy index
        if density_map[voxel_index[0], voxel_index[1], voxel_index[2]] == 1e-3 or case == 'test': # water
            SP_interpolated = np.interp(energy, SP_database[1][:,0], SP_database[1][:,1])*1e2
            SP_ratio = 1
            n_e = density_water*6.022e23*(2/18 + 8/18)
        elif density_map[voxel_index[0], voxel_index[1], voxel_index[2]] == 1.85e-3: # compact bone
            SP_interpolated = np.interp(energy, SP_database[2][:,0], SP_database[2][:,1])*1e2
            SP_ratio = np.interp(energy, SP_database[1][:,0], SP_database[1][:,1])*1e2 / SP_interpolated
            n_e = 1.7*density_water*6.022e23*(2/18 + 8/18) # number of electrons per atom (electron density)
        # 1. compute the energy loss 
        actual_step = actual_step_length(position, direction, voxel_size, offset)
        energy_loss = SP_interpolated * actual_step * density_map[voxel_index[0], voxel_index[1], voxel_index[2]] * SP_ratio
        
        
        beta = np.sqrt(1 - (m_p/(energy+m_p))**2)
        gamma = (energy+m_p)/m_p   

        # 4. energy straggling (Fippel paper)
        T_max = 2*m_e*(beta*gamma)**2/(1+2*gamma*m_e/m_p+(m_e/m_p)**2)
       
        Sigma = 2*np.pi*r_e**2*m_e*n_e*actual_step*(T_max/beta**2)*(1-beta**2/2)
        
        # 1. compute energy total loss= per step length + straggling
        #energy_loss = energy_loss + np.random.normal(scale=np.sqrt(Sigma))
        energy_loss = energy_loss + np.random.normal()*np.sqrt(Sigma)
        # 5. multiple-coulumb scattering (Fippel paper)
        p = np.sqrt((energy+m_p)**2 - m_p**2) #momentum
        theta_0 = E_s/(p*beta) *np.sqrt(actual_step/radiation_length(density_map[voxel_index[0], voxel_index[1], voxel_index[2]])) 
        
        step_nucl= dz_nucl(macroscopic_cross_section_pp(energy),macroscopic_cross_section_pO_elastic(energy), macroscopic_cross_section_pO_inelastic(energy))
        if step_nucl<actual_step:
            actual_step=step_nucl
            # so we have to consider a Nuclear interaction for this step
            # which type of interaction is it?
            type=sample_interaction_type(macroscopic_cross_section_pp(energy),macroscopic_cross_section_pO_elastic(energy), macroscopic_cross_section_pO_inelastic(energy))
            if type=='pp':
                xi = np.random.uniform(0, 1)
                cos_theta_CM_inc=2*xi-1
                W=energy*(1-cos_theta_CM_inc)/2
                cos_theta_inc=CM_to_labframe(m_p,m_p,energy,cos_theta_CM_inc)
             
                # update the totale lost energy and direction 
                energy_loss=energy_loss+W
                theta_0=theta_0+math.acos(cos_theta_inc) 
                
                #add new proton to the structure:
                cos_theta_CM_targ=-cos_theta_CM_inc
                cos_theta_targ=CM_to_labframe(m_p,m_p,energy,cos_theta_CM_targ)
                theta_0_new = theta_0+math.acos(cos_theta_targ)
                # update the particle position
                position_new = position + actual_step * direction
                # 5. update the particle direction considering the multiple-coulumb scattering, the next iteration we hace a new direction
                #theta_new = np.random.normal(scale=theta_0_new) 
                theta_new = np.random.normal() *theta_0_new
                direction_new=cdirect(math.cos(theta_new),direction[0],direction[1],direction[2])
                CSDA(W,direction_new,position_new, initialize=False)
                # TO DO : add the new proton created (target proton) to the structure
                # # aggiungere la particella salvando posizione (=x,y,z della particella incidente)+energia(=energia ceduta dalla particella incidente)+direzione(=mettendo cos_theta_targ dentro all afunz data)

            elif type=='po_e': ## p vs oxygen elastic, the incident proton gives up energy to oxygen and both particles change direction.
                # energy
                T0= To_sample(energy) # energy loss
                cos_theta_CM_inc=deflectionAngle_po_e(energy, T0,m_o,c)
                cos_theta_CM_targ=-cos_theta_CM_inc  
                # change of system
                cos_theta_inc=CM_to_labframe(m_p,m_o,energy,cos_theta_CM_inc)
                cos_theta_targ=CM_to_labframe(m_p,m_o,energy,cos_theta_CM_targ)
                # update the totale lost energy and direction 
                energy_loss=energy_loss+T0
                theta_0=theta_0+math.acos(cos_theta_inc) 
                
            
            elif type=='po_i':
                E_min=3 #MeV
                E_binding=5 #MeV
                E_system=energy-E_binding
                while E_system>E_min:
                    # secondary particle generation
                    E_secondary=np.random.uniform(E_min, E_system) # sample the energy of the secondary particle
                    type_secondary=['p','lrp','srp']
                    prob=[0.50, 0.465, 0.035]
                    type_p = np.random.choice(type_secondary, p=prob) # sample the type of the secondary particle
                    if type_p=='p':
                        #see equation 28 in paper, sample new direction
                        cos_theta_new = np.random.uniform(2*E_secondary/(E_secondary) -1,1)
                        theta_0_new = theta_0+ math.acos(cos_theta_new)
                        # update the particle position
                        position_new = position + actual_step * direction
                        # 5. update the particle direction considering the multiple-coulumb scattering, the next iteration we hace a new direction
                        #theta_new = np.random.normal(scale=theta_0_new)
                        theta_new = np.random.normal() *theta_0_new
                        direction_new=cdirect(math.cos(theta_new),direction[0],direction[1],direction[2])
                        CSDA(E_secondary,direction_new,position_new, initialize=False)
                        
                    E_system =E_system-E_secondary-E_binding # update the remaining system energy
                    energy_loss=energy_loss+E_secondary #? the proton loses energy equal to that transferred to the secondary particle




        
        # update the particle energy
        energy = energy - energy_loss
        # update the particle position
        position = position + actual_step * direction
        # 5. update the particle direction considering the multiple-coulumb scattering, the next iteration we hace a new direction
        #direction = random_small_scatter(direction, theta_0)
        #MIO
       # theta = np.random.normal(scale=theta_0) 
        theta = np.random.normal() *theta_0
        direction=cdirect(math.cos(theta),direction[0],direction[1],direction[2])
        # update the dose grid
        scoring_grid[voxel_index[0], voxel_index[1], voxel_index[2]] += energy_loss

        #mentioned in the paper, if below 0.5 MeV, it is abosrbed locally
        if energy < 0.5:
            scoring_grid[voxel_index[0], voxel_index[1], voxel_index[2]] += energy
            energy = 0
        

#Extra questions
case = 'test'
if case == 'nothing':

    density_map[:,:,:] = 1e-3

elif case == 'test':
    density_map[:,:, :int(grid_size[-1]/2)] = 2e-3
    Spot_size[0] = 3
    Spot_size[1] = 3

elif case == 'homogeneous':
    SP_compactbone = np.loadtxt('data/SP_compactbone.txt', 'float', '#', None, None, 8)
    #add  bone to the database if not already there
    if len(SP_database) == 2:
        SP_database.append(SP_compactbone)
    print(f"length of SP_database: {len(SP_database)}")
    density_map[:,:,:] = 1.85e-3  #density of compact bone
elif case == 'inhomogeneous':
    SP_compactbone = np.loadtxt('data/SP_compactbone.txt', 'float', '#', None, None, 8)
    if len(SP_database) == 2:
        SP_database.append(SP_compactbone)
    density_map[:,:,:] = 1e-3
    for i in range(1, 400, 2):
        density_map[:,:,i] = 1.85e-3


# simulate it for 1000 particles
print('Simulating...')
for i in range(Num_particles):
    CSDA(central_axis=True)


   

####### Post-processing & display #######
    
# Convert scored energy into dose with Gray units
scoring_grid # (MeV)
voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2] / 1000 # (cm3)
dose = scoring_grid / (density_map * voxel_volume) # (MeV / g)
dose = dose * 1000          # (MeV / kg)
dose = dose * 1.602176e-13  # (Gy = J/kg)
dose = dose/Num_particles   # normalize in dose per primary proton

# plot result
voxel_index = get_voxel_index(Beam_position, voxel_size, offset)

plt.figure(figsize=(15,4))
plt.subplot(1, 4, 1)
plt.imshow(np.transpose(dose[:,:,int(round(grid_size[2]/2))]), cmap="jet")
plt.title("Dose map (XY slice)")
plt.xlabel("X (voxels)")
plt.ylabel("Y (voxels)")

plt.subplot(1, 4, 2)
plt.imshow(dose[voxel_index[0],:,:], cmap="jet", aspect=4)
plt.title("Dose map (YZ slice)")
plt.xlabel("Z (voxels)")
plt.ylabel("Y (voxels)")

plt.subplot(1, 4, 3)
plt.imshow(dose[:,voxel_index[1],:], cmap="jet", aspect=4)
plt.title("Dose map (XZ slice)")
plt.xlabel("Z (voxels)")
plt.ylabel("X (voxels)")

plt.subplot(1, 4, 4)
z = np.arange(offset[2], grid_size[2]*voxel_size[2], voxel_size[2])
plt.plot(z,np.sum(dose, axis=(0,1)))
plt.title("Integrated depth-dose")
plt.ylabel('Integrated dose')
plt.xlabel('Depth (voxels)')
plt.xlim(0, 400)
plt.ylim(0, max(np.sum(dose, axis=(0,1))))

plt.tight_layout()
plt.show()

# 'reference'= risultati da comparare
plt.figure(figsize=(15,4))
plt.subplot(1, 4, 1)
plt.imshow(np.transpose(dose_data[:,:,int(round(grid_size[2]/2))]), cmap="jet")
plt.title("Dose map reference (XY slice)")
plt.xlabel("X (voxels)")
plt.ylabel("Y (voxels)")

plt.subplot(1, 4, 2)
plt.imshow(dose_data[voxel_index[0],:,:], cmap="jet", aspect=4)
plt.title("Dose map reference (YZ slice)")
plt.xlabel("Z (voxels)")
plt.ylabel("Y (voxels)")

plt.subplot(1, 4, 3)
plt.imshow(dose_data[:,voxel_index[1],:], cmap="jet", aspect=4)
plt.title("Dose map reference (XZ slice)")
plt.xlabel("Z (voxels)")
plt.ylabel("X (voxels)")

plt.subplot(1, 4, 4)
z = np.arange(offset[2], grid_size[2]*voxel_size[2], voxel_size[2])
plt.plot(z,np.sum(dose_data, axis=(0,1)))
plt.title("Integrated depth-dose reference")
plt.ylabel('Integrated dose')
plt.xlabel('Depth (voxels)')
plt.xlim(0, 400)
plt.ylim(0, max(np.sum(dose_data, axis=(0,1))))

plt.tight_layout()
plt.show()
