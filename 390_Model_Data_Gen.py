import pandas as pd
import numpy as np
import time

start_time = time.time()

delta_t = (1/180) #defining as global variable, unit is seconds

#----- Function definitions -----

#Calculates the position of the slider relative to the base of the crank for given set of input parameters
def slider_displacement(crank_radius, link_length, offset, crank_angle):
    X_s = np.sqrt(link_length**2-(offset+crank_radius*np.sin(crank_angle))**2)+crank_radius*np.cos(crank_angle)
    return X_s

#Calculates the velocity of the slider relative to the base of the crank for given set of input parameters
def slider_velocity(crank_radius, link_length, offset, crank_angle):
    position_before = slider_displacement(crank_radius, link_length, offset, (crank_angle - np.pi/180))
    position_after = slider_displacement(crank_radius, link_length, offset, (crank_angle + np.pi/180))
    V_s = (position_after-position_before)/(2*delta_t)
    return V_s

#Calculates the acceleration of the slider relative to the base of the crank for given set of input parameters
def slider_acceleration(crank_radius, link_length, offset, crank_angle):
    velocity_before = slider_velocity(crank_radius, link_length, offset, (crank_angle - np.pi/180))
    velocity_now = slider_velocity(crank_radius, link_length, offset, crank_angle)
    velocity_after = slider_velocity(crank_radius, link_length, offset, (crank_angle +  np.pi/180))
    A_s = (velocity_after-2*velocity_now+velocity_before)/(delta_t**2)
    return A_s

#Calculate angle of link relative to horizontal plance
def angle_phi(crank_radius, link_length, offset, crank_angle):
    phi = np.asin((crank_radius*np.sin(crank_angle)+offset)/link_length)
    return phi

#Calculates total cross-sectional area occupied by the slider crank mechanism in xy plane, disregarding slider size (currently)
def mechanism_xy_area(crank_radius, link_length, offset):
    area = 2*crank_radius*(crank_radius+np.sqrt((crank_radius+link_length)**2-offset**2))
    return area

#Calculate required link area for given link load to remain within allowable range
def link_sizing(link_force, allowable_stress):
    cross_sectional_area = link_force/allowable_stress
    link_width = np.sqrt(cross_sectional_area)
    return link_width

#Calculate required pin size for given pin load
def pin_sizing(shear_force, allowable_stress):
    cross_sectional_area = shear_force/allowable_stress
    pin_radius = np.sqrt(cross_sectional_area/np.pi)
    return pin_radius

#Calculate force going through link
def link_reaction_force(slider_mass, slider_acceleration, angle_phi, coefficient_mu):
    slider_normal_force = (slider_mass*slider_acceleration+slider_mass*9.81*1/np.tan(angle_phi))/(1/np.tan(angle_phi)-coefficient_mu)
    link_reaction_force = (slider_normal_force-slider_mass*9.81)/np.sin(angle_phi)
    return link_reaction_force

#Calculate torque on crank
def crank_torque(force_magnitude, crank_radius, theta_angle, phi_angle):
    torque = force_magnitude*crank_radius*(np.cos(phi_angle)*np.sin(theta_angle)+np.sin(phi_angle)*np.cos(theta_angle))
    return torque

#Min link width for no buckling (Euler condition)
def link_buckling(link_force, elastic_modulus, link_length, safety_factor):
    max_allowable_load = abs(link_force*safety_factor)
    min_inertia_moment = max_allowable_load*link_length**2/(np.pi**2*elastic_modulus)
    min_link_width = (min_inertia_moment*12)**0.25
    return min_link_width

#--------------

#np.arange creates an array of numbers with specified step size
#Currently just creating all possible combinations, then doing a sweep and filtering out parameters which don't meet criteria

crank_length = np.arange(100,300,0.5, dtype='f8')*10**-3 #Range of all possible crank lengths to check, syntax is (min,max,step)

link_length = np.arange(100,300,0.5, dtype='f8')*10**-3 #using dtype='f8', a.k.a fp64 in order to maintain max precision. f4 (fp32) might be faster

offset = np.arange(0.5,50,0.5, dtype='f8')*10**-3 #Using step size of 0.5mm for all lengths, could be adjusted up/down based on machining tolerance & whatnot

crank_grid, offset_grid = np.meshgrid(crank_length, offset) #meshgrid creates two large matrices. 
#The first input vector is taken as a row which is then copied vertically a number of times equal to the second vector's length. 
#The second input vector is taken as a column, which is then copied horizontally a number of times equal to the first vector's length
#End result is that picking a matrix coordinate and looking at both matrices will give a different combination of the initial input vectors' values

filtered_parameters = [] #Parameters which pass all filter steps end up in here

for link in link_length: #loops through all link length values, combined with above matrices ends up doing full sweep of all permutations
#Grashof condition check
    grashof_mask = (crank_grid + offset_grid <= link)

    if not np.any(grashof_mask): #checks to make sure that at least one set of parameters meets grashof condition
        continue

    g_crank_grid = crank_grid[grashof_mask] #Naming convetion for masked matrices is to add a letter representing each additional filter

    g_offset_grid = offset_grid[grashof_mask]
#-------------

#Stroke distance condition check
    stroke_condition_term_1 = (g_crank_grid + link)**2 - g_offset_grid**2

    stroke_condition_term_2 = (g_crank_grid - link)**2 - g_offset_grid**2

    #Checking for which values the terms in the square root are positive (i.e physically possible)
    stroke_condition_positive_mask = (stroke_condition_term_1 >= 0) & (stroke_condition_term_2 >= 0)

    g_s_crank_grid = g_crank_grid[stroke_condition_positive_mask] #mask the grids to match dimensions of stroke_mask for further masking

    g_s_offset_grid = g_offset_grid[stroke_condition_positive_mask]

    #Calculating square roots avoiding program error by masking out any negative values
    stroke = np.sqrt(stroke_condition_term_1[stroke_condition_positive_mask])-np.sqrt(stroke_condition_term_2[stroke_condition_positive_mask])

    stroke_mask = (stroke >= 249.9*10**-3) & (stroke <= 250.1*10**-3) #the tolerance here drastically affects the amount of parameter sets which filter through

    if not np.any(stroke_mask): #check to ensure at least one valid stroke condition
        continue

    g_s_s_crank_grid = g_s_crank_grid[stroke_mask]

    g_s_s_offset_grid = g_s_offset_grid[stroke_mask]
#--------------

#Calculate return ratio
    alpha_angle = (np.asin(g_s_s_offset_grid/(link-g_s_s_crank_grid)) - np.asin(g_s_s_offset_grid/(link + g_s_s_crank_grid)))

    return_ratio = ((np.pi+alpha_angle)/(np.pi-alpha_angle))

    return_ratio_mask = (return_ratio >= 1.5) & (return_ratio <= 2.5) #filters for return ratio within specified range. This could potentially be uncapped
    #and values greater than 2.5 would likely be filtered out later during stress analysis

    if not np.any(return_ratio_mask): #check to ensure at least one valid return_ratio condition, otherwise skip to next link
        continue

    return_ratio_grid = return_ratio[return_ratio_mask]

    g_s_s_r_crank_grid = g_s_s_crank_grid[return_ratio_mask]

    g_s_s_r_offset_grid = g_s_s_offset_grid[return_ratio_mask]
#---------------

#Calculate cross-sectional area of crank-slider mechanism
    xy_area = mechanism_xy_area(g_s_s_r_crank_grid, link, g_s_s_r_offset_grid) #mm^2
#--------------

    batch_df = pd.DataFrame({
        "Crank": g_s_s_r_crank_grid,
        "Link": link,
        "Offset": g_s_s_r_offset_grid,
        "Return Ratio": return_ratio_grid,
        "Cross Sectional Area": xy_area
             
    }) #create dataframe of all data in batch

    filtered_parameters.append(batch_df)

final_param_output = pd.concat(filtered_parameters, ignore_index=True) #format final dataframe

print(final_param_output)

final_param_output.to_csv("parameters.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet

#Kinematics Analysis
angle_inputs_deg = np.arange(1,360+1,1, dtype='f8') #moved to 1 deg increments, 15 appeared too coarse to be useful

angle_inputs_rad = np.deg2rad(angle_inputs_deg)

kinematic_data = []
peak_data = []

parameters_matrix = final_param_output[["Crank","Link","Offset"]].to_numpy()

#Goes row by row through parameter combinations previously filtered, then calculates kinematics for each angle
slider_mass = 0.5 #Kg
friction_coefficient_mu = 0.1
safety_factor = 1.3

index = 0
for row in parameters_matrix:
    index +=1
    crank, link, offset = row[0], row[1], row[2]
    
    x_s = slider_displacement(crank, link, offset, angle_inputs_rad) #unit
    v_s = slider_velocity(crank, link, offset, angle_inputs_rad) #unit/s
    a_s = slider_acceleration(crank, link, offset, angle_inputs_rad) #unit/s^2
    phi = angle_phi(crank, link, offset, angle_inputs_rad) #rad
    link_force = link_reaction_force(slider_mass, a_s, phi, friction_coefficient_mu) #Newton
    torque = crank_torque(link_force, crank, angle_inputs_rad, phi) #Torque N.m

    #copy crank/link/offset 360 times to fill columns to same length as other variables
    crank_column = np.full_like(angle_inputs_rad, crank)
    link_column = np.full_like(angle_inputs_rad, link)
    offset_column = np.full_like(angle_inputs_rad, offset)

    temp_batch_np = np.column_stack([
        crank_column, #crank
        link_column, #link
        offset_column, #offset
        np.rad2deg(angle_inputs_rad), #theta, degrees
        np.rad2deg(phi), #phi, degrees
        link_force, #N, force through connecting link
        torque,
        x_s, #Xs
        v_s, #Vs
        a_s #As
        ])
    
    #find max values for parameters of interest (note: need to add minimum values as well)
    max_link_force = temp_batch_np[:,5].max()
    max_link_force_compressive = temp_batch_np[:,5].min() #negative values are compressive, used to check for buckling
    max_crank_torque = abs(temp_batch_np[:,6]).max() #absolute value since motor sizing depends on max power regardless of torque direction
    max_v_s = temp_batch_np[:,8].max()
    max_a_s = temp_batch_np[:,9].max()
    peak_power = max_crank_torque*np.pi/30 #Watts
    max_allowable_stress = 30*10**6
    elastic_modulus = 69*10**9
    link_width_normal = link_sizing(max_link_force, max_allowable_stress)
    link_width_buckle = link_buckling(max_link_force_compressive, elastic_modulus, link, safety_factor)
    link_width = max(link_width_buckle, link_width_normal)
    area = mechanism_xy_area(crank, link, offset)

    peak_values = [index, crank, link, offset, max_link_force, max_crank_torque, max_v_s, max_a_s, peak_power, link_width, area]
    
    kinematic_data.extend(temp_batch_np.tolist())
    peak_data.append(peak_values)

final_kinematic_output = pd.DataFrame(kinematic_data, columns= ["Crank Radius",
                                                                 "Link Length",
                                                                 "Offset",
                                                                 "Theta Angle",
                                                                 "Phi Angle",
                                                                 "Link Force",
                                                                 "Crank Torque",
                                                                 "Xs",
                                                                 "Vs",
                                                                 "As"]) #format final dataframe

final_peak_output = pd.DataFrame(peak_data, columns= ["Index",
                                                        "Crank Radius",
                                                        "Link Length",
                                                        "Offset",
                                                        "Max Force",
                                                        "Max Torque",
                                                        "Max Velocity",
                                                        "Max Acceleration",
                                                        "Peak Power",
                                                        "Min Link Width - Normal Stress or Buckling",
                                                        "Cross-Sectional Area"
                                                        ]) #format final dataframe


print(final_kinematic_output)

print(final_peak_output)

final_kinematic_output.to_csv("kinematics_dynamics.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet

final_peak_output.to_csv("peak_values.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet

end_time = time.time()
execution_time = end_time-start_time
print('Program execution time was ' + f'{execution_time:.2f}' + ' seconds')
