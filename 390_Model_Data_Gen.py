import pandas as pd
import numpy as np
import time

start_time = time.time()

delta_t = (1/12) #(np.pi/12)/(30*2*np.pi/60) defining as global variable, unit is seconds, (Angle Step Size)/(Rad/s) = s

#----- Function definitions -----

#Calculates the position of the slider relative to the base of the crank for given set of input parameters
def slider_displacement(crank_radius, link_length, offset, crank_angle):
    X_s = np.sqrt(link_length**2-(offset+crank_radius*np.sin(crank_angle))**2)+crank_radius*np.cos(crank_angle)
    return X_s

#Calculates the velocity of the slider relative to the base of the crank for given set of input parameters
def slider_velocity(crank_radius, link_length, offset, crank_angle):
    position_before = slider_displacement(crank_radius, link_length, offset, (crank_angle - np.pi/12))
    position_after = slider_displacement(crank_radius, link_length, offset, (crank_angle + np.pi/12))
    V_s = (position_after-position_before)/(2*delta_t)
    return V_s

#Calculates the acceleration of the slider relative to the base of the crank for given set of input parameters
def slider_acceleration(crank_radius, link_length, offset, crank_angle):
    velocity_before = slider_velocity(crank_radius, link_length, offset, (crank_angle - np.pi/12))
    velocity_now = slider_velocity(crank_radius, link_length, offset, crank_angle)
    velocity_after = slider_velocity(crank_radius, link_length, offset, (crank_angle +  np.pi/12))
    A_s = (velocity_after-2*velocity_now+velocity_before)/delta_t
    return A_s

#Calculates total cross-sectional area occupied by the slider crank mechanism in xy plane, disregarding slider size (currently)
def mechanism_xy_area(crank_radius, link_length, offset):
    area = 2*crank_radius*(crank_radius+np.sqrt((crank_radius+link_length)**2-offset**2))
    return area

#Calculate normal stress in link
def normal_stress(normal_force, link_width, link_height):
    stress = normal_force/(link_width*link_height)
    return stress

#Calculate angle of link relative to horizontal plance
def angle_phi(crank_radius, link_length, offset, crank_angle):
    phi = np.asin((crank_radius*np.sin(crank_angle)+offset)/link_length)
    return phi

#Calculate shear stress in pin
def pin_shear_stress(shear_force, pin_radius):
    stress = shear_force/(np.pi*pin_radius**2) 
    return stress

#Calculate normal stress in link
def link_reaction_force(slider_mass, slider_acceleration, angle_phi, coefficient_mu):
    slider_normal_force = (slider_mass*slider_acceleration+slider_mass*9.81*1/np.tan(angle_phi))/(1/np.tan(angle_phi)-coefficient_mu)
    link_reaction_force = (slider_normal_force-slider_mass*9.81)/np.sin(angle_phi)
    return link_reaction_force

#Calculate torque on crank
def crank_torque(force_magnitude, crank_radius, theta_angle, phi_angle):
    torque = force_magnitude*crank_radius*(np.cos(phi_angle)*np.sin(theta_angle)+np.sin(phi_angle)*np.cos(theta_angle))
    return torque
#--------------

#np.arange creates an array of numbers with specified step size
#Currently just creating all possible combinations, then doing a sweep and filtering out parameters which don't meet criteria

crank_length = np.arange(10,300,0.5, dtype='f8')*10**-3 #Range of all possible crank lengths to check, syntax is (min,max,step)

link_length = np.arange(10,300,0.5, dtype='f8')*10**-3 #using dtype='f8', a.k.a fp64 in order to maintain max precision. f4 (fp32) might be faster

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

    stroke_mask = (stroke >= 249.99*10**-3) & (stroke <= 250.01*10**-3)

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
    xy_area = mechanism_xy_area(g_s_s_r_crank_grid,link,g_s_s_r_offset_grid) #mm^2
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
angle_inputs_deg = np.arange(15,360+15,15, dtype='f8')

angle_inputs_rad = np.deg2rad(angle_inputs_deg)

kinematic_data = []

parameters_matrix = final_param_output[["Crank","Link","Offset"]].to_numpy()

#Goes row by row through parameter combinations previously filtered, then calculates kinematics for each angle
for row in parameters_matrix:
    for angle in angle_inputs_rad:
        crank = row[0]
        link = row[1]
        offset = row[2]
        x_s = slider_displacement(crank, link, offset, angle) #unit
        v_s = slider_velocity(crank, link, offset, angle) #unit/s
        a_s = slider_acceleration(crank, link, offset, angle) #unit/s^2
        phi = angle_phi(crank, link, offset, angle) #rad
        kinematic_batch =[
            crank, #crank
            link, #link
            offset, #offset
            np.rad2deg(angle), #theta, degrees
            np.rad2deg(phi), #phi, degrees
            normal_stress(link_reaction_force(0.5, a_s, phi, 0.1), 5, 5), #N, mm, mm --> MPa
            crank_torque(link_reaction_force(0.5, a_s, phi, 0.1), crank, angle, phi),
            x_s, #Xs
            v_s, #Vs
            a_s #As
            ]
        kinematic_data.append(kinematic_batch)

final_kinematic_output = pd.DataFrame(kinematic_data, columns= ["Crank Radius",
                                                                 "Link Length",
                                                                 "Offset",
                                                                 "Theta Angle",
                                                                 "Phi Angle",
                                                                 "Link Normal Stress",
                                                                 "Torque",
                                                                 "Xs",
                                                                 "Vs",
                                                                 "As"]) #format final dataframe

print(final_kinematic_output)

final_kinematic_output.to_csv("kinematics_dynamics.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet

end_time = time.time()

execution_time = end_time-start_time

print('Program Execution time was ' + str(execution_time) + ' seconds')