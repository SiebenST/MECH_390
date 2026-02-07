import pandas as pd
import numpy as np
import time

delta_t = (1/12) #(np.pi/12)/(30*2*np.pi/60) defining as global variable, unit is seconds, (Angle Step Size)/(Rad/s) = s

#----- Function definitions -----

#Calculates the position of the slider relative to the base of the crank for given set of input parameters
def slider_displacement(crank_radius, link_length, offset, crank_angle):
    Xs = np.sqrt(link_length**2-(offset+crank_radius*np.sin(crank_angle)))+crank_radius*np.cos(crank_angle)
    return Xs

#Calculates the velocity of the slider relative to the base of the crank for given set of input parameters
def slider_velocity(crank_radius, link_length, offset, crank_angle):
    position_before = slider_displacement(crank_radius, link_length, offset, crank_angle - np.pi/12)
    position_after = slider_displacement(crank_radius, link_length, offset, crank_angle + np.pi/12)
    Vs = position_after-position_before/(2*delta_t)
    return Vs

#Calculates the acceleration of the slider relative to the base of the crank for given set of input parameters
def slider_acceleration(crank_radius, link_length, offset, crank_angle):
    velocity_before = slider_velocity(crank_radius, link_length, offset, crank_angle - np.pi/12)
    velocity_now = slider_velocity(crank_radius, link_length, offset, crank_angle)
    velocity_after = slider_velocity(crank_radius, link_length, offset, crank_angle +  np.pi/12)
    As = (velocity_after-2*velocity_now+velocity_before)/delta_t
    return As

#Calculates total cross-sectional area occupied by the slider crank mechanism in xy plane, disregarding slider size (currently)
def cross_sectional_area(crank_radius, link_length, offset):
    area = 2*crank_radius*(crank_radius+np.sqrt((crank_radius+link_length)**2-offset**2))
    return area


start_time = time.time()

#np.arange creates an array of numbers with specified step size
#Currently just creating all possible combinations, then doing a sweep and filtering out parameters which don't meet criteria

crank_length = np.arange(10,300,0.5, dtype='f8') #Range of all possible crank lengths to check, syntax is (min,max,step)

link_length = np.arange(10,300,0.5, dtype='f8') #using dtype='f8', a.k.a fp64 in order to maintain max precision. f4 (fp32) might be faster

offset = np.arange(0.5,50,0.5, dtype='f8') #Using step size of 0.5mm for all lengths, could be adjusted up/down based on machining tolerance & whatnot

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

    #Checking for which values the terms in the square root would actually be positive (i.e physically possible)
    stroke_condition_positive_mask = (stroke_condition_term_1 >= 0) & (stroke_condition_term_2 >= 0)

    g_s_crank_grid = g_crank_grid[stroke_condition_positive_mask] #mask the grids to match dimensions of stroke_mask for further masking

    g_s_offset_grid = g_offset_grid[stroke_condition_positive_mask]

    #Calculating square roots avoiding program error by masking out any negative values
    stroke = np.sqrt(stroke_condition_term_1[stroke_condition_positive_mask])-np.sqrt(stroke_condition_term_2[stroke_condition_positive_mask])

    stroke_mask = (stroke >= 249.9) & (stroke <= 250.1)

    if not np.any(stroke_mask): #check to ensure at least one valid stroke condition
        continue

    g_s_s_crank_grid = g_s_crank_grid[stroke_mask]

    g_s_s_offset_grid = g_s_offset_grid[stroke_mask]
#--------------

#Calculate return ratio
    alpha_angle = (np.asin(g_s_s_offset_grid/(link-g_s_s_crank_grid)) + np.asin(g_s_s_offset_grid/(link + g_s_s_crank_grid)))

    return_ratio = ((np.pi+alpha_angle)/(np.pi-alpha_angle))

    return_ratio_mask = return_ratio >= 1.5

    if not np.any(return_ratio_mask): #check to ensure at least one valid return_ratio condition, otherwise skip to next link
        continue

    return_ratio_grid = return_ratio[return_ratio_mask]

    g_s_s_r_crank_grid = g_s_s_crank_grid[return_ratio_mask]

    g_s_s_r_offset_grid = g_s_s_offset_grid[return_ratio_mask]
#---------------

#Calculate cross-sectional area of crank-slider mechanism
    xy_area = cross_sectional_area(g_s_s_r_crank_grid,link,g_s_s_r_offset_grid) #mm^2

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

final_param_output.to_csv("parameters.csv", index=False, float_format='%.3f') #exports data to a .csv spreadsheet

#Kinematics Analysis
angle_inputs_deg = np.arange(15,360+15,15, dtype='f8')

angle_inputs_rad = np.deg2rad(angle_inputs_deg)

kinematic_data = []

parameters_matrix = final_param_output[["Crank","Link","Offset"]].to_numpy()

#Actually should do this in the opposite direction (e.g loop through link lengths say, and have the angle be one of the matrices. Might be another np.meshgrid situation)
for row in parameters_matrix:
    for angle in angle_inputs_rad:
        kinematic_batch =[
            row[0], #crank
            row[1], #link
            row[2], #offset
            np.rad2deg(angle),
            slider_displacement(row[0], row[1], row[2], angle), #Xs
            slider_velocity(row[0], row[1], row[2], angle), #Vs
            slider_acceleration(row[0], row[1], row[2], angle) #As
            ]
        kinematic_data.append(kinematic_batch)
    kinematic_data.append("")

final_kinematic_output = pd.DataFrame(kinematic_data, columns= ["Crank Radius", "Link Length", "Offset","Angle", "Xs", "Vs", "As"]) #format final dataframe

print(final_kinematic_output)

final_kinematic_output.to_csv("kinematics.csv", index=False, float_format='%.3f') #exports data to a .csv spreadsheet

end_time = time.time()

execution_time = end_time-start_time

print('Program Execution time was ' + str(execution_time) + ' seconds')