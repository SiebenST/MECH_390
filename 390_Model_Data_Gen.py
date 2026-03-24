import pandas as pd
import numpy as np

#----- Function definitions -----
def slider_displacement(crank_radius, link_length, offset, crank_angle):
    '''Calculates the position of the slider relative to the base of the crank for given set of input parameters'''
    X_s = np.sqrt(link_length**2-(offset+crank_radius*np.sin(crank_angle))**2)+crank_radius*np.cos(crank_angle)
    return X_s

def slider_velocity(crank_radius, link_length, offset, crank_angle, delta_t):
    '''Calculates the velocity of the slider relative to the base of the crank for given set of input parameters'''
    position_before = slider_displacement(crank_radius, link_length, offset, (crank_angle - np.deg2rad(angle_step_size)))
    position_after = slider_displacement(crank_radius, link_length, offset, (crank_angle + np.deg2rad(angle_step_size)))
    V_s = (position_after-position_before)/(2*delta_t)
    return V_s

def slider_acceleration(crank_radius, link_length, offset, crank_angle, delta_t):
    '''Calculates the acceleration of the slider relative to the base of the crank for given set of input parameters'''
    position_before = slider_displacement(crank_radius, link_length, offset, (crank_angle - np.deg2rad(angle_step_size)))
    position_now = slider_displacement(crank_radius, link_length, offset, crank_angle)
    position_after = slider_displacement(crank_radius, link_length, offset, (crank_angle + np.deg2rad(angle_step_size)))
    A_s = (position_after-2*position_now+position_before)/(delta_t**2)
    return A_s

def angle_phi(crank_radius, link_length, offset, crank_angle):
    '''Calculate angle of link relative to horizontal plance'''
    phi = np.asin((crank_radius*np.sin(crank_angle)+offset)/link_length)
    return phi

def return_ratio_calc(crank, link, offset):
    alpha_angle = (np.asin(offset/(link-crank)) - np.asin(offset/(link + crank)))
    q = ((np.pi+alpha_angle)/(np.pi-alpha_angle))
    return q

def mechanism_xy_area(crank_radius, link_length, offset):
    '''Calculates total cross-sectional area occupied by the slider crank mechanism in xy plane, disregarding slider size (currently)'''
    area = 2*crank_radius*(crank_radius+np.sqrt((crank_radius+link_length)**2-offset**2))
    return area

def link_reaction_force(slider_mass, slider_acceleration, slider_velocity, angle_phi, coefficient_mu):
    '''Calculate force going through link. Adds 0.5kg load when slider_velocity > 0 (forward stroke)'''
    # Sign of velocity (+1 or -1) to ensure friction always opposes motion. 
    velocity_sign = np.sign(slider_velocity+1e-9)
    # Add 0.5kg to effective slider mass if pushing forward
    effective_mass = slider_mass + np.where(slider_velocity > 0, 0.5, 0.0)
    link_reaction_force = ((effective_mass * slider_acceleration) + (velocity_sign * coefficient_mu * effective_mass * 9.81))/(np.cos(angle_phi) - (velocity_sign * coefficient_mu * np.sin(angle_phi)))
    return link_reaction_force

def crank_torque(force_magnitude, crank_radius, theta_angle, phi_angle):
    '''Calculate torque on crank'''
    torque = force_magnitude*crank_radius*(np.cos(phi_angle)*np.sin(theta_angle)+np.sin(phi_angle)*np.cos(theta_angle))
    return torque

def link_buckling(link_width, link_depth, aluminum_elastic_modulus, link_length, safety_factor):
    '''Max allowable compressive force for no buckling (Euler condition)'''
    inertia_moment_width = link_depth*link_width**3/12
    inertia_moment_depth = link_width*link_depth**3/12
    min_inertia_moment = min(inertia_moment_width, inertia_moment_depth)
    critical_load = np.pi**2*aluminum_elastic_modulus*min_inertia_moment/link_length**2
    allowable_load = critical_load/safety_factor
    return allowable_load
 
def fatigue_strength(n_cycles):
    '''6061-T6 fatigue strength for given number of cycles, taken from https://www.osti.gov/servlets/purl/10157028'''
    fatigue_strength = (14479/n_cycles**0.5+96.5)*10**6 #Pa
    return fatigue_strength

def crank_bending(crank_torque, crank_width, crank_depth):
    '''Crank Bending Stress'''
    normal_stress = 6*crank_torque / (crank_width*crank_depth**2)
    return normal_stress

#Gerber formula
def equivalent_reversed_stress(stress_amplitude, mean_stress, aluminum_ultimate_tensile_strength):
    '''Gerber Criterion'''
    equivalent_stress = stress_amplitude/(1-(mean_stress/aluminum_ultimate_tensile_strength)**2) #Pa
    return equivalent_stress

def optimization_function(power_input, power_ideal, power_importance, size_input, size_ideal, size_importance, return_ratio, return_ratio_ideal, return_ratio_importance):
    '''Scores design based on weighted factors, perfect score is 1 (100%)'''
    optimization_score = power_ideal/power_input*power_importance + size_ideal/size_input*size_importance + return_ratio/return_ratio_ideal*return_ratio_importance
    return optimization_score

def stroke_distance(crank, link, offset): #may end up unused, only relevant if we decide to expand design sweep in order to train a broader ann model
    '''Returns stroke distance for given geometrical parameters'''
    stroke = np.sqrt((crank + link)**2 - offset**2)-np.sqrt((crank - link)**2 - offset**2)
    return stroke

#Other Properties
omega = np.pi #30rpm * (2pi rad) / rotation * 1min / 60s = pi rad/s
angle_step_size = 1 #[deg]
delta_t = np.deg2rad(angle_step_size)/omega #(1 deg * 2pi rad / 360 deg) / pi rad/s = [s]
cycle_count = 10**8
safety_factor = 1.3
slider_mass = 0.5 #Kg
friction_coefficient_mu = 0.3


#---------Aluminum 6061-T6 Material Properties-------------
aluminum_elastic_modulus = 69*10**9 #Pa
aluminum_tensile_yield_strength = 276*10**6 #Pa
aluminum_ultimate_tensile_strength = 310*10**6 #Pa
aluminum_fatigue_strength = fatigue_strength(cycle_count)
max_allowable_stress = aluminum_fatigue_strength/safety_factor
global_link_width = 5*10**-3 #m, selected based on commonly available bar stock


#np.arange creates an array of numbers with specified step size
#Creates a range of possible combinations, then doing a sweep and filtering out parameters which don't meet criteria
crank_length = np.arange(100,300,0.5, dtype='f8')*10**-3 #Range of all possible crank lengths to check, syntax is (min,max,step)

link_length = np.arange(100,300,0.5, dtype='f8')*10**-3 #using dtype='f8', a.k.a fp64 in order to maintain max precision. f4 (fp32) might be faster

offset = np.arange(0.5,50,0.5, dtype='f8')*10**-3 #Using step size of 0.5mm for all lengths, could be adjusted up/down based on machining tolerance & whatnot

crank_grid, offset_grid = np.meshgrid(crank_length, offset) #meshgrid creates two large matrices. 
#The first input vector is taken as a row which is then copied vertically a number of times equal to the second vector's length. 
#The second input vector is taken as a column, which is then copied horizontally a number of times equal to the first vector's length
#End result is that picking a matrix coordinate and looking at both matrices will give a different combination of the initial input vectors' values

filtered_parameters = [] #Parameters which pass all filter steps end up in here

for link in link_length: #loops through all link length values, combined with above matrices ends up doing full sweep of all permutations
    valid_cranks = crank_grid
    valid_offsets = offset_grid
    #Grashof condition check
    grashof_mask = (valid_cranks + valid_offsets <= link)

    if not np.any(grashof_mask):
        continue

    valid_cranks = valid_cranks[grashof_mask]
    valid_offsets = valid_offsets[grashof_mask]
    
    #Stroke distance condition check
    stroke_condition_term_1 = (valid_cranks + link)**2 - valid_offsets**2

    stroke_condition_term_2 = (valid_cranks - link)**2 - valid_offsets**2

    #Checking for which values the terms in the square root are positive (i.e physically possible)
    stroke_condition_positive_mask = (stroke_condition_term_1 >= 0) & (stroke_condition_term_2 >= 0)

    valid_cranks = valid_cranks[stroke_condition_positive_mask]
    valid_offsets = valid_offsets[stroke_condition_positive_mask]

    #Calculating square roots avoiding program error by masking out any negative values
    stroke = np.sqrt(stroke_condition_term_1[stroke_condition_positive_mask])-np.sqrt(stroke_condition_term_2[stroke_condition_positive_mask])
    
    stroke_mask = (stroke >= 249.5*10**-3) & (stroke <= 250.5*10**-3) #the tolerance here drastically affects the amount of parameter sets which filter through

    if not np.any(stroke_mask):
        continue

    valid_cranks = valid_cranks[stroke_mask]
    valid_offsets = valid_offsets[stroke_mask]

    #Calculate return ratio
    return_ratio = return_ratio_calc(valid_cranks, link, valid_offsets)

    return_ratio_mask = (return_ratio >= 1.5) & (return_ratio <= 2.5) #filters for minium return ratio within specified range. This is left uncapped and any extreme values are filtered out later during stress analysis

    if not np.any(return_ratio_mask):
        continue

    valid_cranks = valid_cranks[return_ratio_mask]
    valid_offsets = valid_offsets[return_ratio_mask]

    #Calculate cross-sectional area of crank-slider mechanism
    xy_area = mechanism_xy_area(valid_cranks, link, valid_offsets)

    #create dataframe of all data in batch
    batch_df = pd.DataFrame({
        "Crank": valid_cranks,
        "Link": link,
        "Offset": valid_offsets,
        "Return Ratio": return_ratio[return_ratio_mask], 
        "Cross Sectional Area": xy_area
    }) 

    filtered_parameters.append(batch_df)

final_param_output = pd.concat(filtered_parameters, ignore_index=True) #format final dataframe

print(final_param_output)

final_param_output.to_csv("parameters.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet

#Kinematics Analysis
angle_inputs_deg = np.arange(0, 360, angle_step_size, dtype='f8') #moved to 1 deg increments, 15 appeared too coarse to be useful
angle_inputs_rad = np.deg2rad(angle_inputs_deg)

pin_sizes = np.arange(0.01,1,0.01, dtype='f8')*10**-3 #range of possible pin sizes, to be checked against loading for max allowable stress, starts at 4 from mcmaster-carr
link_depth_sizes = np.arange(0.01,1,0.01, dtype='f8')*10**-3
crank_depth_sizes = np.arange(0.01,1,0.01, dtype='f8')*10**-3

kinematic_data = [] #all data output
peak_data = [] #peak values of each parameter set + filtering out certain criteria

parameters_matrix = final_param_output[["Crank","Link","Offset"]].to_numpy()

#Goes row by row through parameter combinations previously filtered, then calculates kinematics for each angle

for row in parameters_matrix:

    crank, link, offset = row[0], row[1], row[2]
    
    x_s = slider_displacement(crank, link, offset, angle_inputs_rad) #unit
    v_s = slider_velocity(crank, link, offset, angle_inputs_rad, delta_t) #unit/s
    a_s = slider_acceleration(crank, link, offset, angle_inputs_rad, delta_t) #unit/s^2
    phi = angle_phi(crank, link, offset, angle_inputs_rad) #rad
    link_force = link_reaction_force(slider_mass, a_s, v_s, phi, friction_coefficient_mu) #Newton
    torque = crank_torque(link_force, crank, angle_inputs_rad, phi) #Torque N.m

    #copy crank/link/offset times to fill columns to same length as other variables
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
    
    #find peak values for parameters of interest
    max_link_force = temp_batch_np[:,5].max()
    max_link_force_compressive = temp_batch_np[:,5].min() #negative values are compressive, used to check for buckling

    average_link_force = (max_link_force+max_link_force_compressive)/2
    link_force_amplitude = (max_link_force-max_link_force_compressive)/2

    max_crank_torque = temp_batch_np[:,6].max()
    min_crank_torque = temp_batch_np[:,6].min()

    average_crank_torque = (max_crank_torque+min_crank_torque)/2
    crank_torque_amplitude = (max_crank_torque-min_crank_torque)/2

    valid_pins = []
    
    for diameter in pin_sizes:
        pin_mean_stress = 4*average_link_force/(diameter**2*np.pi)
        pin_stress_amplitude = 4*link_force_amplitude/(diameter**2*np.pi)

        # Fatigue check (Gerber)
        shear_ultimate_strength = aluminum_ultimate_tensile_strength*0.577 #von mises
        pin_equivalent_stress = equivalent_reversed_stress(pin_stress_amplitude, pin_mean_stress, shear_ultimate_strength)
        fatigue_check = pin_equivalent_stress < aluminum_fatigue_strength / safety_factor

        # Static yield check
        shear_yield_strength = aluminum_tensile_yield_strength*0.577
        pin_max_shear_stress = 4 * max_link_force / (np.pi * diameter**2)
        static_check = pin_max_shear_stress < shear_yield_strength / safety_factor

        if fatigue_check == True and static_check == True:
            valid_pins.append(diameter)

    if not np.any(valid_pins):
        continue

    valid_links_depths = []

    for depth in link_depth_sizes:
        link_mean_stress = average_link_force / (depth*global_link_width)
        link_stress_amplitude = link_force_amplitude / (depth*global_link_width)

        # Fatigue check (Gerber)
        link_equivalent_stress = equivalent_reversed_stress(link_stress_amplitude, link_mean_stress, aluminum_ultimate_tensile_strength)
        fatigue_check = link_equivalent_stress < aluminum_fatigue_strength / safety_factor

        # Static yield check
        link_max_normal_stress = max_link_force / (depth*global_link_width)
        static_check = link_max_normal_stress < aluminum_tensile_yield_strength / safety_factor

        if fatigue_check == True and static_check == True and max_link_force_compressive < link_buckling(global_link_width, depth, aluminum_elastic_modulus, link, safety_factor): # Fatigue/Static/Buckling
            valid_links_depths.append(depth)

    valid_links_depths = np.array(valid_links_depths)
    if not np.any(valid_links_depths):
        continue
    
    valid_crank_depths = []

    for depth in crank_depth_sizes:
        crank_mean_stress = crank_bending(average_crank_torque, global_link_width, depth) #don't think this is valid, consider as placeholder for now
        crank_stress_amplitude = crank_bending(crank_torque_amplitude, global_link_width, depth)

        # Fatigue check (Gerber)
        crank_equivalent_stress = equivalent_reversed_stress(crank_stress_amplitude, crank_mean_stress, aluminum_ultimate_tensile_strength)
        fatigue_check = crank_equivalent_stress < aluminum_fatigue_strength / safety_factor

        # Static yield check
        crank_max_normal_stress = crank_bending(max_crank_torque, global_link_width, depth)
        static_check = crank_max_normal_stress < aluminum_tensile_yield_strength / safety_factor

        if fatigue_check == True and static_check == True:
            valid_crank_depths.append(depth)

    valid_crank_depths = np.array(valid_crank_depths)
    if not np.any(valid_crank_depths):
        continue
    
    min_pin_dia = min(valid_pins)

    #valid_links_depths = valid_links_depths[valid_links_depths > 2.25*min_pin_dia] #size check between pin and link to ensure no tearing of link at pin hole. This should maybe be higher to account for bearing sizing?

    #if not np.any(valid_links_depths):
    #    continue

    min_link_depth = min(valid_links_depths)

    min_crank_depth=min(valid_crank_depths)

    max_crank_torque = abs(temp_batch_np[:,6]).max() #absolute value since motor sizing depends on max power regardless of torque direction
    max_v_s = temp_batch_np[:,8].max()
    max_a_s = temp_batch_np[:,9].max()
    peak_power = max_crank_torque*np.pi #P = T*w, [W] -> 30 rotations/minute * 2pi rad/rotation * 1 minute/60s = pi rad/s
    area = mechanism_xy_area(crank, link, offset)
    return_ratio = return_ratio_calc(crank, link, offset)
    stroke_length = stroke_distance(crank, link, offset)
    peak_values = [crank, link, offset, max_link_force, max_crank_torque, max_v_s, max_a_s, peak_power, min_link_depth, min_crank_depth, min_pin_dia, return_ratio, area, stroke_length]
    
    kinematic_data.extend(temp_batch_np.tolist())
    if peak_power < 2:
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

final_peak_output = pd.DataFrame(peak_data, columns= ["Crank Radius",
                                                        "Link Length",
                                                        "Offset",
                                                        "Max Force",
                                                        "Max Torque",
                                                        "Max Velocity",
                                                        "Max Acceleration",
                                                        "Peak Power",
                                                        "Min Link Depth",
                                                        "Min Crank Depth",
                                                        "Min Pin Diameter",
                                                        "Return Ratio",
                                                        "Cross-Sectional Area",
                                                        "Stroke"
                                                        ]) #format final dataframe

minimum_area = final_peak_output["Cross-Sectional Area"].min()
minimum_power = final_peak_output["Peak Power"].min()
max_return_ratio = final_peak_output["Return Ratio"].max()


final_peak_output["Optimization Score"] = optimization_function(final_peak_output["Peak Power"], minimum_power, 0.4, 
                                                                final_peak_output["Cross-Sectional Area"], minimum_area, 0.4, 
                                                                final_peak_output["Return Ratio"], max_return_ratio, 0.2)

final_peak_output_sorted = final_peak_output.sort_values(by = "Optimization Score", ascending=False)

print(final_kinematic_output)

print(final_peak_output_sorted)

final_kinematic_output.to_csv("kinematics_dynamics.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet

final_peak_output_sorted.to_csv("peak_values.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet