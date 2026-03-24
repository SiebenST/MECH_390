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

def link_buckling(link_force, aluminum_elastic_modulus, link_length, safety_factor):
    '''Min link width for no buckling (Euler condition)'''
    max_allowable_load = abs(link_force*safety_factor)
    min_inertia_moment = max_allowable_load*link_length**2/(np.pi**2*aluminum_elastic_modulus)
    min_link_width = (min_inertia_moment*12)**0.25
    return min_link_width
 
def fatigue_strength(n_cycles):
    '''6061-T6 fatigue strength for given number of cycles, taken from https://www.osti.gov/servlets/purl/10157028'''
    fatigue_strength = (14479/n_cycles**0.5+96.5)*10**6 #Pa
    return fatigue_strength

def crank_bending(crank_torque, crank_width):
    '''Crank Bending Stress'''
    normal_stress = (6*crank_torque/crank_width**3)
    return normal_stress

#Gerber formula
def equivalent_reversed_stress(stress_amplitude, mean_stress, aluminum_ultimate_tensile_strength):
    '''Gerber Criterion'''
    equivalent_stress = stress_amplitude/(1-(mean_stress/aluminum_ultimate_tensile_strength)**2) #Pa
    return equivalent_stress


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

pin_sizes = np.arange(2.0,15,0.5, dtype='f8')*10**-3 #range of possible pin sizes, to be checked against loading for max allowable stress
link_width_sizes = np.arange(2.0,30,0.5, dtype='f8')*10**-3
crank_width_sizes = np.arange(5.0,30,0.5, dtype='f8')*10**-3

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

    valid_links = []

    for width in link_width_sizes:
        link_mean_stress = average_link_force/width**2
        link_stress_amplitude = link_force_amplitude/width**2
        # Fatigue check (Gerber)
        link_equivalent_stress = equivalent_reversed_stress(link_stress_amplitude, link_mean_stress, aluminum_ultimate_tensile_strength)
        fatigue_check = link_equivalent_stress < aluminum_fatigue_strength / safety_factor

        # Static yield check
        link_max_normal_stress = max_link_force / width**2
        static_check = link_max_normal_stress < aluminum_tensile_yield_strength / safety_factor

        if fatigue_check == True and static_check == True and width > link_buckling(max_link_force_compressive, aluminum_elastic_modulus, link, safety_factor): # Fatigue/Static/Buckling
            valid_links.append(width)

    valid_links = np.array(valid_links)
    if not np.any(valid_links):
        continue
    
    valid_crank_widths = []

    for width in crank_width_sizes:
        crank_mean_stress = crank_bending(average_crank_torque, width) #don't think this is valid, consider as placeholder for now
        crank_stress_amplitude = crank_bending(crank_torque_amplitude, width)

        # Fatigue check (Gerber)
        crank_equivalent_stress = equivalent_reversed_stress(crank_stress_amplitude, crank_mean_stress, aluminum_ultimate_tensile_strength)
        fatigue_check = crank_equivalent_stress < aluminum_fatigue_strength / safety_factor

        # Static yield check
        crank_max_normal_stress = crank_bending(max_crank_torque, width)
        static_check = crank_max_normal_stress < aluminum_tensile_yield_strength / safety_factor

        if fatigue_check == True and static_check == True:
            valid_crank_widths.append(width)

    valid_crank_widths = np.array(valid_crank_widths)
    if not np.any(valid_crank_widths):
        continue
    
    min_pin_dia = min(valid_pins)

    valid_links = valid_links[valid_links > 2.25*min_pin_dia] #size check between pin and link to ensure no tearing of link at pin hole. This should maybe be higher to account for bearing sizing?

    if not np.any(valid_links):
        continue

    min_link_width = min(valid_links)

    min_crank_width=min(valid_crank_widths)

    max_crank_torque = abs(temp_batch_np[:,6]).max() #absolute value since motor sizing depends on max power regardless of torque direction
    max_v_s = temp_batch_np[:,8].max()
    max_a_s = temp_batch_np[:,9].max()
    peak_power = max_crank_torque*np.pi #P = T*w, [W] -> 30 rotations/minute * 2pi rad/rotation * 1 minute/60s = pi rad/s
    area = mechanism_xy_area(crank, link, offset)
    return_ratio = return_ratio_calc(crank, link, offset)

    peak_values = [crank, link, offset, max_link_force, max_crank_torque, max_v_s, max_a_s, peak_power, min_link_width, min_crank_width, min_pin_dia, return_ratio, area]
    
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
                                                        "Min Link Width",
                                                        "Min Crank Width",
                                                        "Min Pin Diameter",
                                                        "Return Ratio",
                                                        "Cross-Sectional Area"
                                                        ]) #format final dataframe

print(final_kinematic_output)

print(final_peak_output)

final_kinematic_output.to_csv("kinematics_dynamics.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet

final_peak_output.to_csv("peak_values.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet