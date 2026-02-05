import pandas as pd
import numpy as np
import time

start_time = time.time()

#np.arange creates an array of numbers with specified step size
#A few ways to approach the input data generation portion: could go fully random within range & then trim (which is what this code does currently)
#or alternatively, do stricter compatibilty check before, then generate inputs within known useful range (would probably be more efficient)

crank_length = np.arange(100,250,0.5, dtype='f8') #Range of all possible crank lengths, syntax is (min,max,step)

link_length = np.arange(100,250,0.5, dtype='f8') #using dtype='f8', a.k.a fp64 in order to maintain max precision. f4 might be faster

offset = np.arange(0.1,30,0.5, dtype='f8') #Currently using step size of 0.5mm for all lengths, could be adjusted up or down based on machining tolerance & whatnot

crank_grid, offset_grid = np.meshgrid(crank_length, offset)

filtered_parameters = []

for link in link_length:
#Grashof condition check
    grashof_mask = (crank_grid + offset_grid <= link)

    if not np.any(grashof_mask): #checks to make sure that at least one value meets grashof condition
        continue

    g_crank_grid = crank_grid[grashof_mask]

    g_offset_grid = offset_grid[grashof_mask]
#-------------

#Stroke distance conditoin check
    stroke_condition_term_1 = (g_crank_grid + link)**2 - g_offset_grid**2

    stroke_condition_term_2 = (g_crank_grid - link)**2 - g_offset_grid**2

    #Checking for which values the terms in the square root would actually be positive (i.e physically possible)
    stroke_condition_positive_mask = (stroke_condition_term_1 >= 0) & (stroke_condition_term_2 >= 0)

    g_s_crank_grid = g_crank_grid[stroke_condition_positive_mask] #mask the grids to match dimensions of stroke_mask for further masking. Need to clean up names a bit

    g_s_offset_grid = g_offset_grid[stroke_condition_positive_mask]

    #Calculating square roots avoiding error by masking any negative values
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

    return_ratio_grid = return_ratio[return_ratio_mask]

    g_s_s_r_crank_grid = g_s_s_crank_grid[return_ratio_mask]

    g_s_s_r_offset_grid = g_s_s_offset_grid[return_ratio_mask]

    batch_df = pd.DataFrame({
        "Crank": g_s_s_r_crank_grid,
        "Link": link,
        "Offset": g_s_s_r_offset_grid,
        "Return Ratio": return_ratio_grid
             
    }) #create dataframe of all data in batch

    filtered_parameters.append(batch_df)

final_param_output = pd.concat(filtered_parameters, ignore_index=True)

print(final_param_output)

#Save the dataframe to a csv 

final_param_output.to_csv("parameters.csv", index=False, float_format='%.4f') 

end_time = time.time()

execution_time = end_time-start_time

print('Program Execution time was ' + str(execution_time) + ' seconds')