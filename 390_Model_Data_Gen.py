import pandas as pd
import numpy as np
import time
from itertools import product, islice

start_time = time.time()

#np.arange creates an array of numbers with specified step size
#A few ways to approach the input data generation portion: could go fully random within range & then trim
# or alternatively, do stricter compatibilty check before, then generate inputs within known useful range


crank_length = np.arange(100,250,0.1, dtype='f8')

link_length = np.arange(100,250,0.1, dtype='f8') #using dtype='f8', a.k.a fp64 in order to maintain max precision. f4 would probably work too, maybe even f2

offset = np.arange(0.1,20,0.1, dtype='f8')

#product() combines the above arrays into all possible permutations. Need to chunk the permutations for filtering, otherwise run out of ram
all_permutations = product(crank_length, link_length, offset)

filtered_permutations = [] #permutations which pass all checks end up in here

batch_size = 1000000
while True:

    permutations_batch = list(islice(all_permutations, batch_size))
    if not permutations_batch:
        break

    batch_df = pd.DataFrame(permutations_batch, columns=["Crank","Link","Offset"]) #create dataframe of all data in batch

        #Create a new column based on checking for Grashof's mobility condition r + e <= L
    batch_df["Grashof"] = np.where(
        batch_df["Crank"] + batch_df["Offset"] <= batch_df["Link"],
        "True",
        "False"
    )

    batch_df = (batch_df[batch_df.Grashof == "True"]) # filter dataframe for Grashof Condition


    #Stroke length calculation = sqrt[(r+L)^2-e^2] - sqrt[(r-L)^2-e^2]
    #Calculating the first and second terms in the square roots
    stroke_condition_term_1 = (batch_df["Crank"]+batch_df["Link"])**2-batch_df["Offset"]**2

    stroke_condition_term_2 = (batch_df["Crank"]-batch_df["Link"])**2-batch_df["Offset"]**2

    #Checking for which values the terms in the square root would actually be positive (i.e physically possible)
    stroke_condition_positive = (stroke_condition_term_1 >= 0) & (stroke_condition_term_2 >= 0)

    #Calculating square roots avoiding error by rounding negatives to zero
    stroke_value = np.sqrt(stroke_condition_term_1.clip(lower = 0))-np.sqrt(stroke_condition_term_2.clip(lower = 0))

    #find stroke distance, then discard all that aren't within 0.1mm of target distance
    batch_df["Stroke"] = np.where(
        stroke_condition_positive & (stroke_value >= 249.9) & (stroke_value <= 250.1),
        "True",
        "False"
    )

    batch_df = (batch_df[batch_df.Stroke == "True"]) #filter dataframe for Stroke condition
    
    #Calculate return ratio
    batch_df["Alpha"] = (np.asin(batch_df["Offset"]/(batch_df["Link"]-batch_df["Crank"])) + np.asin(batch_df["Offset"]/(batch_df["Link"] + batch_df["Crank"])))

    batch_df["Return_Ratio"] = ((np.pi+batch_df["Alpha"])/(np.pi-batch_df["Alpha"]))

    batch_df = (batch_df[batch_df.Return_Ratio >= 1.5]) #filter dataframe for Return ratio condition (greater than 1.5)


    filtered_permutations.append(batch_df) #add filtered rows to the output list
    



filtered_parameters_df = pd.concat(filtered_permutations, ignore_index=True)

print(filtered_parameters_df)

#Save the dataframe to a csv 
filtered_parameters_df.to_csv("parameters.csv", index=False, float_format='%.4f') 

end_time = time.time()

execution_time = end_time-start_time

print('Program Execution time was ' + str(execution_time/60) + ' minutes')