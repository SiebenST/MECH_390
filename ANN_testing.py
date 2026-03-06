import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #set this to check for available cuda device (desktop has a gpu, laptop does not)

np.random.seed(0) #set seeds so that results are reproducible run to run
torch.manual_seed(0)

doe_dataset = pd.read_csv("peak_values.csv")

input_scaler = sklearn.preprocessing.MinMaxScaler() #for normalization of data
output_scaler = sklearn.preprocessing.MinMaxScaler()


#input values
crank, link, offset = doe_dataset["Crank Radius"].to_numpy(), doe_dataset["Link Length"].to_numpy(), doe_dataset["Offset"].to_numpy()
input_dataset = np.column_stack(np.float32((crank, link, offset)))

#output values
peak_power, link_width, area = doe_dataset["Peak Power"].to_numpy(), doe_dataset["Min Link Width - Normal Stress or Buckling"].to_numpy(), doe_dataset["Cross-Sectional Area"].to_numpy()
output_dataset = np.column_stack(np.float32((peak_power, link_width, area)))

#split the dataset into two randomized arrays for training & validation
training_inputs, validation_inputs, training_outputs, validation_outputs = sklearn.model_selection.train_test_split(input_dataset, output_dataset, test_size=0.3, train_size=0.7, random_state=1, shuffle=True)
training_inputs_norm = input_scaler.fit_transform(training_inputs)
validation_inputs_norm = input_scaler.fit_transform(validation_inputs)

training_outputs_norm = output_scaler.fit_transform(training_outputs)
validation_outputs_norm = output_scaler.fit_transform(validation_outputs)

# validation_data = np.column_stack([validation_inputs, validation_outputs])
# print(validation_data)

def test(validation, prediction):
    rmse = sklearn.metrics.root_mean_squared_error(validation, prediction)
    r_squared = sklearn.metrics.r2_score(validation, prediction)
    return rmse, r_squared

error_vs_epoch_count = []

def train_target_error(input_data, output_target, optimizer, loss, mse_target, ann_model, input_val_data, output_val_data):
    inputs = torch.from_numpy(np.float32(input_data)).to(device)
    targets = torch.from_numpy(np.float32(output_target)).to(device)
    
    error = 100
    epochs = 0
    max_epochs = 50000

    while error > mse_target:
        ann_model.train()

        epochs += 1

        outputs = ann_model(inputs)

        error = loss(outputs, targets)

        optimizer.zero_grad()

        error.backward()

        optimizer.step()

        ann_model.eval()

        print('Epochs: ' + str(epochs) + ' Error: ' + str(error))

        error_vs_epoch_count.append(error.item())

        if epochs >= max_epochs: #check to see if the training has taken more than allowable epochs
            break
    

class ANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(3,32), #Number of inputs to layer / Number of outputs from layer
            torch.nn.ReLU(), #outputs zero for any value < 0, outputs input value for values > 0
            torch.nn.BatchNorm1d(32),
            torch.nn.Linear(32,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,8), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(8),
            torch.nn.Linear(8,3)
        )

    def forward(self, input): #runs forward pass through network
        pred = self.layer_stack(input)
        return pred

ANN_model = ANN().to(device)

print(ANN_model)

#------Model training--------
loss = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(ANN_model.parameters(), lr = 0.001) #parameters, learning rate
target_error = 0.00001
train_target_error(training_inputs_norm, training_outputs_norm, optimizer, loss, target_error, ANN_model, validation_inputs_norm, validation_outputs_norm)
ANN_model.eval()
validation_inputs_norm_tensor = torch.tensor(validation_inputs_norm, dtype=torch.float32).to(device) #necessary when using gpu, have to send val data in tensor form over to gpu
model_outputs_norm = ANN_model(validation_inputs_norm_tensor).cpu().detach().numpy() #pulling output data from gpu to cpu ram

validation_inputs_denorm = input_scaler.inverse_transform(validation_inputs_norm)

model_outputs_denorm = output_scaler.inverse_transform(model_outputs_norm)

results_table = np.column_stack([validation_inputs_denorm, model_outputs_denorm])

results_table_df = pd.DataFrame(results_table, columns= ["Crank Radius", "Link Length", "Offset", "Peak Power", "Min Link Width", "Cross-Sectional Area"])
sorted_results_df = results_table_df.sort_values(by = "Crank Radius", ascending=True)
sorted_results_df.to_csv("Model_Predictions.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet
print(sorted_results_df)

model_accuracy = test(validation_outputs, model_outputs_denorm)
print(model_accuracy)

plot_error_vs_epoch_count = plt.figure()
ax = plt.axes()
ax.set_xscale('log')
ax.plot(error_vs_epoch_count)
ax.set_xlabel('Epoch Count')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Error vs Training Epochs')
ax.grid(True)

plt.show()