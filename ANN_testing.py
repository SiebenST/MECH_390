import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

np.random.seed(0) #set seeds so that results are reproducible run to run
torch.manual_seed(0)

doe_dataset = pd.read_csv("peak_values.csv")

training_input_scaler = sklearn.preprocessing.MinMaxScaler() #for normalization of data
validation_input_scaler = sklearn.preprocessing.MinMaxScaler() 
training_output_scaler = sklearn.preprocessing.MinMaxScaler()
validation_output_scaler = sklearn.preprocessing.MinMaxScaler()

#input values
crank, link, offset = doe_dataset["Crank Radius"].to_numpy(), doe_dataset["Link Length"].to_numpy(), doe_dataset["Offset"].to_numpy()
input_dataset = np.column_stack(np.float32((crank, link, offset)))
training_inputs, validation_inputs = sklearn.model_selection.train_test_split(input_dataset, test_size=0.3, train_size=0.7, random_state=None, shuffle=True)
training_inputs_norm = training_input_scaler.fit_transform(training_inputs)
validation_inputs_norm = validation_input_scaler.fit_transform(validation_inputs)

#output values
peak_power, area, link_width = doe_dataset["Peak Power"].to_numpy(), doe_dataset["Cross-Sectional Area"].to_numpy(), doe_dataset["Min Link Width - Normal Stress or Buckling"].to_numpy()
output_dataset = np.column_stack(np.float32((peak_power, area, link_width)))
training_outputs, validation_outputs = sklearn.model_selection.train_test_split(output_dataset, test_size=0.3, train_size=0.7, random_state=None, shuffle=True) #split the dataset into two randomized arrays for training & validation
training_outputs_norm = training_output_scaler.fit_transform(training_outputs)
validation_outputs_norm = validation_output_scaler.fit_transform(validation_outputs)

def test(validation, prediction):
    rmse = sklearn.metrics.root_mean_squared_error(validation, prediction)
    r_squared = sklearn.metrics.r2_score(validation, prediction)
    return rmse, r_squared

error_vs_epoch_count = []

def train_target_error(input_data, output_target, optimizer, loss, mse_target, ann_model, input_val_data, output_val_data):
    inputs = torch.from_numpy(np.float32(input_data))
    targets = torch.from_numpy(np.float32(output_target))
    prediction_dataset = torch.from_numpy(input_val_data)
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

        output_prediction = ann_model(prediction_dataset).detach().numpy()

        rmse, r_squared = test(output_val_data, output_prediction)

        print('Epochs: ' + str(epochs) + ' Error: ' + str(error) + ' // RMSE: ' + str(rmse) + ' --- R^2: ' + str(r_squared))

        error_vs_epoch_count.append(error.detach().numpy())

        if epochs >= max_epochs: #check to see if the training has taken more time than
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

ANN_model = ANN().to('cpu')

print(ANN_model)

#------Model training--------
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ANN_model.parameters(), lr = 0.001) #parameters, learning rate
target_error = 0.01
train_target_error(training_inputs_norm, training_outputs_norm, optimizer, loss, target_error, ANN_model, validation_inputs_norm, validation_outputs_norm)
ANN_model.eval()
model_outputs_norm = ANN_model(torch.from_numpy(validation_inputs_norm)).detach().numpy()


validation_inputs_denorm = validation_input_scaler.inverse_transform(validation_inputs_norm)

model_outputs_denorm = training_output_scaler.inverse_transform(model_outputs_norm)

results_table = np.column_stack([validation_inputs_denorm, model_outputs_denorm])

results_table_df = pd.DataFrame(results_table, columns= ["Crank Radius", "Link Length", "Offset", "Peak Power", "Min Link Width", "Cross-Sectional Area"])
sorted_results_df = results_table_df.sort_values(by = "Crank Radius", ascending=True)
print(sorted_results_df)


plot_error_vs_epoch_count = plt.figure()
ax = plt.axes()
ax.plot(error_vs_epoch_count)
ax.set_xlabel('Epoch Count')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Error vs Training Epochs')
ax.grid(True)

plt.show()