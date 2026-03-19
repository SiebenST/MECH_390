import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0) #set seeds so that results are reproducible run to run
torch.manual_seed(0)

doe_dataset = pd.read_csv("peak_values.csv")

input_scaler = sklearn.preprocessing.MinMaxScaler() #for normalization of data
output_scaler = sklearn.preprocessing.MinMaxScaler()


#input values
crank, link, offset = doe_dataset["Crank Radius"].to_numpy(), doe_dataset["Link Length"].to_numpy(), doe_dataset["Offset"].to_numpy()
input_dataset = np.column_stack((crank, link, offset)).astype(np.float32)

#output values
peak_power, link_width, area = doe_dataset["Peak Power"].to_numpy(), doe_dataset["Min Link Width - Normal Stress/Buckling"].to_numpy(), doe_dataset["Cross-Sectional Area"].to_numpy()
output_dataset = np.column_stack((peak_power, link_width, area)).astype(np.float32)

#split the dataset into two randomized arrays for training & validation
training_inputs, validation_inputs, training_outputs, validation_outputs = sklearn.model_selection.train_test_split(input_dataset, output_dataset, test_size=0.3, train_size=0.7, random_state=1, shuffle=True)
training_inputs_norm = input_scaler.fit_transform(training_inputs)
validation_inputs_norm = input_scaler.transform(validation_inputs)

training_outputs_norm = output_scaler.fit_transform(training_outputs)
validation_outputs_norm = output_scaler.transform(validation_outputs)

second_training_inputs, final_validation_inputs, second_training_outputs, final_validation_outputs = sklearn.model_selection.train_test_split(validation_inputs, validation_outputs, test_size=2/3, train_size=1/3, random_state=1, shuffle=True)
second_training_inputs_norm = input_scaler.fit_transform(second_training_inputs)
final_validation_inputs_norm = input_scaler.transform(final_validation_inputs)

second_training_outputs_norm = output_scaler.fit_transform(second_training_outputs)
final_validation_outputs_norm = output_scaler.transform(final_validation_outputs)


# validation_data = np.column_stack([validation_inputs, validation_outputs])
# print(validation_data)

def test(validation, prediction):
    rmse = sklearn.metrics.root_mean_squared_error(validation, prediction)
    r_squared = sklearn.metrics.r2_score(validation, prediction)
    return rmse, r_squared


def train_target_loss(input_data, output_target, optimizer, loss, loss_threshold, ann_model, max_epochs):
    error_vs_epoch_count = []
    inputs = torch.from_numpy(np.float32(input_data)).to(device)
    targets = torch.from_numpy(np.float32(output_target)).to(device)
    
    error = 1
    epochs = 0

    ann_model.train()

    while error > loss_threshold and epochs < max_epochs:

        epochs += 1
        outputs = ann_model(inputs)
        error = loss(outputs, targets)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        error_vs_epoch_count.append(error.item())

        if epochs % 1000 == 0:
          print('Epochs: ' + str(epochs) + ' Error: ' + str(error))

    ann_model.eval()
    return error_vs_epoch_count
    

class ANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(3,32), #Number of inputs to layer / Number of outputs from layer
            torch.nn.ReLU(), #outputs zero for any value < 0, outputs input value for values > 0
            torch.nn.Linear(32,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,8), 
            torch.nn.ReLU(), 
            torch.nn.Linear(8,3)
        )

    def forward(self, input): #runs forward pass through network
        pred = self.layer_stack(input)
        return pred

ANN_model = ANN().to(device)

print(ANN_model)

#------Model training--------
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ANN_model.parameters(), lr = 0.001) #parameters, learning rate
loss_threshold = 0.0001

error_vs_epoch_count_initial_training  = train_target_loss(training_inputs_norm, training_outputs_norm, optimizer, loss, loss_threshold, ANN_model, 20000)
ANN_model.eval()
validation_inputs_norm_tensor = torch.tensor(validation_inputs_norm, dtype=torch.float32).to(device) #necessary when using gpu, have to send val data in tensor form over to gpu
with torch.no_grad(): #reduces memory usage when evaluating model, disables gradient calculation
    model_outputs_norm = ANN_model(validation_inputs_norm_tensor).cpu().detach().numpy() #pulling output data from gpu to cpu ram

model_outputs_denorm = output_scaler.inverse_transform(model_outputs_norm)

results_table = np.column_stack([validation_inputs, model_outputs_denorm, validation_outputs])

results_table_df = pd.DataFrame(results_table, columns= ["Crank Radius", "Link Length", "Offset", "Peak Power", "Min Link Width", "Cross-Sectional Area", "Real Power", "Real Link WIdth", "Real Area"])
sorted_results_df = results_table_df.sort_values(by = "Crank Radius", ascending=True)
sorted_results_df.to_csv("Model_Predictions.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet
print(sorted_results_df)

model_rmse, model_r_squared = test(validation_outputs, model_outputs_denorm)
print("---------------------------------")
print('ANN RMSE: ' + str(model_rmse) + ' // ANN R^2: ' + str(model_r_squared))
print("---------------------------------")


#Additional training with additional 10% of validation data

error_vs_epoch_count_final_training = train_target_loss(second_training_inputs_norm, second_training_outputs_norm, optimizer, loss, loss_threshold, ANN_model, 20000)
ANN_model.eval()
val_test_inputs_norm_tensor = torch.tensor(final_validation_inputs_norm, dtype=torch.float32).to(device) #necessary when using gpu, have to send val data in tensor form over to gpu
with torch.no_grad(): 
    model_outputs_norm_val = ANN_model(val_test_inputs_norm_tensor).cpu().detach().numpy() #pulling output data from gpu to cpu ram

model_outputs_denorm_val = output_scaler.inverse_transform(model_outputs_norm_val)

results_table_val = np.column_stack([final_validation_inputs, model_outputs_denorm_val, final_validation_outputs])

results_table_val_df = pd.DataFrame(results_table_val, columns= ["Crank Radius", "Link Length", "Offset", "Peak Power", "Min Link Width", "Cross-Sectional Area", "Real Power", "Real Link Width", "Real Area"])
sorted_results_df = results_table_val_df.sort_values(by = "Crank Radius", ascending=True)
sorted_results_df.to_csv("Model_Predictions_Val.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet
print(sorted_results_df)

model_rmse, model_r_squared = test(final_validation_outputs, model_outputs_denorm_val)
print("---------------------------------")
print('Second Eval: ANN RMSE: ' + str(model_rmse) + ' // ANN R^2: ' + str(model_r_squared))
print("---------------------------------")

#save model weights for later reuse
torch.save(ANN_model.state_dict(), "ANN_Model.pt")


#Plotting training progress

plt.figure(1)
plt.xscale('log')
plt.plot(error_vs_epoch_count_initial_training)
plt.xlabel('Epoch Count')
plt.ylabel('Mean Squared Error')
plt.title('Error vs Training Epochs - Log Scale, Main Training')
plt.grid(True)

plt.figure(2)
plt.xscale('log')
plt.plot(error_vs_epoch_count_final_training)
plt.xlabel('Epoch Count')
plt.ylabel('Mean Squared Error')
plt.title('Error vs Training Epochs - Log Scale, Second Training')
plt.grid(True)

plt.figure(3)
plt.plot(error_vs_epoch_count_initial_training, label = 'initial')
plt.plot(error_vs_epoch_count_final_training, label = 'final')
plt.legend()
plt.xlabel('Epoch Count')
plt.ylabel('Mean Squared Error')
plt.title('Error vs Training Epochs - Main vs Secondary Training')
plt.grid(True)

plt.figure(4)
plt.xscale('log')
plt.plot(error_vs_epoch_count_initial_training, label = 'initial')
plt.plot(error_vs_epoch_count_final_training, label = 'final')
plt.legend()
plt.xlabel('Epoch Count')
plt.ylabel('Mean Squared Error')
plt.title('Error vs Training Epochs - Main vs Secondary Training - Log Scale')
plt.grid(True)

plt.show()