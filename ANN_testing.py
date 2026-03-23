import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0) #set seeds so that results are reproducible run to run
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

doe_dataset = pd.read_csv("peak_values.csv")

input_scaler = sklearn.preprocessing.MinMaxScaler() #for normalization of data
output_scaler = sklearn.preprocessing.MinMaxScaler()

#input values
crank, link, offset = doe_dataset["Crank Radius"].to_numpy(), doe_dataset["Link Length"].to_numpy(), doe_dataset["Offset"].to_numpy()
input_dataset = np.column_stack((crank, link, offset)).astype(np.float32)

#output values
peak_power, link_width, area = doe_dataset["Peak Power"].to_numpy(), doe_dataset["Min Link Width"].to_numpy(), doe_dataset["Cross-Sectional Area"].to_numpy()
output_dataset = np.column_stack((peak_power, link_width, area)).astype(np.float32)

#split the dataset into two randomized arrays for training & validation
training_inputs, temp_inputs, training_outputs, temp_outputs = sklearn.model_selection.train_test_split(input_dataset, output_dataset, test_size=0.3, random_state=1, shuffle=True)
validation_inputs, testing_inputs, validation_outputs, testing_outputs = sklearn.model_selection.train_test_split(temp_inputs, temp_outputs, test_size=0.5, random_state=1, shuffle=True)

training_inputs_norm = input_scaler.fit_transform(training_inputs)
validation_inputs_norm = input_scaler.transform(validation_inputs)
testing_inputs_norm = input_scaler.transform(testing_inputs)

training_outputs_norm = output_scaler.fit_transform(training_outputs)
validation_outputs_norm = output_scaler.transform(validation_outputs)

def test(validation, prediction):
    rmse = sklearn.metrics.root_mean_squared_error(validation, prediction)
    r_squared = sklearn.metrics.r2_score(validation, prediction)
    return rmse, r_squared


def train_target_loss(input_data, output_target, optimizer, loss, loss_threshold, ann_model, max_epochs, validation_inputs, validation_targets):
    training_error_vs_epoch = []
    validation_error_vs_epoch = []
    inputs = torch.from_numpy(np.float32(input_data)).to(device)
    targets = torch.from_numpy(np.float32(output_target)).to(device)
    
    training_error = 1
    epochs = 0

    ann_model.train()

    while training_error > loss_threshold and epochs < max_epochs:

        epochs += 1
        outputs = ann_model(inputs)
        training_error = loss(outputs, targets)
        optimizer.zero_grad()
        training_error.backward()
        optimizer.step()
        training_error_vs_epoch.append([epochs, training_error.item()])

        if epochs % 10 == 0: #periodic validation
            ann_model.eval()
            with torch.no_grad():
                validation_error = loss(ann_model(torch.tensor(validation_inputs, dtype=torch.float32).to(device)), torch.tensor(validation_targets, dtype=torch.float32).to(device))
            validation_error_vs_epoch.append([epochs, validation_error.item()])
            ann_model.train()

        if epochs % 10 == 0:
          print('Epochs: ' + str(epochs) + ' Training Error: ' + str(training_error) + ' Validation Error' + str(validation_error))

    ann_model.eval()
    return training_error_vs_epoch, validation_error_vs_epoch
    

class ANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(3,32), #Number of inputs to layer / Number of outputs from layer
            torch.nn.ReLU(), #outputs zero for any value < 0, outputs input value for values > 0
            torch.nn.Dropout(p=0.05),
            torch.nn.Linear(32,16),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.05),
            torch.nn.Linear(16,3)
        )

    def forward(self, input): #runs forward pass through network
        pred = self.layer_stack(input)
        return pred

ANN_model = ANN().to(device)

print(ANN_model)

#------Model training--------
loss = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(ANN_model.parameters(), lr = 0.001, weight_decay=0.0) #parameters, learning rate
loss_threshold = 0.001

training_error_vs_epoch, validation_error_vs_epoch = train_target_loss(training_inputs_norm, training_outputs_norm, optimizer, loss, loss_threshold, ANN_model, 500, validation_inputs_norm, validation_outputs_norm)

testing_inputs_norm_tensor = torch.tensor(testing_inputs_norm, dtype=torch.float32).to(device) #necessary when using gpu, have to send val data in tensor form over to gpu
with torch.no_grad(): #reduces memory usage when evaluating model, disables gradient calculation
    ANN_model.eval()
    model_outputs_norm = ANN_model(testing_inputs_norm_tensor).cpu().numpy() #pulling output data from gpu to cpu ram

model_outputs_denorm = output_scaler.inverse_transform(model_outputs_norm)

results_table = np.column_stack([testing_inputs, model_outputs_denorm, testing_outputs])

results_table_df = pd.DataFrame(results_table, columns= ["Crank Radius", "Link Length", "Offset", "Peak Power", "Min Link Width", "Cross-Sectional Area", "Real Power", "Real Link Width", "Real Area"])
sorted_results_df = results_table_df.sort_values(by = "Crank Radius", ascending=True)
sorted_results_df.to_csv("Model_Predictions.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet
print(sorted_results_df)

peak_power_rmse, peak_power_r_squared = test(testing_outputs[:,0], model_outputs_denorm[:,0])
print("---------------------------------")
print('Peak Power RMSE: ' + str(peak_power_rmse) + ' // Peak Power R^2: ' + str(peak_power_r_squared))
print("---------------------------------")

link_width_rmse, link_width_r_squared = test(testing_outputs[:,1], model_outputs_denorm[:,1])
print("---------------------------------")
print('Link Width RMSE: ' + str(link_width_rmse) + ' // Link Width R^2: ' + str(link_width_r_squared))
print("---------------------------------")

area_rmse, area_r_squared = test(testing_outputs[:,2], model_outputs_denorm[:,2])
print("---------------------------------")
print('Area RMSE: ' + str(area_rmse) + ' // Area R^2: ' + str(area_r_squared))
print("---------------------------------")

#save model weights for later reuse
torch.save(ANN_model.state_dict(), "ANN_Model.pt")

#Plotting training progress
plot_data_training = np.array(training_error_vs_epoch)
plot_data_validation = np.array(validation_error_vs_epoch)

plt.figure(1)
plt.plot(plot_data_training[:,0], plot_data_training[:,1], label = 'training')
plt.plot(plot_data_validation[:,0], plot_data_validation[:,1], label = 'validation')
plt.legend()
plt.xlabel('Epoch Count')
plt.ylabel('Mean Squared Error')
plt.title('Training Error vs Validation Error')
plt.grid(True)

plt.figure(2)
plt.xscale('log')
plt.plot(plot_data_training[:,0], plot_data_training[:,1], label = 'training')
plt.plot(plot_data_validation[:,0], plot_data_validation[:,1], label = 'validation')
plt.legend()
plt.xlabel('Epoch Count')
plt.ylabel('Mean Squared Error')
plt.title('Training Error vs Validation Error - Log Scale')
plt.grid(True)

peak_power_prediction = sorted_results_df["Peak Power"].to_numpy()
peak_power_validation = sorted_results_df["Real Power"].to_numpy()

plt.figure(3)
plt.plot(peak_power_prediction, 's', label = 'Prediction')
plt.plot(peak_power_validation, 's', label = 'Validation')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Predicted Value')
plt.title('Model Predictions Vs Validation - Peak Power')
plt.grid(True)

link_width_prediction = sorted_results_df["Min Link Width"].to_numpy()
link_width_validation = sorted_results_df["Real Link Width"].to_numpy()

plt.figure(4)
plt.plot(link_width_prediction, 's', label = 'Prediction')
plt.plot(link_width_validation, 's', label = 'Validation')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Predicted Value')
plt.title('Model Predictions Vs Validation - Link Width')
plt.grid(True)

area_prediction = sorted_results_df["Cross-Sectional Area"].to_numpy()
area_validation = sorted_results_df["Real Area"].to_numpy()

plt.figure(5)
plt.plot(area_prediction, 's', label = 'Prediction')
plt.plot(area_validation, 's', label = 'Validation')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Predicted Value')
plt.title('Model Predictions Vs Validation - Area')
plt.grid(True)

plt.show()