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
input_columns = ["Crank Radius", "Link Length", "Offset"]
input_dataset = doe_dataset[input_columns].to_numpy().astype(np.float32)

#output values
output_columns = ["Peak Power", "Min Link Width", "Min Crank Width", "Min Pin Diameter", "Return Ratio", "Cross-Sectional Area"]
output_dataset = doe_dataset[output_columns].to_numpy().astype(np.float32)

number_inputs = input_dataset.shape[1] #counts columns of dataset
number_outputs = output_dataset.shape[1]

#split the dataset into two randomized arrays for training & validation
training_inputs, temp_inputs, training_outputs, temp_outputs = sklearn.model_selection.train_test_split(input_dataset, output_dataset, test_size=0.3, random_state=1, shuffle=True)
validation_inputs, testing_inputs, validation_outputs, testing_outputs = sklearn.model_selection.train_test_split(temp_inputs, temp_outputs, test_size=0.5, random_state=1, shuffle=True)

training_inputs_norm = input_scaler.fit_transform(training_inputs)
validation_inputs_norm = input_scaler.transform(validation_inputs)
testing_inputs_norm = input_scaler.transform(testing_inputs)

training_outputs_norm = output_scaler.fit_transform(training_outputs)
validation_outputs_norm = output_scaler.transform(validation_outputs)

def performance_metrics(validation, prediction):
    '''Outputs RMSE and R^2'''
    rmse = sklearn.metrics.root_mean_squared_error(validation, prediction)
    r_squared = sklearn.metrics.r2_score(validation, prediction)
    return rmse, r_squared


def train_target_loss(input_data, output_target, optimizer, loss_function, loss_threshold, ann_model, max_epochs, validation_inputs, validation_targets):
    training_error_vs_epoch = []
    validation_error_vs_epoch = []
    inputs = torch.from_numpy(np.float32(input_data)).to(device)
    targets = torch.from_numpy(np.float32(output_target)).to(device)
    
    training_error = 1
    validation_error = float('inf')
    best_validation_loss = float('inf')
    epochs_no_improvement = 0
    early_stop_counter = 100 #number of epochs with no improvement required before ending training
    epochs = 0

    ann_model.train()

    while training_error > loss_threshold and epochs < max_epochs and epochs_no_improvement < early_stop_counter:
        epochs += 1
        outputs = ann_model(inputs)
        training_error = loss_function(outputs, targets)
        optimizer.zero_grad()
        training_error.backward()
        optimizer.step()
        training_error_vs_epoch.append([epochs, training_error.item()])

        if epochs % 10 == 0: #periodic validation
            ann_model.eval()
            with torch.no_grad():
                validation_error = loss_function(ann_model(torch.tensor(validation_inputs, dtype=torch.float32).to(device)), torch.tensor(validation_targets, dtype=torch.float32).to(device))
            validation_loss = validation_error.item()
            validation_error_vs_epoch.append([epochs, validation_error.item()])
            ann_model.train()

            if validation_loss < best_validation_loss: #validation has improved over last 10 epochs
                best_validation_loss = validation_loss
                best_model_weights = ann_model.state_dict()
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 10 #validation hasn't improved over last 10 epochs

        if epochs % 10 == 0:
          print('Epochs: ' + str(epochs) + ' Training Error: ' + str(training_error) + ' Validation Error' + str(validation_error))

    if best_model_weights is not None: #Restores model to best validation loss 
        ann_model.load_state_dict(best_model_weights)

    ann_model.eval()
    return training_error_vs_epoch, validation_error_vs_epoch
    

class ANN(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(n_inputs,64), #Number of inputs to layer / Number of outputs from layer
            torch.nn.ReLU(), #outputs zero for any value < 0, outputs input value for values > 0
            torch.nn.Dropout(p=0.05),
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.05),
            torch.nn.Linear(32,n_outputs)
        )

    def forward(self, input): #runs forward pass through network
        pred = self.layer_stack(input)
        return pred

slider_ann_model = ANN(number_inputs, number_outputs).to(device)

print(slider_ann_model)

#------Model training--------
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(slider_ann_model.parameters(), lr = 0.001, weight_decay=0.05) #parameters, learning rate
loss_threshold = 0.0001

training_error_vs_epoch, validation_error_vs_epoch = train_target_loss(training_inputs_norm, training_outputs_norm, optimizer, loss_function, loss_threshold, slider_ann_model, 100000, validation_inputs_norm, validation_outputs_norm)

testing_inputs_norm_tensor = torch.tensor(testing_inputs_norm, dtype=torch.float32).to(device) #necessary when using gpu, have to send val data in tensor form over to gpu
with torch.no_grad(): #reduces memory usage when evaluating model, disables gradient calculation
    slider_ann_model.eval()
    model_outputs_norm = slider_ann_model(testing_inputs_norm_tensor).cpu().numpy() #pulling output data from gpu to cpu ram

model_outputs_denorm = output_scaler.inverse_transform(model_outputs_norm)

results_table = np.column_stack([testing_inputs, model_outputs_denorm, testing_outputs])

results_table_df = pd.DataFrame(results_table, columns= ["Crank Radius", "Link Length", "Offset", "Peak Power", "Min Link Width", "Min Crank Width", "Min Pin Diameter", "Return Ratio", "Cross-Sectional Area", "Real Peak Power", "Real Min Link Width", "Real Min Crank Width", "Real Min Pin Diameter", "Real Return Ratio", "Real Cross-Sectional Area"])

sorted_results_df = results_table_df.sort_values(by = "Crank Radius", ascending=True)
sorted_results_df.to_csv("Model_Predictions.csv", index=False, float_format='%.4f') #exports data to a .csv spreadsheet
print(sorted_results_df)

for i, col_name in enumerate(output_columns):
    rmse, r2 = performance_metrics(testing_outputs[:, i], model_outputs_denorm[:, i])
    print(f"  {col_name:25s} | RMSE: {rmse:.6f} | R^2: {r2:.4f}")

# Overall metrics
overall_rmse, overall_r2 = performance_metrics(testing_outputs, model_outputs_denorm)
print("-"*60)
print(f"  {'OVERALL':25s} | RMSE: {overall_rmse:.6f} | R^2: {overall_r2:.4f}")
print("="*60)

#save model weights for later reuse
torch.save(slider_ann_model.state_dict(), "ANN_Model.pt")

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
peak_power_validation = sorted_results_df["Real Peak Power"].to_numpy()

plt.figure(3)
plt.plot(peak_power_prediction, 's', label = 'Prediction')
plt.plot(peak_power_validation, 's', label = 'Validation')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Predicted Value')
plt.title('Model Predictions Vs Validation - Peak Power')
plt.grid(True)

link_width_prediction = sorted_results_df["Min Link Width"].to_numpy()
link_width_validation = sorted_results_df["Real Min Link Width"].to_numpy()

plt.figure(4)
plt.plot(link_width_prediction, 's', label = 'Prediction')
plt.plot(link_width_validation, 's', label = 'Validation')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Predicted Value')
plt.title('Model Predictions Vs Validation - Link Width')
plt.grid(True)

area_prediction = sorted_results_df["Cross-Sectional Area"].to_numpy()
area_validation = sorted_results_df["Real Cross-Sectional Area"].to_numpy()

plt.figure(5)
plt.plot(area_prediction, 's', label = 'Prediction')
plt.plot(area_validation, 's', label = 'Validation')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Predicted Value')
plt.title('Model Predictions Vs Validation - Area')
plt.grid(True)

# Parity plots: predicted vs actual for each output
n_outputs = len(output_columns)
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
axes2 = axes2.flatten()

for i, col_name in enumerate(output_columns):
    pred = model_outputs_denorm[:, i]
    actual = testing_outputs[:, i]
    rmse, r2 = performance_metrics(actual, pred)
    
    axes2[i].scatter(actual, pred, alpha=0.5, s=15)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    axes2[i].plot(lims, lims, 'r--', linewidth=1, label='Perfect fit')
    axes2[i].set_xlabel('Actual')
    axes2[i].set_ylabel('Predicted')
    axes2[i].set_title(f'{col_name}\nR²={r2:.4f}, RMSE={rmse:.4f}')
    axes2[i].legend()
    axes2[i].grid(True)
    axes2[i].set_aspect('equal', adjustable='box')

# Hide unused subplot
if n_outputs < len(axes2):
    for j in range(n_outputs, len(axes2)):
        axes2[j].set_visible(False)

plt.tight_layout()

plt.show()