import torch
import numpy as np
import sklearn
import matplotlib.pyplot as plt

np.random.seed(0) #set seeds so that results are reproducible run to run
torch.manual_seed(0)

def f(x,y):
    z = x**2-y**2
    return z

def normalize_data(dataset):
    normalized_data = (dataset-dataset.min())/(dataset.max()-dataset.min())
    return np.array(normalized_data).reshape(-1,1) #reshape(-1,1) flips the array from a row to a colum

def denormalize_data(normalized_dataset, dataset_max, dataset_min):
    denormalized_data = normalized_dataset*(dataset_max-dataset_min)+dataset_min
    return denormalized_data

#generate input dataset
x_dataset = np.arange(-100,100,0.1)

y_dataset = np.arange(-100,100,0.1)

#split the input dataset into two randomized arrays for training and validation
x_train, x_val = sklearn.model_selection.train_test_split(x_dataset, test_size=0.3, train_size=0.7, random_state=None, shuffle=True)

y_train, y_val = sklearn.model_selection.train_test_split(x_dataset, test_size=0.3, train_size=0.7, random_state=None, shuffle=True)

z_train = f(x_train,y_train) #output training dataset

z_val = f(x_val, y_val) #output validation dataset

#normalize input data
x_val_normalized = normalize_data(x_val)

y_val_normalized = normalize_data(y_val)

z_val_normalized = normalize_data(z_val)

x_train_normalized = normalize_data(x_train)

y_train_normalized = normalize_data(y_train)

z_train_normalized = normalize_data(z_train)

#Stack x and y columns widthwise
train_dataset = np.column_stack(np.float32((x_train_normalized,y_train_normalized)))

val_dataset = np.column_stack(np.float32((x_val_normalized, y_val_normalized)))

def test(z_val,z_pred):
    rmse = sklearn.metrics.root_mean_squared_error(z_val, z_pred)
    r_squared = sklearn.metrics.r2_score(z_val, z_pred)
    return rmse, r_squared

def train(input_data, output_target, optimizer, loss, n_epochs):
    inputs = torch.from_numpy(np.float32(input_data))
    targets = torch.from_numpy(np.float32(output_target))

    for epoch in range(n_epochs):

        outputs = ANN_model(inputs)

        error = loss(outputs, targets)

        optimizer.zero_grad()

        error.backward()

        optimizer.step()

        print(str(epoch) + " Error: " + str(error))

def train_target_error(input_data, output_target, optimizer, loss, r_squared_target):
    inputs = torch.from_numpy(np.float32(input_data))
    targets = torch.from_numpy(np.float32(output_target))
    r_squared = 0
    epochs = 0

    while r_squared < r_squared_target:

        epochs += 1

        outputs = ANN_model_2(inputs)

        error = loss(outputs, targets)

        optimizer.zero_grad()

        error.backward()

        optimizer.step()

        z_pred = ANN_model_2(torch.from_numpy(val_dataset)).detach().numpy()

        r_squared  = test(z_val_normalized, z_pred)[1]

        print('Epochs: ' + str(epochs) + ' Error: ' + str(error) + ' // RMSE: ' + str(test(z_val_normalized, z_pred)[0]) + ' --- R^2: ' + str(test(z_val_normalized, z_pred)[1]))
    

class ANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(2,32), #Number of inputs to layer / Number of outputs from layer
            torch.nn.ReLU(), #outputs zero for any value < 0, outputs input value for values > 0
            torch.nn.BatchNorm1d(32),
            torch.nn.Linear(32,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,8), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(8),
            torch.nn.Linear(8,1)
        )

    def forward(self, input): #runs forward pass through network
        pred = self.layer_stack(input)
        return pred

ANN_model = ANN().to('cpu')
ANN_model_2 = ANN().to('cpu') 

print(ANN_model)

#------Model training--------
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ANN_model.parameters(), lr = 0.01) #parameters, learning rate

#25 Epochs
train(train_dataset,z_train_normalized, optimizer, loss, 25)
ANN_model.eval()
z_pred_25_epoch = ANN_model(torch.from_numpy(val_dataset)).detach().numpy()

#50 Epochs
ANN_model.train()
train(train_dataset,z_train_normalized, optimizer, loss, 25)
ANN_model.eval()
z_pred_50_epoch = ANN_model(torch.from_numpy(val_dataset)).detach().numpy()

#100 Epochs
ANN_model.train()
train(train_dataset,z_train_normalized, optimizer, loss, 50)
ANN_model.eval()
z_pred_100_epoch = ANN_model(torch.from_numpy(val_dataset)).detach().numpy()

#Doesn't work currently
ANN_model_2.train()
train_target_error(train_dataset,z_train_normalized, optimizer, loss, 0.99)
ANN_model_2.eval()
z_pred_r_squared_method = ANN_model_2(torch.from_numpy(val_dataset)).detach().numpy()


print('25 Epochs // RMSE: ' + str(test(z_val_normalized,z_pred_25_epoch)[0]) + ' --- R^2: ' + str(test(z_val_normalized,z_pred_25_epoch)[1]))
print('50 Epochs // RMSE: ' + str(test(z_val_normalized,z_pred_50_epoch)[0]) + ' --- R^2: ' + str(test(z_val_normalized,z_pred_50_epoch)[1]))
print('100 Epochs // RMSE: ' + str(test(z_val_normalized,z_pred_100_epoch)[0]) + ' --- R^2 :' + str(test(z_val_normalized,z_pred_100_epoch)[1]))


z_val_denorm = denormalize_data(z_val_normalized, z_val.max(), z_val.min())

x_val_denorm = denormalize_data(x_val_normalized, x_val.max(), x_val.min())

y_val_denorm = denormalize_data(y_val_normalized, y_val.max(), y_val.min())

z_pred_25_epoch_denorm = denormalize_data(z_pred_25_epoch, z_val.max(), z_val.min())

z_pred_50_epoch_denorm = denormalize_data(z_pred_50_epoch, z_val.max(), z_val.min())

z_pred_100_epoch_denorm = denormalize_data(z_pred_100_epoch, z_val.max(), z_val.min())

plot_3d_denormalized = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x_val_denorm,y_val_denorm,z_val_denorm,'blue') #val_data
ax.scatter(x_val_denorm,y_val_denorm,z_pred_25_epoch_denorm,'red') #25 epoch
ax.scatter(x_val_denorm,y_val_denorm,z_pred_50_epoch_denorm,'orange') #50 epoch
ax.scatter(x_val_denorm,y_val_denorm,z_pred_100_epoch_denorm,'green') #100 epoch
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('ANN Predictions Compared to Actual Values (Denormalized)')
ax.grid(True)

plt.show()

