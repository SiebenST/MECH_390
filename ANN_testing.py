import torch
import numpy as np
import sklearn
import matplotlib.pyplot as plt

def f(x):
    y = x**3 + 5
    return y

#generate input dataset
x_dataset = np.arange(1,1001,1)

#split the input dataset into two randomized arrays for training and validation
x_train, x_val = sklearn.model_selection.train_test_split(x_dataset, test_size=300, train_size=700, random_state=None, shuffle=True)

y_train = f(x_train) #output training dataset

y_val = f(x_val) #output validation dataset

y_max = y_train.max()

y_min = y_train.min()

x_max = x_train.max()

x_min = x_train.min()

y_val_normalized = (y_val-y_min)/(y_max-y_min)

x_val_normalized = (x_val-x_min)/(x_max-x_min)

y_train_normalized = (y_train-y_min)/(y_max-y_min)

x_train_normalized = (x_train-x_min)/(x_max-x_min)

x_val_normalized = np.array(x_val_normalized).reshape(-1,1)

y_val_normalized = np.array(y_val_normalized).reshape(-1,1)

x_train_normalized = np.array(x_train_normalized).reshape(-1,1)

y_train_normalized = np.array(y_train_normalized).reshape(-1,1)

#normalize input data


def test(y_val,y_pred):
    rmse = sklearn.metrics.root_mean_squared_error(y_val, y_pred)
    r_squared = sklearn.metrics.r2_score(y_val, y_pred)
    return rmse, r_squared

def train(input_data, output_target, optimizer, loss, n_epochs):
    for epoch in range(n_epochs):
        inputs = torch.from_numpy(np.float32(input_data))
        targets = torch.from_numpy(np.float32(output_target))

        outputs = ANN_model(inputs)

        error = loss(outputs, targets)

        optimizer.zero_grad()

        error.backward()

        optimizer.step()

        print(str(epoch) + " Error: " + str(error))

class ANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(1,32), #Number of inputs to layer / Number of outputs from layer
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
print(ANN_model)

#------Model training--------
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ANN_model.parameters(), lr = 0.01) #parameters, learning rate
train(x_train_normalized,y_train_normalized, optimizer, loss, 10)

y_pred_10_epoch = ANN_model(torch.from_numpy(np.float32(x_val_normalized))).detach().numpy()

train(x_train_normalized,y_train_normalized, optimizer, loss, 50)

y_pred_50_epoch = ANN_model(torch.from_numpy(np.float32(x_val_normalized))).detach().numpy()

train(x_train_normalized,y_train_normalized, optimizer, loss, 100)

y_pred_100_epoch = ANN_model(torch.from_numpy(np.float32(x_val_normalized))).detach().numpy()

plt.figure(1)
val_data = plt.scatter(x_val_normalized,y_val_normalized)
epoch_10 = plt.scatter(x_val_normalized,y_pred_10_epoch)
epoch_50 = plt.scatter(x_val_normalized,y_pred_50_epoch)
epoch_100 = plt.scatter(x_val_normalized,y_pred_100_epoch)
plt.xlabel('x_val')
plt.ylabel('y_val')
plt.title('ANN Predictions Compared to Actual Values (Normalized)')
plt.legend([val_data, epoch_10, epoch_50, epoch_100], ["Validaton", "10 Epochs", "50 Epochs", "100 Epochs"])
plt.grid(True)

y_val_denorm = y_val_normalized*(y_max-y_min)+y_min

x_val_denorm = x_val_normalized*(x_max-x_min)+x_min

y_pred_10_epoch_denorm = y_pred_10_epoch*(y_max-y_min)+y_min

y_pred_50_epoch_denorm = y_pred_50_epoch*(y_max-y_min)+y_min

y_pred_100_epoch_denorm = y_pred_100_epoch*(y_max-y_min)+y_min

plt.figure(2)
val_data = plt.scatter(x_val_denorm,y_val_denorm)
epoch_10 = plt.scatter(x_val_denorm,y_pred_10_epoch_denorm)
epoch_50 = plt.scatter(x_val_denorm,y_pred_50_epoch_denorm)
epoch_100 = plt.scatter(x_val_denorm,y_pred_100_epoch_denorm)
plt.xlabel('x_val')
plt.ylabel('y_val')
plt.title('ANN Predictions Compared to Actual Values (Denormalized)')
plt.legend([val_data, epoch_10, epoch_50, epoch_100], ["Validaton", "10 Epochs", "50 Epochs", "100 Epochs"])
plt.grid(True)

plt.show()

