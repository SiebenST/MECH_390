import torch
import numpy as np
import sklearn
import matplotlib.pyplot as plt

np.random.seed(0) #set seeds so that results are reproducible run to run
torch.manual_seed(0)

def f(x,y):
    z = x**2-y**2
    return z

#generate input dataset
x_dataset = np.arange(-100,100,0.1)

y_dataset = np.arange(-100,100,0.1)

#split the input dataset into two randomized arrays for training and validation
x_train, x_val = sklearn.model_selection.train_test_split(x_dataset, test_size=0.3, train_size=0.7, random_state=None, shuffle=True)

y_train, y_val = sklearn.model_selection.train_test_split(x_dataset, test_size=0.3, train_size=0.7, random_state=None, shuffle=True)

z_train = f(x_train,y_train) #output training dataset

z_val = f(x_val, y_val) #output validation dataset

x_max = x_train.max()

x_min = x_train.min()

y_max = y_train.max()

y_min = y_train.min()

z_max = z_train.max()

z_min = z_train.min()


x_val_normalized = (x_val-x_min)/(x_max-x_min)

y_val_normalized = (y_val-y_min)/(y_max-y_min)

z_val_normalized = (z_val-z_min)/(z_max-z_min)

x_train_normalized = (x_train-x_min)/(x_max-x_min)

y_train_normalized = (y_train-y_min)/(y_max-y_min)

z_train_normalized = (z_train-z_min)/(z_max-z_min)

x_val_normalized = np.array(x_val_normalized).reshape(-1,1)

y_val_normalized = np.array(y_val_normalized).reshape(-1,1)

z_val_normalized = np.array(z_val_normalized).reshape(-1,1)

x_train_normalized = np.array(x_train_normalized).reshape(-1,1)

y_train_normalized = np.array(y_train_normalized).reshape(-1,1)

z_train_normalized = np.array(z_train_normalized).reshape(-1,1)

#normalize input data
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


print('25 Epochs // RMSE: ' + str(test(z_val_normalized,z_pred_25_epoch)[0]) + ' --- R^2: ' + str(test(z_val_normalized,z_pred_25_epoch)[1]))
print('50 Epochs // RMSE: ' + str(test(z_val_normalized,z_pred_50_epoch)[0]) + ' --- R^2: ' + str(test(z_val_normalized,z_pred_50_epoch)[1]))
print('100 Epochs // RMSE: ' + str(test(z_val_normalized,z_pred_100_epoch)[0]) + ' --- R^2 :' + str(test(z_val_normalized,z_pred_100_epoch)[1]))

# plot_3d_normalized = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(x_val_normalized,y_val_normalized,z_val_normalized,'blue') #val_data
# ax.scatter(x_val_normalized,y_val_normalized,z_pred_25_epoch,'red') #10 epoch
# ax.scatter(x_val_normalized,y_val_normalized,z_pred_50_epoch,'orange') #50 epoch
# ax.scatter(x_val_normalized,y_val_normalized,z_pred_100_epoch,'green') #100 epoch
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title('ANN Predictions Compared to Actual Values (Normalized)')
# ax.grid(True)

z_val_denorm = z_val_normalized*(z_max-z_min)+z_min

x_val_denorm = x_val_normalized*(x_max-x_min)+x_min

z_pred_25_epoch_denorm = z_pred_25_epoch*(z_max-z_min)+z_min

z_pred_50_epoch_denorm = z_pred_50_epoch*(z_max-z_min)+z_min

z_pred_100_epoch_denorm = z_pred_100_epoch*(z_max-z_min)+z_min

plot_3d_denormalized = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x_val_normalized,y_val_normalized,z_val_denorm,'blue') #val_data
ax.scatter(x_val_normalized,y_val_normalized,z_pred_25_epoch_denorm,'red') #25 epoch
ax.scatter(x_val_normalized,y_val_normalized,z_pred_50_epoch_denorm,'orange') #50 epoch
ax.scatter(x_val_normalized,y_val_normalized,z_pred_100_epoch_denorm,'green') #100 epoch
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('ANN Predictions Compared to Actual Values (Denormalized)')
ax.grid(True)

plt.show()

