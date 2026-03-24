import torch
import pickle

class ANN(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(n_inputs,64), #Number of inputs to layer / Number of outputs from layer
            torch.nn.ReLU(), #outputs zero for any value < 0, outputs input value for values > 0
            #torch.nn.Dropout(p=0.05),
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
            #torch.nn.Dropout(p=0.05),
            torch.nn.Linear(32,n_outputs)
        )

    def forward(self, input): #runs forward pass through network
        pred = self.layer_stack(input)
        return pred

ann_model = ANN(3,7).to('cpu')
ann_model.load_state_dict(torch.load("ANN_Model.pt", weights_only=True))
ann_model.eval()

output_scaler = pickle.load(open('scaler.pkl', 'rb'))

with torch.no_grad():
    input_tensor = torch.tensor([0.11, 0.162, 0.049]) #crank, link, offset
    results = ann_model(input_tensor).numpy()
    scaled_results = output_scaler.inverse_transform(results)

print(results)
    