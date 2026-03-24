import torch

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

with torch.no_grad():
    input_tensor = torch.tensor([[0.114, 0.172, 0.0485], [0.114, 0.122, 0.005], [0.104, 0.132, 0.003]]) #crank, link, offset
    results = ann_model(input_tensor).numpy()

print(results)
    