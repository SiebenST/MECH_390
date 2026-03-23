import torch

def optimization_function(power_input, size_input, power_ideal, size_ideal, power_importance, size_importance):
    optimization_score = power_ideal/power_input*power_importance + size_ideal/size_input*size_importance
    return optimization_score

class ANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(3,32), #Number of inputs to layer / Number of outputs from layer
            torch.nn.ReLU(), #outputs zero for any value < 0, outputs input value for values > 0
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(32,16),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(16,3)
        )

    def forward(self, input): #runs forward pass through network
        pred = self.layer_stack(input)
        return pred

ann_model = ANN().to('cpu')
ann_model.load_state_dict(torch.load("ANN_Model.pt", weights_only=True))
ann_model.eval()

with torch.no_grad():
    input_tensor = torch.tensor([[0.114, 0.172, 0.0485], [0.114, 0.172, 0.0485], [0.114, 0.172, 0.0485]]) #crank, link, offset
    results = ann_model(input_tensor).numpy()

print(results)
    