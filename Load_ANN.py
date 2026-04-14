import torch
import numpy as np
import pickle

# 1. Define the Model Architecture (Must match the training script exactly)
class ANN(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(16, n_outputs)
        )

    def forward(self, x):
        return self.layer_stack(x)

def load_inference_system(model_path, scaler_path):
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Scalers
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
        # Handle case if you saved a dict or just the output_scaler
        if isinstance(scalers, dict):
            input_scaler = scalers['input_scaler']
            output_scaler = scalers['output_scaler']
        else:
            input_scaler = None # You will need to fit this again if not saved
            output_scaler = scalers

    # Load Model
    # Note: number_inputs=3, number_outputs=7 based on your training script
    model = ANN(3, 7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode (turns off Dropout)
    
    return model, input_scaler, output_scaler, device

def predict(input_data, model, input_scaler, output_scaler, device):
    """
    input_data: list or numpy array [Crank Radius, Link Length, Offset]
    """
    # 1. Format input
    data_array = np.array(input_data).reshape(1, -1).astype(np.float32)
    
    # 2. Scale input
    data_norm = input_scaler.transform(data_array)
    
    # 3. Convert to Tensor
    data_tensor = torch.from_numpy(data_norm).to(device)
    
    # 4. Predict
    with torch.no_grad():
        prediction_norm = model(data_tensor)
    
    # 5. Denormalize output
    prediction_denorm = output_scaler.inverse_transform(prediction_norm.cpu().numpy())
    
    return prediction_denorm[0]

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # File paths
    MODEL_FILE = "ANN_Model.pt"
    SCALER_FILE = "scalers.pkl"
    
    try:
        model, in_scaler, out_scaler, device = load_inference_system(MODEL_FILE, SCALER_FILE)
        
        # Example: Define new inputs [Crank Radius, Link Length, Offset]
        my_inputs = [121.5*10**-3 , 138.0*10**-3 , 13.5*10**-3 ]
        
        results = predict(my_inputs, model, in_scaler, out_scaler, device)
        
        # Define column names for printing
        output_columns = ["Return Ratio", "Peak Power", "System Volume", "System Area", 
                          "Min Link Width", "Min Pin Diameter", "Min Crank Width"]
        
        print("\n--- Prediction Results ---")
        for name, value in zip(output_columns, results):
            print(f"{name:20s}: {value:.4f}")
            
    except FileNotFoundError:
        print("Error: Ensure ANN_Model.pt and scalers.pkl are in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")