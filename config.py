import torch

# === Configuration parameters ===
T = 0.5                          # Total time horizon
input_dim = 1                   # Input feature dimension
neuron = 20                     # Hidden units per layer
N = 25                          # Number of time intervals
device = "cuda" if torch.cuda.is_available() else "cpu"
window_size = 2                 # Window size for input data, adjust as needed

# === Training settings ===
batch_size = 16384 * 15         # Large batch for 48D problem; reduce for lower-dimensional inputs
times_in_one_epochs = 256       # Iterations per epoch
num_epochs_terminal = 30        # Epochs for terminal condition training
num_epochs_general = 30         # Epochs for general case training

# === Derived parameters ===
dt = torch.tensor(T / N, device=device)          # Time step size
t_space = torch.linspace(0, T, N + 1, device=device)  # Discretized time grid


