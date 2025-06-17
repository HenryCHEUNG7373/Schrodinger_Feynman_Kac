# main.py

# === Environment Config ===
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# === Standard Library ===
import torch

# === Local Modules ===
from config import (
    T, input_dim, neuron, N, device, dt, t_space,
    batch_size, times_in_one_epochs,
    num_epochs_terminal, num_epochs_general, window_size
)
from linear_case_related_functions import (
    funr, funi, fun_Gr, fun_Gi, fun_dGr, fun_dGi,
    true_solution_real, true_solution_img
)
'''
from nonlinear_case_related_functions import (
    funr, funi, fun_Gr, fun_Gi, fun_dGr, fun_dGi,
    true_solution_real, true_solution_img
)
'''
from losses import loss_fn_till_end
from train import train_backward_sweep
from model import BSDEModel_y_series, BSDEModel_z_series
from utils import draw_X_and_dW

# === Model Initialization ===
# y stands for the real part of the solution
# y_bar stands for the imaginary part of the solution
# z stands for the gradient of the real part
# z_bar stands for the gradient of the imaginary part

y = BSDEModel_y_series(N, input_dim, neuron).to(device)
y_bar = BSDEModel_y_series(N, input_dim, neuron).to(device)
z = BSDEModel_z_series(N, input_dim, neuron).to(device)
z_bar = BSDEModel_z_series(N, input_dim, neuron).to(device)

# === Training Entrypoint ===
if __name__ == "__main__":
    history = train_backward_sweep(
        y=y,
        y_bar=y_bar,
        z=z,
        z_bar=z_bar,
        loss_fn_till_end=loss_fn_till_end,
        draw_X_and_dW=draw_X_and_dW,
        funr=funr,
        funi=funi,
        fun_dGr=fun_dGr,
        fun_dGi=fun_dGi,
        fun_Gr=fun_Gr,
        fun_Gi=fun_Gi,
        true_solution_real=true_solution_real,  # used for plot comparison
        y_lower_bd=0,
        y_upper_bd=1.2
    )
