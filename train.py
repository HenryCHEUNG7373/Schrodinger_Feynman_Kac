from config import T, input_dim, neuron, N, device,dt, t_space, batch_size, times_in_one_epochs, num_epochs_terminal, num_epochs_general, window_size
import torch
from time import time
from torch.cuda.amp import autocast, GradScaler
import math
import matplotlib.pyplot as plt
import datetime
import torch.optim as optim
import gc
from torch.quasirandom import SobolEngine
from utils import save_models, plot_graph_varies_one_axis

def train_backward_sweep(
    y, y_bar, z, z_bar,
    loss_fn_till_end,
    draw_X_and_dW,
    funr, funi, fun_dGr, fun_dGi, fun_Gr, fun_Gi, 
    true_solution_real, y_lower_bd, y_upper_bd
):
    '''
    Train the model backward.
    Args:
        y (nn.Module): The model for y.
        y_bar (nn.Module): The model for y_bar.
        z (nn.Module): The model for z.
        z_bar (nn.Module): The model for z_bar.
        loss_fn_till_end (callable): The loss function to be used.
        draw_X_and_dW (callable): Function to generate input data and noise.
        funr, funi, fun_dGr, fun_dGi, fun_Gr, fun_Gi (callable): Functions defining the problem.
        true_solution_real (callable): The true solution function of the real part for comparison. 
        y_lower_bd (float): Lower bound for y-axis in plots.
        y_upper_bd (float): Upper bound for y-axis in plots.
    '''
    history = []
    t0 = time()

    for current_step in reversed(range(N)):
        torch.cuda.empty_cache()
        if current_step == N-1:
            num_epochs = num_epochs_terminal
        else:
            num_epochs = num_epochs_general

        # Move models not being trained at this step to CPU to save memory
        if current_step < N - 2:
            y.y[current_step + 2].to("cpu")
            y_bar.y[current_step + 2].to("cpu")
            z.z[current_step + 2].to("cpu")
            z_bar.z[current_step + 2].to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        # Collect trainable parameters at this time step
        current_params = (
            list(y.y[current_step].parameters()) +
            list(y_bar.y[current_step].parameters()) +
            list(z.z[current_step].parameters()) +
            list(z_bar.z[current_step].parameters())
        )

        # Optimizer and scheduler defined per time step
        optimizer = torch.optim.AdamW(current_params, lr=5e-3, weight_decay=1e-2,amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_terminal)
        scaler = GradScaler()

        # Training
        for i in range(num_epochs):

            print('Step   Iter        Loss       |  Time')
            for j in range(times_in_one_epochs):

                # Generate Sobol sequence for input
                sobol = SobolEngine(dimension=input_dim, scramble=True)  # scramble=True adds randomness
                x =  2*window_size*(sobol.draw(batch_size).to(device)-0.5) 
                X, dW = draw_X_and_dW(batch_size, x, input_dim, dt)

                # Standard PyTorch training steps
                optimizer.zero_grad()
                with autocast():
                    loss = loss_fn_till_end(current_step, (X, dW), y, y_bar, z, z_bar, funr,funi, fun_dGr, fun_dGi, fun_Gr, fun_Gi)
                scaler.scale(loss).backward()       
                scaler.step(optimizer)             
                scaler.update()  
                scheduler.step()

                # Print the training progress
                currtime = time() - t0
                hentry = (current_step, i, loss.item(), currtime, scheduler.get_last_lr()[0])
                history.append(hentry)
                if j == 2:
                    print('{:5d} {:5d} {:12.6f}      | {:8.2e}     {:8.2e}'.format(*hentry))
        
        # Plot graph and save the model after each time step
        plot_graph_varies_one_axis(y,current_step, window_size, 0, true_solution_real, y_lower_bd,y_upper_bd)
        save_models(y,y_bar,z,z_bar)