import torch
from src.config import T, input_dim, neuron, N, device,dt, t_space, batch_size, times_in_one_epochs, num_epochs_terminal, num_epochs_general, window_size

def simulate_Y_once(current_step, inp, model_y, model_y_bar, model_z, model_z_bar, funr, funi, fun_dGr, fun_dGi):
    '''
    Simulate the model for one time step given the current step.
    Args:
        current_step (int): The current time step index (N-1 to 0).
        inp (tuple): A tuple containing the input tensor X and the Wiener increment dW.
        model_y (nn.Module): The model for y.
        model_y_bar (nn.Module): The model for y_bar.
        model_z (nn.Module): The model for z.
        model_z_bar (nn.Module): The model for z_bar.
        funr (function): Function to compute the real part of f.
        funi (function): Function to compute the imaginary part of f.
        fun_dGr (function): Function to compute the derivative of Gr.
        fun_dGi (function): Function to compute the derivative of Gi.
    Returns:
        tuple: The updated values of y and y_bar after the simulation step.
    '''
    #Check the input current_step to be within 0 to N-1
    if not (0 <= current_step <= N-1):
        raise ValueError(f"Current step must be between 0 and N-1")
    X, dW = inp
    
    #Calculate the output of the model
    y_now = model_y(X[:, :, 0], current_step)
    ybar_now = model_y_bar(X[:, :, 0], current_step)
    z_now = model_z(X[:, :, 0], current_step)
    zbar_now = model_z_bar(X[:, :, 0], current_step)
    if current_step == N-1:
        z_next = fun_dGr(X[:,:,1])
        zbar_next = fun_dGi(X[:,:,1])
    else:
        z_next = model_z(X[:, :, 1], current_step + 1)
        zbar_next = model_z_bar(X[:, :, 1], current_step + 1)
    
    eta1r = funi(current_step * T / N, X[:, :, 0], y_now, ybar_now) * dt
    eta1i = -funr(current_step * T / N, X[:, :, 0], y_now, ybar_now) * dt

    eta2r = 0.5 * torch.sum((z_next + zbar_next - z_now - zbar_now) * dW, dim=1, keepdims=True)
    eta2r += torch.sum(z_now * dW, dim=1, keepdims=True)

    eta2i = -0.5 * torch.sum((z_next - zbar_next - z_now + zbar_now) * dW, dim=1, keepdims=True)
    eta2i += torch.sum(zbar_now * dW, dim=1, keepdims=True)

    y_now = y_now + eta1r + eta2r
    ybar_now = ybar_now + eta1i + eta2i

    return y_now, ybar_now

def loss_fn_till_end(current_step, inp, model_y, model_y_bar, model_z, model_z_bar, funr,funi, fun_dGr, fun_dGi, fun_Gr, fun_Gi):
    '''
    Compute the loss function for the model at a given time step.
    Args:
        current_step (int): The current time step index (N-1 to 0).
        inp (tuple): A tuple containing the input tensor X and the Wiener increment dW.
        model_y (nn.Module): The model for y.
        model_y_bar (nn.Module): The model for y_bar.
        model_z (nn.Module): The model for z.
        model_z_bar (nn.Module): The model for z_bar.
        funr (function): Function to compute the real part of f.
        funi (function): Function to compute the imaginary part of f.
        fun_dGr (function): Function to compute the derivative of Gr.
        fun_dGi (function): Function to compute the derivative of Gi.
        fun_Gr (function): Function to compute Gr.
        fun_Gi (function): Function to compute Gi.
    Returns:
        torch.Tensor: The computed loss value.
    '''
    #Check the input current_step to be within 0 to N-1
    if not (0 <= current_step <= N-1):
        raise ValueError(f"Current step must be between 0 and N-1")
    X, _ = inp
    y_pred, ybar_pred = simulate_Y_once(current_step, inp, model_y, model_y_bar, model_z, model_z_bar, funr, funi, fun_dGr, fun_dGi)
    if current_step == N-1:
        y_true = fun_Gr(X[:,:,1])
        ybar_true = fun_Gi(X[:,:,1])
    else:
        y_true = model_y(X[:,:,1], current_step + 1)
        ybar_true = model_y_bar(X[:,:,1], current_step + 1)
    #Compute the loss
    y_diff = y_true - y_pred
    ybar_diff = ybar_true - ybar_pred
    loss = torch.mean(torch.square(y_diff)) + torch.mean(torch.square(ybar_diff))

    return loss