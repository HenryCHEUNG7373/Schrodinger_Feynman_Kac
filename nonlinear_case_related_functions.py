import torch
import math
from config import input_dim, device
from config import T, input_dim, neuron, N, device,dt, t_space, batch_size, times_in_one_epochs, num_epochs_terminal, num_epochs_general, window_size

def funr(t,x,yr,yi):
    sum_x = x.sum(dim=1, keepdim=True)  # Sum across `input_dim` (dim=1)
    frest_r1 = torch.cos(T-t+sum_x/input_dim)/torch.cosh(sum_x/input_dim) + 1/input_dim*torch.cos(T-t+sum_x/input_dim)/(torch.cosh(sum_x/input_dim)**3)
    frest_r2 = -1/input_dim*torch.sin(T-t+sum_x/input_dim)/torch.cosh(sum_x/input_dim)*torch.tanh(sum_x/input_dim) - torch.cos(T-t+sum_x/input_dim)/(torch.cosh(sum_x/input_dim)**3)
    return (yr**2+yi**2)*yr + frest_r1 + frest_r2

def funi(t,x,yr,yi):
    sum_x = x.sum(dim=1, keepdim=True)  # Sum across `input_dim` (dim=1)
    frest_i1 = torch.sin(T-t+sum_x/input_dim)/torch.cosh(sum_x/input_dim) + 1/input_dim*torch.sin(T-t+sum_x/input_dim)/(torch.cosh(sum_x/input_dim)**3)
    frest_i2 = 1/input_dim*torch.cos(T-t+sum_x/input_dim)/torch.cosh(sum_x/input_dim)*torch.tanh(sum_x/input_dim) - torch.sin(T-t+sum_x/input_dim)/(torch.cosh(sum_x/input_dim)**3)
    return (yr**2+yi**2)*yi + frest_i1 + frest_i2

def fun_Gr(x):
    sum_x = x.sum(dim=1, keepdim=True)  # Sum across `input_dim` (dim=1)
    y_r_T = torch.cos(sum_x/input_dim)/torch.cosh(sum_x/input_dim)
    return y_r_T
def fun_Gi(x):
    sum_x = x.sum(dim=1, keepdim=True)  # Sum across `input_dim` (dim=1)
    y_i_T = torch.sin(sum_x/input_dim)/torch.cosh(sum_x/input_dim)
    return y_i_T
def fun_dGr(x):
    sum_x = x.sum(dim=1, keepdim=True)  # Sum across `input_dim` (dim=1)
    return torch.cat([-torch.sin(sum_x/input_dim)/(input_dim*torch.cosh(sum_x/input_dim))-torch.cos(sum_x/input_dim)*torch.tanh(sum_x/input_dim)/torch.cosh(sum_x/input_dim)]*input_dim,dim=1)
def fun_dGi(x):
    sum_x = x.sum(dim=1, keepdim=True)  # Sum across `input_dim` (dim=1)
    return torch.cat([torch.cos(sum_x/input_dim)/(input_dim*torch.cosh(sum_x/input_dim))+torch.sin(sum_x/input_dim)*torch.tanh(sum_x/input_dim)/torch.cosh(sum_x/input_dim)]*input_dim,dim=1)

def true_solution_real(x1, current_step):
    return torch.cos(x1/input_dim + (N - current_step) * T / N)/torch.cosh(x1/input_dim)

def true_solution_img(x1, current_step):
    return torch.sin(x1/input_dim + (N - current_step) * T / N)/torch.cosh(x1/input_dim)