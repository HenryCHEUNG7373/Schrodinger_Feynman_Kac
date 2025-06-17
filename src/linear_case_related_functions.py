import torch
import math
from src.config import input_dim, device
from src.config import T, input_dim, neuron, N, device,dt, t_space, batch_size, times_in_one_epochs, num_epochs_terminal, num_epochs_general, window_size

def funr(t,x,yr,yi):
    return torch.zeros(yr.shape,device=device)
def funi(t,x,yr,yi):
    return torch.zeros(yi.shape,device=device)
def fun_Gr(x):
    sum_x = x.sum(dim=1, keepdim=True)  
    y_r_T = torch.cos(sum_x*math.sqrt(2/input_dim))
    return y_r_T
def fun_Gi(x):
    sum_x = x.sum(dim=1, keepdim=True) 
    y_i_T = torch.sin(sum_x*math.sqrt(2/input_dim))
    return y_i_T
def fun_dGr(x):
    sum_x = x.sum(dim=1, keepdim=True)  
    return torch.cat([-torch.sin(sum_x*math.sqrt(2/input_dim))*math.sqrt(2/input_dim)]*input_dim,dim=1)
def fun_dGi(x):
    sum_x = x.sum(dim=1, keepdim=True) 
    return torch.cat([torch.cos(sum_x*math.sqrt(2/input_dim))*math.sqrt(2/input_dim)]*input_dim,dim=1)

def true_solution_real(x1, current_step):
    return torch.cos(math.sqrt(2/input_dim)*x1 - (N - current_step) * T / N)

def true_solution_img(x1, current_step):
    return torch.sin(math.sqrt(2/input_dim)*x1 - (N - current_step) * T / N)
