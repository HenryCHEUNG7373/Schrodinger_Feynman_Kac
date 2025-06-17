import torch
from src.config import T, input_dim, neuron, N, device,dt, t_space, batch_size, times_in_one_epochs, num_epochs_terminal, num_epochs_general, window_size
import matplotlib.pyplot as plt

#Real ramdom number generator
def draw_X_and_dW(batch_size, x, input_dim, dt):
    dW = torch.normal(mean=0, std=torch.sqrt(dt), size=(batch_size, input_dim), device=device)
    X = torch.zeros(batch_size, input_dim, 2,device=device)
    X[:,:,0] = x
    X[:,:,1] = X[:,:,0] + 1*dW
    return X,dW

def save_models(y,y_bar,z,z_bar):
    '''
    Save the model states to disk.
    Args:
        y (nn.Module): The model for y.
        y_bar (nn.Module): The model for y_bar.
        z (nn.Module): The model for z.
        z_bar (nn.Module): The model for z_bar.
    '''
    torch.save(y.state_dict(), "y.pth")
    torch.save(y_bar.state_dict(), "y_bar.pth")
    torch.save(z.state_dict(), "z.pth")
    torch.save(z_bar.state_dict(), "z_bar.pth")

def load_models(y, y_bar, z, z_bar):
    '''
    Load the model states from disk.
    Args:
        y (nn.Module): The model for y.
        y_bar (nn.Module): The model for y_bar.
        z (nn.Module): The model for z.
        z_bar (nn.Module): The model for z_bar.
    '''
    y.load_state_dict(torch.load('y.pth', map_location=torch.device('cpu')))
    y_bar.load_state_dict(torch.load('y_bar.pth', map_location=torch.device('cpu')))
    z.load_state_dict(torch.load('z.pth', map_location=torch.device('cpu')))
    z_bar.load_state_dict(torch.load('z_bar.pth', map_location=torch.device('cpu')))


def plot_graph_varies_one_axis(y,current_step, window_size, axis, true_solution, y_lower_bd,y_upper_bd):
        '''
        Plot the model's output varying one axis while keeping others constant.
        Args:
            y (nn.Module): The model to evaluate.
            current_step (int): The current time step.
            window_size (float): The range for the varying axis.
            axis (int): The index of the axis to vary (0 for x1, 1 for x2, etc.).
            true_solution (callable): The true solution function to compare against.
            y_lower_bd (float): Lower bound for y-axis.
            y_upper_bd (float): Upper bound for y-axis.
        '''
        # Create a grid of points with x1 varying and other dimensions set to 0
        x1 = torch.linspace(-window_size, window_size, 200, device=device)
        x = torch.zeros(200, input_dim, device=device)
        x[:, axis] = x1  

        # Evaluate the model
        y.eval()
        with torch.no_grad():  
            eval_y = y(x, current_step) 
    
        # Compute the true solution (same as 1D case, depends only on x1)
        j = true_solution(x1, current_step)
        
        #j = torch.cos(math.sqrt(2/input_dim)*x1 - (N - current_step) * T / N)
        
        # Move tensors to CPU and convert to NumPy for plotting
        x1 = x1.cpu().detach().numpy()
        eval_y = eval_y.cpu().detach().numpy()
        j = j.cpu().detach().numpy()
    
        # Create the plot
        plt.figure(figsize=(8, 4))  # Set the figure size
        plt.plot(x1, eval_y, label='predicted') 
        plt.plot(x1, j, label='true')  
    
        # Add title and labels
        plt.title('Difference between Predicted and True Solution')
        plt.xlabel('x1')
        plt.ylabel('Value')
        
        # Add a legend
        plt.legend()
        plt.ylim(y_lower_bd,y_upper_bd)
        # Save the plot
        filename = f"time_step_{current_step}.png"
        plt.savefig(filename)
        plt.close()  # Close the figure to free memory
        print(f"Plot saved as {filename}")