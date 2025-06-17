import torch
import torch.nn as nn

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class CosActivation(nn.Module):
    def forward(self, x):
        return torch.cos(x)
    
#We define the class of model y here, the output is 1 dimensional 
class BSDEModel_y_series(nn.Module):
    def __init__(self, N, input_dim, neuron):
        super().__init__()

        def make_subnetwork():
            # Shared structure for both branches
            def make_branch_tanh():
                return nn.Sequential(
                    nn.Linear(input_dim, input_dim + neuron, bias=True),
                    nn.Softplus(),
                    nn.Linear(input_dim + neuron, input_dim + neuron, bias=True),
                    nn.Tanh()
                )

            def make_branch_sin():
                return nn.Sequential(
                    nn.Linear(input_dim, input_dim + neuron, bias=True),
                    nn.Softplus(),
                    nn.Linear(input_dim + neuron, input_dim + neuron, bias=True),
                    SinActivation()
                )

            branch1 = make_branch_tanh()
            branch2 = make_branch_sin()

            # Fusion layer: takes concatenated output of both branches
            fusion = nn.Linear(1 * (input_dim + neuron), 1, bias=True)
            gate_layer = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid() 
            )

            return nn.ModuleDict({
                'branch1': branch1,
                'branch2': branch2,
                'fusion': fusion,
                'gate_layer': gate_layer
            })

        self.y = nn.ModuleList([make_subnetwork() for _ in range(N)])

    def forward(self, x, t_idx):
        sub = self.y[t_idx]
        out1 = sub['branch1'](x)  
        out2 = sub['branch2'](x)  
        gate = sub['gate_layer'](x)

        # Concatenate along feature dim
        # Gating scalar in [0, 1]
        # Soft interpolation
        fused = gate * out1 + (1 - gate) * out2  

        # Pass to fusion layer
        out = sub['fusion'](fused)  

        # Final fusion linear layer
        return out

class BSDEModel_z_series(nn.Module):
    def __init__(self, N, input_dim, neuron):
        super().__init__()

        def make_subnetwork():
            # Shared structure for both branches
            def make_branch_tanh():
                return nn.Sequential(
                    nn.Linear(input_dim, input_dim*neuron, bias=True),
                    nn.Softplus(),
                    nn.Linear(input_dim*neuron, input_dim*neuron, bias=True),
                    nn.Tanh()
                )

            def make_branch_sin():
                return nn.Sequential(
                    nn.Linear(input_dim, input_dim*neuron, bias=True),
                    nn.Softplus(),
                    nn.Linear(input_dim*neuron, input_dim*neuron, bias=True),
                    SinActivation()
                )

            branch1 = make_branch_tanh()
            branch2 = make_branch_sin()
            gate_layer = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid()  
            )
            # Fusion layer: takes concatenated output of both branches
            fusion = nn.Linear(1 * (input_dim*neuron), input_dim, bias=True)

            return nn.ModuleDict({
                'branch1': branch1,
                'branch2': branch2,
                'fusion': fusion,
                'gate_layer': gate_layer
            })

        self.z = nn.ModuleList([make_subnetwork() for _ in range(N)])

    def forward(self, x, t_idx):
        sub = self.z[t_idx]
        out1 = sub['branch1'](x)  
        out2 = sub['branch2'](x)  
        gate = sub['gate_layer'](x)

        # Concatenate along feature dim
        # Gating scalar in [0, 1]
        # Soft interpolation
        fused = gate * out1 + (1 - gate) * out2  
        # Concatenate along feature dim
        # Gating scalar in [0, 1]
        # Pass to fusion layer
        out = sub['fusion'](fused)

        # Final fusion linear layer
        return out
