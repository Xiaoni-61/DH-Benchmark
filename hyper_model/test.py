import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm


class HyperMLP(nn.Module):
    """
    Integrated hypernetwork that generates parameters for its internal MLP and performs forward pass.
    """

    def __init__(self, args, n_usrs, usr_used, device, input_dim, hidden_dims, output_dim, spec_norm=False):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_users = n_usrs
        self.usr_used = usr_used
        self.device = device

        # Initialize control vector
        self.input_ray = Variable(torch.FloatTensor([[1 / len(usr_used) for _ in usr_used]])).to(device)
        self.input_ray.requires_grad = True

        # MLP structure to generate weights
        layer_dims = [len(usr_used)] + hidden_dims
        self.generators = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            layer = spectral_norm(nn.Linear(layer_dims[i], layer_dims[i + 1])) if spec_norm else nn.Linear(
                layer_dims[i], layer_dims[i + 1])
            self.generators.append(nn.Sequential(
                layer,
                nn.LeakyReLU(0.2, inplace=True)
            ))

        # Generators for weights and biases for the internal MLP
        self.weight_generators = nn.ModuleList()
        self.bias_generators = nn.ModuleList()
        last_dim = hidden_dims[-1]
        all_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(all_dims) - 1):
            self.weight_generators.append(nn.Linear(last_dim, all_dims[i] * all_dims[i + 1]))
            self.bias_generators.append(nn.Linear(last_dim, all_dims[i + 1]))

    def forward(self, x, input_ray=None):
        if input_ray is not None:
            self.input_ray.data = input_ray.to(self.device)

        # Generate weights and biases
        feature = self.input_ray
        for generator in self.generators:
            feature = generator(feature)

        weights = []
        biases = []
        for wg, bg in zip(self.weight_generators, self.bias_generators):
            weights.append(wg(feature))
            biases.append(bg(feature))

        # Apply weights and biases in the internal MLP
        current_input = x
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            # The weight needs to be reshaped to [output_dim, input_dim]
            # The last dimension calculation is based on current input and output required by the current layer
            out_dim = self.hidden_dims[i] if i < len(self.hidden_dims) else self.output_dim
            in_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            weight = weight.view(out_dim, in_dim)
            bias = bias.view(-1)
            current_input = F.linear(current_input, weight, bias)
            if i < len(weights) - 1:  # Apply ReLU for all but the last layer
                current_input = F.relu(current_input)

        return current_input


# Example usage
args = {}
n_usrs = 10
usr_used = range(n_usrs)
device = 'cuda'
input_dim = 784
hidden_dims = [100, 50]
output_dim = 10
spec_norm = True

hyper_mlp = HyperMLP(args, n_usrs, usr_used, device, input_dim, hidden_dims, output_dim, spec_norm)
hyper_mlp.to(device)
x = torch.randn(64, 784).to(device)  # Example batch of data
output = hyper_mlp(x)
pass