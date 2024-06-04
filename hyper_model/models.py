from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from torch.autograd import Variable

import pdb



class LocalOutput(nn.Module):
    '''
    output layer module
    '''
    def __init__(self, n_input=84, n_output=2, nonlinearity=False):
        super().__init__()
        self.nonlinearity = nonlinearity
        layers = []
        if nonlinearity:
            layers.append(nn.ReLU())

        layers.append(nn.Linear(n_input, n_output))
        layers.append(nn.Softmax(dim=1))
        self.layer = nn.Sequential(*layers)
        self.loss_CE = nn.CrossEntropyLoss()
    def forward(self, x, y):
        pred = self.layer(x)
        loss = self.loss_CE(pred, y)
        return pred, loss 



class HyperSimpleNet(nn.Module):
    '''
    hypersimplenet for adult and synthetic experiments
    '''
    def __init__(self, args, device, user_used):
        super(HyperSimpleNet, self).__init__()
        self.n_users = args.num_users
        # usr_used = [i for i in range(self.n_users)]
        self.usr_used = user_used
        self.device = device
        self.dataset = args.dataset
        hidden_dim = args.hidden_dim
        self.hidden_dim = hidden_dim
        self.train_pt = args.train_pt
        self.input_ray = Variable(torch.FloatTensor([[1/len(self.usr_used) for i in self.usr_used]])).to(device)
        self.init_ray(0)

        self.input_ray.requires_grad = True
        spec_norm = args.spec_norm
        layers = [spectral_norm(nn.Linear(len(self.usr_used), hidden_dim)) if spec_norm else nn.Linear(len(self.usr_used), hidden_dim)]
        for _ in range(args.n_hidden - 1):
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)
        self.input_dim = args.input_dim

        if self.dataset == "synthetic1" or self.dataset == "synthetic2":
            self.loss = nn.MSELoss(reduction='mean')
            self.output_dim = 1
            self.l1_weights = nn.Linear(hidden_dim, self.input_dim*100)
            self.l1_bias = nn.Linear(hidden_dim, 1)
            self.l2_weights = nn.Linear(hidden_dim, self.output_dim*100)
            self.l2_bias = nn.Linear(hidden_dim, 1)






    def forward(self, x, y, usr_id,  input_ray = None):
        if input_ray!=None:
            self.input_ray.data = input_ray
        feature = self.mlp(self.input_ray)
        l1_weight = self.l1_weights(feature).view(100, self.input_dim)
        l1_bias = self.l1_bias(feature).view(-1)
        l2_weight = self.l2_weights(feature).view(self.output_dim, 100)
        l2_bias = self.l2_bias(feature).view(-1)
        x = F.leaky_relu( F.linear(x, weight=l1_weight, bias=l1_bias),0.2)
        x = F.linear(x, weight=l2_weight, bias=l2_bias)
        if self.dataset == "synthetic1" or self.dataset == "synthetic2":
            pred = x.flatten()

        y = y.float()
        loss = self.loss(pred,y)
        return pred, loss

    def init_ray(self,target_usr):

        if self.dataset == "synthetic1":
            big_usr = []
            big_usr_idx = []


            for usr in self.usr_used:
                if usr in [0, 1, 4, 5]:
                    big_usr.append(usr)
                    big_usr_idx.append(self.usr_used.index(usr))

            if self.train_pt == True and len(big_usr) != 0:
                if (len(self.input_ray.data.shape) == 1):
                    for i in range(len(self.usr_used)):
                        if i in big_usr_idx :
                            self.input_ray.data[i] = 1 / len(big_usr)
                        else:
                            self.input_ray.data[i] = 0

                elif (len(self.input_ray.data.shape) == 2):
                    for i in range(len(self.usr_used)):
                        if i in big_usr_idx:
                            self.input_ray.data[0, i] = 1 / len(big_usr)
                        else:
                            self.input_ray.data[0, i] = 0

            elif self.train_pt == True and len(big_usr) == 0:
                if (len(self.input_ray.data.shape) == 1):
                    for i in range(len(self.usr_used)):
                        if i == target_usr:
                            self.input_ray.data[i] = 1
                        else:
                            self.input_ray.data[i] = 0

                elif (len(self.input_ray.data.shape) == 2):
                    for i in range(len(self.usr_used)):
                        if i == target_usr:
                            self.input_ray.data[0, i] = 1
                        else:
                            self.input_ray.data[0, i] = 0
            else:
                self.input_ray.data.fill_(1 / len(self.usr_used))

        elif self.dataset == "synthetic2":
            self.input_ray.data.fill_(1 / len(self.usr_used))




class SimpleNet(nn.Module):

    def __init__(self,args, device):
        super().__init__()
        hidden_dim = args.hidden_dim
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100,1)

        self.n_users = args.num_users
        usr_used = [i for i in range(self.n_users)]
        self.usr_used = usr_used
        self.device = device
        self.dataset = args.dataset

        if self.dataset == "synthetic1" or self.dataset == "synthetic2":
            self.loss = nn.MSELoss(reduction='mean')
            self.output_dim = 1


    def forward(self, x, y, usr_id, input_tay = None):

        x.view(-1,20)
        x = F.leaky_relu(self.fc1(x),0.2)
        x = self.fc2(x)
        if self.dataset == "synthetic1" or self.dataset == "synthetic2":
            pred = x.flatten()

        y = y.float()
        loss = self.loss(pred,y)
        return pred, loss


class HyperMLP(nn.Module):
    """
    Integrated hypernetwork that generates parameters for its internal MLP and performs forward pass.
    """

    def __init__(self, args, n_usrs, usr_used, device, n_classes, input_dim, hidden_dims, output_dim, spec_norm=False):
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

        self.locals = nn.ModuleList([LocalOutput(n_output=n_classes) for i in range(self.n_users)])

    def forward(self, x, y, usr_id, input_ray=None):
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

        loss = nn.CrossEntropyLoss()(current_input, y.long())
        return current_input, loss

    def init_ray(self, target_usr):

        if self.args.train_pt == True:
            self.input_ray.data.fill_(1 / len(self.usr_used))
        else:

            if (len(self.input_ray.data.shape) == 1):
                if (len(self.usr_used) == 1):
                    self.input_ray.data[0] = 1.0
                else:
                    for i in range(len(self.usr_used)):
                        if i == target_usr:
                            self.input_ray.data[i] = 0.8
                        else:
                            self.input_ray.data[i] = (1.0 - 0.8) / (len(self.usr_used) - 1)
            elif (len(self.input_ray.data.shape) == 2):
                if (len(self.usr_used) == 1):
                    self.input_ray.data[0, 0] = 1.0
                else:
                    for i in range(len(self.usr_used)):
                        if i == target_usr:
                            self.input_ray.data[0, i] = 0.8
                        else:
                            self.input_ray.data[0, i] = (1.0 - 0.8) / (len(self.usr_used) - 1)

class Hypernet(nn.Module):
    '''
    Hypernet for CIFAR10 experiments
    '''
    def __init__(self, args, n_usrs, usr_used,  device, n_classes = 10,  in_channels=3, n_kernels=16, hidden_dim=100,spec_norm=False, n_hidden = 2):
        super().__init__()
        self.args = args
        self.in_channels = in_channels
        self.n_kernels = n_kernels
        if args.dataset in ('mnist', 'femnist', 'fmnist'):
            final_dim = 120 * 2 * self.n_kernels * 4 * 4
            self.in_channels = 1
        if args.dataset in ('cifar10', 'svhn'):
            final_dim = 120 * 2 * self.n_kernels * 5 * 5
            self.in_channels = 3
        self.n_classes = n_classes
        self.n_users = n_usrs
        self.usr_used = usr_used
        self.device = device

        self.input_ray = Variable(torch.FloatTensor([[1/len(usr_used) for i in usr_used]])).to(device)
        self.input_ray.requires_grad = True

        layers = [spectral_norm(nn.Linear(len(usr_used), hidden_dim)) if spec_norm else nn.Linear(len(usr_used), hidden_dim)]

        for _ in range(2):
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)


        self.c1_weights = []
        self.c1_bias = []
        self.c2_weights = []
        self.c2_bias = []
        self.l1_weights = []
        self.l1_bias = []
        self.l2_weights = []
        self.l2_bias = []
        for _ in range(n_hidden-1):
            self.c1_weights.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c1_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.c1_bias.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c1_bias.append(nn.LeakyReLU(0.2, inplace=True))
            self.c2_weights.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c2_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.c2_bias.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c2_bias.append(nn.LeakyReLU(0.2, inplace=True))
            self.l1_weights.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l1_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.l1_bias.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l1_bias.append(nn.LeakyReLU(0.2, inplace=True))
            self.l2_weights.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l2_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.l2_bias.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l2_bias.append(nn.LeakyReLU(0.2, inplace=True))


        self.c1_weights = nn.Sequential( *(self.c1_weights + [spectral_norm(nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)) if spec_norm else nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)] ))
        self.c1_bias = nn.Sequential( *(self.c1_bias + [spectral_norm(nn.Linear(hidden_dim, self.n_kernels)) if spec_norm else nn.Linear(hidden_dim, self.n_kernels)] )) 
        self.c2_weights = nn.Sequential( *(self.c2_weights + [spectral_norm(nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)) if spec_norm else nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)] )) 
        self.c2_bias = nn.Sequential( *(self.c2_bias + [spectral_norm(nn.Linear(hidden_dim, 2 * self.n_kernels)) if spec_norm else nn.Linear(hidden_dim, 2 * self.n_kernels)] ))  




        self.l1_weights = nn.Sequential( *(self.l1_weights + [spectral_norm(nn.Linear(hidden_dim, final_dim)) if spec_norm else nn.Linear(hidden_dim, final_dim)] ))
        self.l1_bias = nn.Sequential( *(self.l1_bias + [spectral_norm(nn.Linear(hidden_dim, 120)) if spec_norm else nn.Linear(hidden_dim, 120)] )) 
        self.l2_weights = nn.Sequential( *(self.l2_weights + [spectral_norm(nn.Linear(hidden_dim, 84 * 120)) if spec_norm else nn.Linear(hidden_dim, 84 * 120)] )) 
        self.l2_bias =  nn.Sequential( *(self.l2_bias + [spectral_norm(nn.Linear(hidden_dim, 84)) if spec_norm else nn.Linear(hidden_dim, 84)] ))

        self.locals = nn.ModuleList([LocalOutput(n_output =n_classes) for i in range(self.n_users)])
    
    def forward(self, x, y, usr_id, input_ray=None):
        if input_ray != None:
            self.input_ray.data = input_ray.to(self.device)

        feature = self.mlp(self.input_ray) 

        weights = {
            "conv1.weight": self.c1_weights(feature).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(feature).view(-1),
            "conv2.weight": self.c2_weights(feature).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(feature).view(-1),
            "fc1.weight": self.l1_weights(feature).view(120, -1),
            # 2 * self.n_kernels * 5 * 5
            "fc1.bias": self.l1_bias(feature).view(-1),
            "fc2.weight": self.l2_weights(feature).view(84, 120),
            "fc2.bias": self.l2_bias(feature).view(-1),
        }
        x = F.conv2d( x, weight=weights['conv1.weight'], bias=weights['conv1.bias'], stride=1)
        x = F.max_pool2d(x, 2)
        x = F.conv2d( x, weight=weights['conv2.weight'], bias=weights['conv2.bias'], stride=1)
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(F.linear(x, weight=weights["fc1.weight"], bias=weights["fc1.bias"]), 0.2) 
        logits = F.leaky_relu(F.linear(x, weight=weights["fc2.weight"], bias=weights["fc2.bias"]), 0.2) 

        pred, loss = self.locals[usr_id](logits, y.long())

        return pred, loss

    def init_ray(self, target_usr):

        if self.args.train_pt == True:
            self.input_ray.data.fill_(1 / len(self.usr_used))
        else:

            if (len(self.input_ray.data.shape) == 1):
                if (len(self.usr_used) == 1):
                    self.input_ray.data[0] = 1.0
                else:
                    for i in range(len(self.usr_used)):
                        if i == target_usr:
                            self.input_ray.data[i] = 0.8
                        else:
                            self.input_ray.data[i] = (1.0 - 0.8) / (len(self.usr_used) - 1)
            elif (len(self.input_ray.data.shape) == 2):
                if (len(self.usr_used) == 1):
                    self.input_ray.data[0, 0] = 1.0
                else:
                    for i in range(len(self.usr_used)):
                        if i == target_usr:
                            self.input_ray.data[0, i] = 0.8
                        else:
                            self.input_ray.data[0, i] = (1.0 - 0.8) / (len(self.usr_used) - 1)






class Basenet_cifar(nn.Module):
    def __init__(self, n_usrs, usr_used,  device, n_classes = 10,  in_channels=3, n_kernels=16):
        super().__init__()
        self.in_channels = in_channels
        self.n_kernels = n_kernels
        self.n_classes = n_classes
        self.n_users = n_usrs
        self.usr_used = usr_used
        self.device = device

        self.conv1 = nn.Conv2d(self.in_channels, self.n_kernels, 5)
        self.conv2 = nn.Conv2d(self.n_kernels, 2 * self.n_kernels, 5)
        self.fc1 = nn.Linear(2 * self.n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()




    def forward(self,x,y,usr_id):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x),0.2)
        x = F.leaky_relu(self.fc2(x),0.2)
        pred = self.softmax(self.fc3(x))
        loss = self.loss(pred,y)
        return pred, loss



