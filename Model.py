import torch 
import scipy.io
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

class PhysicsINN(nn.Module):
    
    '''
    Physics Informed Neural Network
    Written: Siddesh Sambasivam Suseela
    
    '''
    
    def __init__(self, num_layers:int=2, num_neurons:int=20) -> None:
        
        super(PhysicsINN, self).__init__()
        
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        
        # Each hidden layer contained 20 neurons and a hyperbolic tangent activation function.
        self.activation_func = torch.nn.Tanh
        
        ordered_layers = list()
                
        ordered_layers.append(("input_layer", nn.Linear(2, self.num_neurons)))
        ordered_layers.append(("input_activation", self.activation_func()))
        
        # Create num_layers-2 linear layers with num_neuron neurons and tanh activation function
        for i in range(self.num_layers-2):
            
            ordered_layers.append(("layer_%d" % (i+1), nn.Linear(self.num_neurons, self.num_neurons)))
            ordered_layers.append(("layer_%d_activation" % (i+1), self.activation_func()))
                        
        ordered_layers.append(("output_layer", nn.Linear(self.num_neurons, 1)))
        
        self.net = nn.Sequential(OrderedDict(ordered_layers))    
        
        self.init_weights()
        
    def init_weights(self, ) -> None:
        """
        Initializes the weights and biases of all the layers in the model
        
        NOTE: According to the paper, the model's weights are initialized by xaviers' distribution
        and biases are initialized as zeros
        
        """
        for param in self.parameters():
            if len(param.shape) >= 2: torch.nn.init.xavier_normal_(param, )
            elif len(param.shape) == 1: torch.nn.init.zeros_(param)

        
    def forward(self, inputs) -> torch.Tensor:
        '''returns the output from the model'''
        
        out = self.net(inputs)

        return out 
