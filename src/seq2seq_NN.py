import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self,number_layers,input_size,input_channels,growth_factor,
                 batch_size,num_layers_memory_unit,hidden_size_memory_unit_1,
                 hidden_size_memory_unit_2, linear_layer_units_1,linear_layer_units_2,
                 Unique_character_list):
        super(NeuralNetwork, self).__init__()
        
        self.Unique_character_list = Unique_character_list
        self.hidden_size_memory_unit_2 = hidden_size_memory_unit_2
        self.hidden_size_memory_unit_1 = hidden_size_memory_unit_1
        self.num_layer_memory_unit = num_layers_memory_unit
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_channels = input_channels
        self.growth_factor  = growth_factor
        self.num_layers  = number_layers
        self.linear_layer_units_1 = linear_layer_units_1
        self.linear_layer_units_2 = linear_layer_units_2
        self.Model = nn.Sequential()

        # Looping to produce the basic sequential architecture
        for i in range(0,self.num_layers):
          if i == 0:
            self.Model.add_module("layer_"+str(i), torch.nn.Conv2d(self.input_channels,self.input_channels*self.growth_factor[i], kernel_size=5))
            last_layer_output = self.input_channels*self.growth_factor[i]
          else:
            self.Model.add_module("layer_"+str(i), torch.nn.Conv2d(last_layer_output,last_layer_output*self.growth_factor[i], kernel_size=5))
            last_layer_output = last_layer_output*self.growth_factor[i] 

        # Getting the total number of features after the convolution to be fed into the Linear layers
        #self.number_flat_features = self.calculate_flat_features () 
        #self.flatten = nn.Flatten()

        # As per sequence2sequence models there shall be
        # N length M sequences per each image/input
        # Every N length will be equal to = self.number_of_feature_axis
        # Every M number of sequences per image depend upon the convolution layer.



        self.number_of_features_axis = self.Model(torch.rand(self.input_size)).shape[3]

        self.linear_relu_stack = nn.Sequential(
            #nn.Linear(self.number_flat_features, 512),
            nn.Linear(self.number_of_features_axis, self.linear_layer_units_1),
            nn.ReLU(),
            nn.Linear(self.linear_layer_units_1, self.linear_layer_units_2),
            nn.ReLU()
            )
        # h_0: tensor of shape (D * \text{num\_layers}, N, H_{out})(D∗num_layers,N,H out) containing the initial hidden state for each element in the batch. Defaults to zeros if (h_0, c_0) is not provided.
        # c_0: tensor of shape (D * \text{num\_layers}, N, H_{cell})(D∗num_layers,N,H cell) containing the initial cell state for each element in the batch. Defaults to zeros if (h_0, c_0) is not provided.
        # Giving zeros at h_o, c_0 
        # Initializing the hidden states to be given at every iteration 

        self.hidden_state_1   = torch.zeros( 2*self.num_layer_memory_unit,self.batch_size,self.hidden_size_memory_unit_1 ).to(device)
        self.hidden_state_2   = torch.zeros( 2*self.num_layer_memory_unit,self.batch_size,self.hidden_size_memory_unit_2 ).to(device)


        self.memory_stack_1  = nn.LSTM( input_size=self.linear_layer_units_2, hidden_size=self.hidden_size_memory_unit_1 ,num_layers= self.num_layer_memory_unit, batch_first=True,bidirectional =True)
        self.memory_stack_2  = nn.LSTM( input_size=self.hidden_size_memory_unit_1*2, hidden_size=self.hidden_size_memory_unit_2  ,num_layers= self.num_layer_memory_unit, batch_first=True,bidirectional =True)

        self.classifier_stack = nn.Sequential(
            nn.Linear(self.hidden_size_memory_unit_2*2, len(self.Unique_character_list)+1)
        )
        

    def calculate_flat_features (self):
        return self.num_flat_features( self.Model(torch.rand(self.input_size)) )

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
          num_features *= s
        return num_features

    def forward(self, x):
        x = self.Model(x)
        x = x.reshape(self.batch_size, self.Model(torch.rand(self.input_size)).shape[1]*self.Model(torch.rand(self.input_size)).shape[2], self.number_of_features_axis)
        x = self.linear_relu_stack(x)
        x, _ = self.memory_stack_1(x,(self.hidden_state_1,self.hidden_state_1))
        x, _ = self.memory_stack_2(x,(self.hidden_state_2,self.hidden_state_2))
        # Stacking logic taken from
        # https://medium.com/swlh/multi-digit-sequence-recognition-with-crnn-and-ctc-loss-using-pytorch-framework-269a7aca2a6 
        x = torch.stack([F.log_softmax(self.classifier_stack(x[i]), dim=-1) for i in range(x.shape[0])])
        return x