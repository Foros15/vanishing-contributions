import random
import math
import numpy as np
import torch
import torch.nn as nn
from transformers.models.vit.modeling_vit import ViTConfig, VIT_ATTENTION_CLASSES, ViTIntermediate, ViTOutput, Optional, Union, Tuple

# Base functions

def set_seed(seed: int):
    # Set the Python random seed
    random.seed(seed)
    # Set the NumPy random seed
    np.random.seed(seed)
    # Set the PyTorch random seed
    torch.manual_seed(seed)
    # If you are using GPUs, you also need to set the seed for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        
def generate_vit_paths(query: bool, key: bool, value: bool, attention_output: bool, intermediate: bool, output: bool, layers_number=12) -> list[str]:
    list_paths=[]
    if layers_number >= 0:
        if query:
            list_paths += [f'vit.encoder.layer.{i}.attention.attention.query' for i in range(layers_number)]
        if key:
            list_paths += [f'vit.encoder.layer.{i}.attention.attention.key' for i in range(layers_number)]
        if value:
            list_paths += [f'vit.encoder.layer.{i}.attention.attention.value' for i in range(layers_number)]
        if attention_output:
            list_paths += [f'vit.encoder.layer.{i}.attention.output.dense' for i in range(layers_number)]
        if intermediate:
            list_paths += [f'vit.encoder.layer.{i}.intermediate.dense' for i in range(layers_number)]
        if output:
            list_paths += [f'vit.encoder.layer.{i}.output.dense' for i in range(layers_number)]
    else:
        if query:
            list_paths += [f'vit.encoder.layer.{i}.attention.attention.query' for i in range(12+layers_number, 12)]
        if key:
            list_paths += [f'vit.encoder.layer.{i}.attention.attention.key' for i in range(12+layers_number, 12)]
        if value:
            list_paths += [f'vit.encoder.layer.{i}.attention.attention.value' for i in range(12+layers_number, 12)]
        if attention_output:
            list_paths += [f'vit.encoder.layer.{i}.attention.output.dense' for i in range(12+layers_number, 12)]
        if intermediate:
            list_paths += [f'vit.encoder.layer.{i}.intermediate.dense' for i in range(12+layers_number, 12)]
        if output:
            list_paths += [f'vit.encoder.layer.{i}.output.dense' for i in range(12+layers_number, 12)]
    
    return list_paths

def get_layers(model: nn.Module, layers_str: list[str], split_attributes=False) -> list[torch.tensor]:
    paths = [layer.split('.') for layer in layers_str]
    if split_attributes:
        attributes = [layer[-1] for layer in paths]
        paths = [layer[:-1] for layer in paths]

    layers_list = []
    for layer in paths:
        tmp_layer = model
        for sub_layer in layer:
            tmp_layer = getattr(tmp_layer, sub_layer)
        layers_list.append(tmp_layer)
    if split_attributes:
        return layers_list, attributes
    else:
        return layers_list
           

def combine_config(config_dict_list):
    combined_config = {}
    for config in config_dict_list.values():
        combined_config.update(config)
    return combined_config

def convert_string(value):
    # Handle boolean conversion
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    
    # Try integer conversion
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try float conversion
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string if no other conversion is valid
    return value

#LowRank Layer
class LowRankLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.vanishing = nn.Linear(in_features, out_features, bias=False)
        self.matA = nn.Linear(in_features, rank, bias=False)
        self.matB = nn.Linear(rank, out_features, bias=False)
        self.bias= nn.Parameter(torch.zeros(self.out_features))

        self.beta = 0.0

    def upd_beta(self,beta):
        self.beta=beta

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.beta>0:
            output = self.vanishing(input)*self.beta + self.matB(self.matA(input))*(1-self.beta)
        else:
            output= self.matB(self.matA(input))
        output+=self.bias
        return output
    
# Pruner class
class Pruner:
    def __init__(self, prune_layers, prune_granularity='layer', prune_ratio=0.0, lambda_w=0.0):
        self.prune_layers = prune_layers
        self.prune_granularity = prune_granularity
        self.prune_ratio = prune_ratio
        self.lambda_w = lambda_w
        self.stored_weights = []
        self.stored_masks = []
        self.old_weights = []
        self.old_masks = []

    def prune_model(self, prune_beta=0.0):
        # reset stored tensors
        self.old_weights = self.stored_weights
        self.old_masks = self.stored_masks
        self.stored_weights = []
        self.stored_masks = []
        # prune the weights
        for layer in self.prune_layers:
            w = layer.weight.data
            self.stored_weights.append(w)
            if self.prune_granularity == "layer":
                kept = int(round(w.numel()*(1-self.prune_ratio)))
                kth_largest = torch.topk(torch.abs(w.flatten()), kept).values[-1]
                mask = torch.ge(torch.abs(w), kth_largest)
            elif self.prune_granularity == "neuron":
                kept = int(round(w.shape[1]*(1-self.prune_ratio)))
                kth_largest = torch.topk(torch.abs(w), kept, dim=1).values[:, -1]
                mask = torch.ge(torch.abs(w), kth_largest.unsqueeze(1))
            elif type(self.prune_granularity) is int:
                kept = int(round(self.prune_granularity*(1-self.prune_ratio)))
                reshaped_w = w.reshape(-1, self.prune_granularity)
                kth_largest = torch.topk(torch.abs(reshaped_w), kept, dim=1).values[:, -1]
                mask = torch.ge(torch.abs(reshaped_w), kth_largest.unsqueeze(1)).reshape(w.shape)
            else:
                raise Exception("Invalid prune_granularity value:", self.prune_granularity)

            self.stored_masks.append(mask)
            layer.weight.data = prune_beta*w + (1-prune_beta)*w*mask

    def restore_model(self):
        for j, layer in enumerate(self.prune_layers):
            layer.weight.data = self.stored_weights[j]

    def regularize_SAD(self):
        # apply pruned masks to gradients
        for j, layer in enumerate(self.prune_layers):
            layer.weight.grad += self.stored_weights[j]*self.stored_masks[j]*self.lambda_w

    def reset_SAD(self):
        self.SAD_instances = 0
        self.SAD_sum = 0.0

    def update_SAD(self):
        for mask, old_mask in zip(self.stored_masks, self.old_masks):
            self.SAD_sum += torch.sum(torch.abs(mask.float()-old_mask.float())).detach().cpu().numpy()

    def get_SAD(self):
        if self.SAD_instances > 0:
            return self.SAD_sum/self.SAD_instances
        else:
            return 0.0
        
# Quantizer class
class Quantizer:
    def __init__(self, quantize_layers):
        self.quantize_layers = quantize_layers
        self.stored_weights = []

    def quantize_model(self, quantize_beta=0.0):
        self.stored_weights = []
        # quantize the weights
        for layer in self.quantize_layers:
            self.stored_weights.append(layer.weight.data)
            wq = torch.sign(layer.weight.data)*torch.mean(torch.abs(layer.weight.data))
            layer.weight.data = quantize_beta*layer.weight.data + (1-quantize_beta)*wq

    def restore_model(self):
        for j, layer in enumerate(self.quantize_layers):
            layer.weight.data = self.stored_weights[j]
        

# LowRankDecomposition class
class LRDer:
    def __init__(self,model, lrd_layers_str, rank):
        self.rank=rank

        layers_list, attributes=get_layers(model=model, layers_str=lrd_layers_str, split_attributes=True)
        self.lrd_layers=[]
        for layer, attr in zip(layers_list, attributes):
            layer_attr = getattr(layer, attr)
            tmp_weight = layer_attr.weight.data
            tmp_bias = layer_attr.bias.data
            U, S, V = torch.linalg.svd(tmp_weight, full_matrices=False)
            U_r = U[:, :rank]        
            S_r = S[:rank]           
            V_r = V[:rank, :]  
            U_S = U_r @ torch.diag(S_r) 
            van = LowRankLayer(layer_attr.in_features, layer_attr.out_features, rank)
            van.vanishing.weight.data= tmp_weight
            van.bias.data= tmp_bias
            van.matA.weight.data= V_r
            van.matB.weight.data= U_S
            
            setattr(layer, attr, van)
            self.lrd_layers.append(van)

        self.lrd_beta=0

    def upd_beta(self, lrd_beta=0.0):
        self.lrd_beta=lrd_beta
        for l in self.lrd_layers:
            l.upd_beta(lrd_beta)
        
        