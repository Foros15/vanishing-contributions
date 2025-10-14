#!/usr/bin/env python -u
# coding: utf-8

import os
import configparser
import argparse
from datetime import datetime
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.transforms import v2

from transformers import ViTForImageClassification

from utils import set_seed, generate_vit_paths, get_layers, combine_config, convert_string, Pruner, Quantizer, LRDer

# Default configuration
config = configparser.ConfigParser()
config['System-setup'] = {
    'num_workers': 36,
    'seed': 42,
}
config['Model'] = {
    'model_name': "model",
    'load_model': "model_dense", # pre-trained model name
    'teacher_model': "", # teacher model name
    'from_teacher': False,
    'vit_model': 'WinKawaks/vit-small-patch16-224',
}
config['Task'] = {
    'dataset': 'cifar10',
    'dataset_root': '.',
}
config['Knowledge distillation'] = {
    'kd_loss': 'cosine',
    'query_to_kd': False,
    'key_to_kd': False,
    'value_to_kd': False,
    'attention_output_to_kd': False,
    'intermediate_to_kd': False,
    'output_to_kd': False,
}
config['Online RS-STE pruning'] = {
    'prune_query': False,
    'prune_key': False,
    'prune_value': False,
    'prune_attention_output': False,
    'prune_intermediate': False,
    'prune_output': False,
    'prune_blocks_num': 12,
    'prune_ratio': 0, # layer-wise pruning ratio
    'prune_granularity': 'layer', # "layer", "neuron", or value 
    'prune_lambda_w': 0.0,#4e-4,
    'prune_vcon_epochs': 12,
}
config['Online STE quantization'] = {
    'quantize_query': False,
    'quantize_key': False,
    'quantize_value': False,
    'quantize_attention_output': False,
    'quantize_intermediate': False,
    'quantize_output': False,
    'quantize_blocks_num': 12,
    'quantize_vcon_epochs': 12,
}
config['LRD'] = {
    'lrd_query': False,
    'lrd_key': False,
    'lrd_value': False,
    'lrd_attention_output': False,
    'lrd_intermediate': False,
    'lrd_output': False,
    'lrd_blocks_num': 12,
    'lrd_vcon_epochs': 12,
    'lrd_rank': 16,
}
config['Freezed layers'] = {
    'freezed_blocks_num': 0,
    'freeze_patch_embeddings': False,
    'freeze_query': True,
    'freeze_key': True,
    'freeze_value': True,
    'freeze_attention_output': True,
    'freeze_intermediate': True,
    'freeze_output': True,
}
config['Batch and epochs'] = {
    'batch_size': 128,
    'batch_accumulation': 1,
    'starting_epoch': 0,
    'total_epochs': 60,
}
config['Learning rate scheduling'] = {
    'learning_rate': 1e-4,
    'lr_decay_factor': 1.0,
    'lr_patience': 10000,
    'lr_minimum': 1e-6,
    'lr_warmup_epochs': 1,
    'lr_warmup_start': 1e-5,
    'lr_cosine_decay': True
}
config['Regularizations'] = {
    'weight_decay': 0,
    'clipping_global_norm': False,
    'wsatstd': '', # weight magnitude saturation
    'attention_dropout_prob': 0,
    'output_dropout_prob': 0,
}
config['Dataugmentation'] = {
    'use_cutmix_mixup': True,
    'use_mixup_only': True,
    'mixup_alpha': 1.0,
    'use_autoaugment': False,
    'use_randaugment': True,
}
config['Metrics'] = {
    'store_attention_output_norm': False,
    'store_output_norm': False,
    'store_attention_input_gradient_norm': False,
    'store_intermediate_input_gradient_norm': False,
}

# Get from argparse configuration variables
parser = argparse.ArgumentParser(description="Update config with command-line arguments.")
for key in config:
    for subkey in config[key]:
        parser.add_argument(f"--{subkey}", type=str, help=f"Set the value for {subkey}")

parser.add_argument("-c", "--config", action='store_true', help=f"Get config file together with loaded model")
parser.add_argument("-f", "--force", action='store_true', help=f"Force overwrite of the config file")
parser.add_argument("-t", "--test", action='store_true', help=f"Perform only a run on the test dataset")

args = parser.parse_args()

# Load from the [optional] specified config file (and overwrite default config)
if args.config:
    if os.path.exists(args.load_model):
        loaded_config = configparser.ConfigParser()
        if os.path.isdir(args.load_model):
            loaded_config.read(os.path.join(args.load_model, 'config.ini'))
            print(f"Config loaded from {os.path.join(args.load_model, 'config.ini')}.")
        else:
            loaded_config.read(args.load_model)
            print(f"Config loaded from {args.load_model}.")

        # Update config with loaded_config
        for key in loaded_config:
            for subkey in loaded_config[key]:
                if key in config and subkey in config[key]:
                    value = loaded_config[key][subkey]
                    if value is not None:  # Only update if a value was provided
                        config[key][subkey] = value
    else:
        print(f"Config file {args.load_model} does not exist.")

# Update config with args
for key in config:
    for subkey in config[key]:
        value = getattr(args, subkey)  # Get the argument value
        if value is not None:  # Only update if a value was provided
            config[key][subkey] = value

# Printout model configuration
print("-----------------------")
print("Model configuration")
for key in config:
    print(f"{key}:")
    for subkey in config[key]:
        print(f"    {subkey}: {config[key][subkey]}")
print("-----------------------")

# File paths
if not args.test:
    model_name = config["Model"]["model_name"]
    os.makedirs(model_name, exist_ok=True)
    config_path = os.path.join(model_name, 'config.ini')
    model_path = os.path.join(model_name, 'model.pt')
    log_path = os.path.join(model_name, 'log.txt')
    def backup_path(epoch):
        return os.path.join(model_name, f'backup_e{epoch}.pt')
    tb_path = model_name

    # Store configuration in the model folder

    write_option = 'w' if args.force else 'x'
    with open(config_path, write_option) as configfile:
        config.write(configfile)

# Extract configuration to variables

config = SimpleNamespace(**combine_config(config))
for c in vars(config):
    setattr(config, c, convert_string(getattr(config, c)))

# Set seed for execution

set_seed(config.seed)

# Define transformations for the training and test sets

if config.dataset == 'cifar10':
    transform_mean = (0.49139968, 0.48215827, 0.44653124)
    transform_std = (0.24703233, 0.24348505, 0.26158768)
elif config.dataset == 'cifar100':
    transform_mean = (0.5074,0.4867,0.4411)
    transform_std = (0.2011,0.1987,0.2025)
elif config.dataset == 'imagenet1k':
    transform_mean = (0.485, 0.456, 0.406)
    transform_std = (0.229, 0.224, 0.225)

transform_train_list = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224)),
]

if config.use_autoaugment:
    transform_train_list.append(v2.AutoAugment())
if config.use_randaugment:
    transform_train_list.append(v2.RandAugment())
transform_train_list.append(v2.Normalize(transform_mean, transform_std))

transform_train = v2.Compose(transform_train_list)

transform_test = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224)),
    v2.Normalize(transform_mean, transform_std),
])

if config.dataset == 'cifar10':
    NUM_CLASSES = 10
    # Download CIFAR-10 dataset and apply transformations
    trainset = torchvision.datasets.CIFAR10(root=config.dataset_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=config.dataset_root, train=False, download=True, transform=transform_test)
elif config.dataset == 'cifar100':
    NUM_CLASSES = 100
    # Download CIFAR-100 dataset and apply transformations
    trainset = torchvision.datasets.CIFAR100(root=config.dataset_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=config.dataset_root, train=False, download=True, transform=transform_test)
elif config.dataset == 'imagenet1k':
    NUM_CLASSES = 1000
    # Download ImageNet1k dataset and apply transformations
    trainset = torchvision.datasets.ImageNet(root=config.dataset_root, split='train', transform=transform_train)
    testset = torchvision.datasets.ImageNet(root=config.dataset_root, split='val', transform=transform_test)

valset, testset = torch.utils.data.random_split(testset, [0.5, 0.5], generator=torch.Generator().manual_seed(42))    

trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
print("training set length:", len(trainloader))

valloader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

print("validation set length:", len(valloader))
print("test set length:", len(testloader))

# Cutmix and Mixup data augmentation

mixup = v2.MixUp(alpha=config.mixup_alpha, num_classes=NUM_CLASSES)
if config.use_mixup_only:
    cutmix_or_mixup = mixup
else:
    cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

# Instantiate ViT model
model_checkpoint = config.vit_model
model = ViTForImageClassification.from_pretrained(
    model_checkpoint,
    ignore_mismatched_sizes=True,
    num_labels=NUM_CLASSES,
)

# instantiate teacher model
if config.from_teacher:
    teacher = copy.deepcopy(model)
    teacher.load_state_dict(torch.load(config.teacher_model, weights_only=True))




if config.load_model != "":
    if os.path.isdir(config.load_model):
        model.load_state_dict(torch.load(os.path.join(config.load_model, "model.pt"), weights_only=True))
        print("Loaded model from", os.path.join(config.load_model, "model.pt"))
    else:
        model.load_state_dict(torch.load(config.load_model, weights_only=True))
        print("Loaded model from", config.load_model) 

# get layers to online rs-ste prune
prune_layers_str = generate_vit_paths(config.prune_query,
                                      config.prune_key,
                                      config.prune_value,
                                      config.prune_attention_output, 
                                      config.prune_intermediate,
                                      config.prune_output,
                                      layers_number=config.prune_blocks_num)
prune_layers = get_layers(model, prune_layers_str)

# get layers to online ste quantize
quantize_layers_str = generate_vit_paths(config.quantize_query,
                                      config.quantize_key,
                                      config.quantize_value,
                                      config.quantize_attention_output, 
                                      config.quantize_intermediate,
                                      config.quantize_output,
                                      layers_number=config.quantize_blocks_num)
quantize_layers = get_layers(model, quantize_layers_str)

# get layers to lrd
lrd_layers_str = generate_vit_paths(config.lrd_query,
                                      config.lrd_key,
                                      config.lrd_value,
                                      config.lrd_attention_output, 
                                      config.lrd_intermediate,
                                      config.lrd_output,
                                      layers_number=config.lrd_blocks_num)
lowranker= LRDer(model, lrd_layers_str , config.lrd_rank)



# put models on GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
if config.from_teacher:
    teacher.to(device)

# define forward hooks
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

if config.from_teacher:
    for i in range(12):
        for i in range(12):
            if config.query_to_kd:
                teacher.vit.encoder.layer[i].attention.attention.query.register_forward_hook(get_activation(f'teacher.query{i}'))
                model.vit.encoder.layer[i].attention.attention.query.register_forward_hook(get_activation(f'student.query{i}'))
            if config.key_to_kd:
                teacher.vit.encoder.layer[i].attention.attention.key.register_forward_hook(get_activation(f'teacher.key{i}'))
                model.vit.encoder.layer[i].attention.attention.key.register_forward_hook(get_activation(f'student.key{i}'))
            if config.value_to_kd:
                teacher.vit.encoder.layer[i].attention.attention.value.register_forward_hook(get_activation(f'teacher.value{i}'))
                model.vit.encoder.layer[i].attention.attention.value.register_forward_hook(get_activation(f'student.value{i}'))
            if config.attention_output_to_kd:
                teacher.vit.encoder.layer[i].attention.output.dense.register_forward_hook(get_activation(f'teacher.attention_output{i}'))
                model.vit.encoder.layer[i].attention.output.dense.register_forward_hook(get_activation(f'student.attention_output{i}'))
            if config.intermediate_to_kd:
                teacher.vit.encoder.layer[i].intermediate.dense.register_forward_hook(get_activation(f'teacher.intermediate{i}'))
                model.vit.encoder.layer[i].intermediate.dense.register_forward_hook(get_activation(f'student.intermediate{i}'))
            if config.output_to_kd:
                teacher.vit.encoder.layer[i].output.dense.register_forward_hook(get_activation(f'teacher.output{i}'))
                model.vit.encoder.layer[i].output.dense.register_forward_hook(get_activation(f'student.output{i}'))

if config.store_attention_output_norm:
    for i in range(12):
        model.vit.encoder.layer[i].attention.output.dense.register_forward_hook(get_activation(f"model.attention_output{i}"))

if config.store_output_norm:
    for i in range(12):
        model.vit.encoder.layer[i].output.dense.register_forward_hook(get_activation(f"model.output{i}"))

if config.store_attention_input_gradient_norm:
    model.vit.embeddings.register_forward_hook(get_activation(f"model.attention{0}.in"))
    for i in range(1, 12):
        model.vit.encoder.layer[i-1].output.register_forward_hook(get_activation(f"model.attention{i}.in"))

if config.store_intermediate_input_gradient_norm:
    for i in range(12):
        model.vit.encoder.layer[i].attention.output.register_forward_hook(get_activation(f"model.intermediate{i}.in"))

# freeze patch embeddings
for params in model.vit.embeddings.parameters():
    params.requires_grad = not config.freeze_patch_embeddings

# freeze layers
for i, l in enumerate(model.vit.encoder.layer):
    if i < config.freezed_blocks_num:
        for params in l.attention.attention.query.parameters():
            params.requires_grad = not config.freeze_query
        for params in l.attention.attention.key.parameters():
            params.requires_grad = not config.freeze_key
        for params in l.attention.attention.value.parameters():
            params.requires_grad = not config.freeze_value
        for params in l.attention.output.dense.parameters():
            params.requires_grad = not config.freeze_attention_output
        for params in l.intermediate.dense.parameters():
            params.requires_grad = not config.freeze_intermediate
        for params in l.output.dense.parameters():
            params.requires_grad = not config.freeze_output

# set dropout
for l in model.vit.encoder.layer:
    l.attention.attention.dropout.p = config.attention_dropout_prob
    l.attention.output.dropout.p = config.output_dropout_prob

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
if config.kd_loss == "mse":
    kd_criterion = nn.MSELoss()
elif config.kd_loss == "cosine":
    kd_criterion = nn.CosineEmbeddingLoss()
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# Prepare model pruner
pruner = Pruner(prune_layers, config.prune_granularity, config.prune_ratio, config.prune_lambda_w)
quantizer = Quantizer(quantize_layers)


# Test function

def test():
    # Test the model
    correct = 0
    total = 0
    model.eval()
    pruner.prune_model(0.0)
    quantizer.quantize_model(0.0)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    quantizer.restore_model()
    pruner.restore_model()

    print(f'Accuracy of the network on the {len(testloader)*config.batch_size} test images: {(100 * correct / total)}%')

print(model)

# If test flag is set, run test and exit
if args.test:
    test()
    exit()

# Train the model

writer = SummaryWriter(log_dir=tb_path)

start_time = datetime.now()
last_time = start_time

progressless_epochs = 0
best_val_loss = 100000
total_train_steps = config.total_epochs*len(trainloader)
lr_warmup_steps = config.lr_warmup_epochs*len(trainloader)

prune_vcon_steps = config.prune_vcon_epochs*len(trainloader)//config.batch_accumulation
prune_vcon_steps_remaining = prune_vcon_steps
quantize_vcon_steps = config.quantize_vcon_epochs*len(trainloader)//config.batch_accumulation
quantize_vcon_steps_remaining = quantize_vcon_steps

lrd_vcon_steps=config.lrd_vcon_epochs*len(trainloader)//config.batch_accumulation
lrd_vcon_steps_remaining=lrd_vcon_steps
if prune_vcon_steps > 0:
    prune_beta = 1
else:
    prune_beta = 0
if quantize_vcon_steps > 0:
    quantize_beta = 1
else:
    quantize_beta = 0
if lrd_vcon_steps > 0:
    lrd_beta = 1
else:
    lrd_beta = 0


optimizer.zero_grad()
if config.from_teacher:
    teacher.eval()

for epoch in range(config.total_epochs):    
    # prepare lists to store activations norms
    if config.store_attention_output_norm:
        for i in range(12):
            train_attention_output_norm_mean = [0.0 for i in range(12)]
            train_attention_output_norm_var = [0.0 for i in range(12)]
            val_attention_output_norm_mean = [0.0 for i in range(12)]
            val_attention_output_norm_var = [0.0 for i in range(12)]

    if config.store_output_norm:
        for i in range(12):
            train_output_norm_mean = [0.0 for i in range(12)]
            train_output_norm_var = [0.0 for i in range(12)]
            val_output_norm_mean = [0.0 for i in range(12)]
            val_output_norm_var = [0.0 for i in range(12)]

    if config.store_attention_input_gradient_norm:
        for i in range(12):
            train_attention_input_gradient_norm_mean = [0.0 for i in range(12)]
            train_attention_input_gradient_norm_var = [0.0 for i in range(12)]

    if config.store_intermediate_input_gradient_norm:
        for i in range(12):
            train_intermediate_input_gradient_norm_mean = [0.0 for i in range(12)]
            train_intermediate_input_gradient_norm_var = [0.0 for i in range(12)]

    # train for one epoch
    running_loss = 0.0
    running_activations_loss = 0.0
    running_output_loss = 0.0
    correct = 0
    pruner.reset_SAD()
    total = 0
    model.train()
    for i, data in enumerate(trainloader, 0):
        # evaluate current training step
        train_step = epoch*len(trainloader) + i

        # learning rate warmup (linear increase from lr_warmup_start to learning_rate for lr_warmup_epochs epochs)
        if train_step <= lr_warmup_steps:
            optimizer.param_groups[0]['lr'] = train_step*(config.learning_rate-config.lr_warmup_start)/lr_warmup_steps+config.lr_warmup_start

        # apply learning rate cosine decay
        if train_step > config.lr_warmup_epochs and config.lr_cosine_decay:
            optimizer.param_groups[0]['lr'] = config.lr_minimum + 0.5 * (config.learning_rate - config.lr_minimum) * (1 + np.cos(np.pi * train_step/total_train_steps))

        # prepare input data
        inputs, labels = data[0].to(device), data[1].to(device)
        tmp_bsize = inputs.shape[0]
        if config.use_cutmix_mixup:
            inputs, labels = cutmix_or_mixup(inputs, labels)

        pruner.prune_model(prune_beta)
        quantizer.quantize_model(quantize_beta)
        lowranker.upd_beta(lrd_beta)
        # forward pass
        if config.from_teacher:
            outputs = model(inputs)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            activations_loss = 0
            for j in range(10):
                if config.kd_loss == "cosine":
                    if config.query_to_kd:
                        activations_loss += kd_criterion(activations[f"student.query{j}"].flatten(0, 1), activations[f"teacher.query{j}"].flatten(0, 1), torch.ones(tmp_bsize*197, device=device))
                    if config.key_to_kd:
                        activations_loss += kd_criterion(activations[f"student.key{j}"].flatten(0, 1), activations[f"teacher.key{j}"].flatten(0, 1), torch.ones(tmp_bsize*197, device=device))
                    if config.value_to_kd:
                        activations_loss += kd_criterion(activations[f"student.value{j}"].flatten(0, 1), activations[f"teacher.value{j}"].flatten(0, 1), torch.ones(tmp_bsize*197, device=device))
                    if config.attention_output_to_kd:
                        activations_loss += kd_criterion(activations[f"student.attention_output{j}"].flatten(0, 1), activations[f"teacher.attention_output{j}"].flatten(0, 1), torch.ones(tmp_bsize*197, device=device))
                    if config.intermediate_to_kd:
                        activations_loss += kd_criterion(activations[f"student.intermediate{j}"].flatten(0, 1), activations[f"teacher.intermediate{j}"].flatten(0, 1), torch.ones(tmp_bsize*197, device=device))
                    if config.output_to_kd:
                        activations_loss += kd_criterion(activations[f"student.output{j}"].flatten(0, 1), activations[f"teacher.output{j}"].flatten(0, 1), torch.ones(tmp_bsize*197, device=device))
                else:
                    if config.query_to_kd:
                        activations_loss += kd_criterion(activations[f"student.query{j}"], activations[f"teacher.query{j}"])
                    if config.key_to_kd:
                        activations_loss += kd_criterion(activations[f"student.key{j}"], activations[f"teacher.key{j}"])
                    if config.value_to_kd:
                        activations_loss += kd_criterion(activations[f"student.value{j}"], activations[f"teacher.value{j}"])
                    if config.attention_output_to_kd:
                        activations_loss += kd_criterion(activations[f"student.attention_output{j}"], activations[f"teacher.attention_output{j}"])
                    if config.intermediate_to_kd:
                        activations_loss += kd_criterion(activations[f"student.intermediate{j}"], activations[f"teacher.intermediate{j}"])
                    if config.output_to_kd:
                        activations_loss += kd_criterion(activations[f"student.output{j}"], activations[f"teacher.output{j}"])
            
            # output loss
            output_loss = criterion(outputs.logits, labels)
            loss = activations_loss + output_loss
        else:
            outputs = model(inputs)
            output_loss = criterion(outputs.logits, labels)
            loss = output_loss

        # get predictions
        if config.use_cutmix_mixup:
            _, maxlabels = torch.max(labels, 1)
        _, predicted = torch.max(outputs.logits, 1)

        # prepare tensor with gradients for storing
        if config.store_attention_input_gradient_norm:
            for j in range(12):
                activations[f"model.attention{j}.in"].retain_grad()
        if config.store_intermediate_input_gradient_norm:
            for j in range(12):
                activations[f"model.intermediate{j}.in"].retain_grad()

        # backward pass
        loss.backward()

        # restore pruned weights
        quantizer.restore_model()
        pruner.restore_model()

        # evaluate norms of the gradients
        if config.store_attention_input_gradient_norm:
            for j in range(12):
                norm = torch.norm(activations[f"model.attention{j}.in"].grad, p=2, dim=-1)
                train_attention_input_gradient_norm_mean[j] = (train_attention_input_gradient_norm_mean[j]*i+torch.mean(norm).detach().cpu().numpy())/(i+1)
                train_attention_input_gradient_norm_var[j] = (train_attention_input_gradient_norm_var[j]*i+torch.mean(torch.square(norm-train_attention_input_gradient_norm_mean[j])).detach().cpu().numpy())/(i+1)
                del norm
        if config.store_intermediate_input_gradient_norm:
            for j in range(12):
                norm = torch.norm(activations[f"model.intermediate{j}.in"].grad, p=2, dim=-1)
                train_intermediate_input_gradient_norm_mean[j] = (train_intermediate_input_gradient_norm_mean[j]*i+torch.mean(norm).detach().cpu().numpy())/(i+1)
                train_intermediate_input_gradient_norm_var[j] = (train_intermediate_input_gradient_norm_var[j]*i+torch.mean(torch.square(norm-train_intermediate_input_gradient_norm_var[j])).detach().cpu().numpy())/(i+1)
                del norm

        pruner.regularize_SAD()

        # optimize
        if (i+1) % config.batch_accumulation == 0:
            if config.clipping_global_norm:
                # apply gradient clipping with maximum norm = 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            optimizer.zero_grad()


        # evaluate norm of the activations
        if config.store_attention_output_norm:
            for j in range(12):
                norm = torch.norm(activations[f"model.attention_output{j}"], p=2, dim=-1)
                train_attention_output_norm_mean[j] = (train_attention_output_norm_mean[j]*i+torch.mean(norm).detach().cpu().numpy())/(i+1)
                train_attention_output_norm_var[j] = (train_attention_output_norm_var[j]*i+torch.mean(torch.square(norm-train_attention_output_norm_mean[j])).detach().cpu().numpy())/(i+1)
                del norm
        if config.store_output_norm:
            for j in range(12):
                norm = torch.norm(activations[f"model.output{j}"], p=2, dim=-1)
                train_output_norm_mean[j] = (train_output_norm_mean[j]*i+torch.mean(norm).detach().cpu().numpy())/(i+1)
                train_output_norm_var[j] = (train_output_norm_var[j]*i+torch.mean(torch.square(norm-train_output_norm_mean[j])).detach().cpu().numpy())/(i+1)
                del norm

        # evaluate current loss and accuracy
        running_loss += loss.item()
        if config.from_teacher:
            running_activations_loss += activations_loss.item()
        running_output_loss += output_loss.item()
        total += labels.size(0)
        if config.use_cutmix_mixup:
            correct += (predicted == maxlabels).sum().item()
        else:
            correct += (predicted == labels).sum().item()
        
        train_accuracy = (100 * correct / total)
        train_loss = running_loss * config.batch_size / total
        if config.from_teacher:
            train_activations_loss = running_activations_loss * config.batch_size / total
        train_output_loss = running_output_loss * config.batch_size / total

        # evaluate SAD
        pruner.update_SAD()

        # apply vanishing contributions (only when update is performed)
        if (i+1) % config.batch_accumulation == 0:
            if prune_vcon_steps_remaining > 0:
                prune_vcon_steps_remaining -= 1
                prune_beta -= 1/prune_vcon_steps
            else:
                prune_beta = 0
            if quantize_vcon_steps_remaining > 0:
                quantize_vcon_steps_remaining -= 1
                quantize_beta -= 1/quantize_vcon_steps
            else:
                quantize_beta = 0
            if lrd_vcon_steps_remaining > 0:
                lrd_vcon_steps_remaining -= 1
                lrd_beta -= 1/lrd_vcon_steps
            else:
                lrd_vcon_steps_beta = 0


        if config.from_teacher:
            print(f'[{epoch+1}, {i+1:4d}] activations_loss: {train_activations_loss:1.3e} output_loss: {train_output_loss:1.3e} total_loss: {train_loss:1.3e} acc: {train_accuracy:3.2f}%', end="\r")
        else:
            print(f'[{epoch+1}, {i+1:4d}] output_loss: {train_output_loss:1.3e} acc: {train_accuracy:3.2f}%', end="\r")
    if config.from_teacher:
        print(f'[{epoch+1}, {i+1:4d}] activations_loss: {train_activations_loss:1.3e} output_loss: {train_output_loss:1.3e} total_loss: {train_loss:1.3e} acc: {train_accuracy:3.2f}%')
    else:
        print(f'[{epoch+1}, {i+1:4d}] output_loss: {train_output_loss:1.3e} acc: {train_accuracy:3.2f}%')
            
    # validation
    running_loss = 0.0
    correct = 0
    total = 0
    validation_SAD = 0
    model.eval()
    pruner.prune_model(prune_beta)
    quantizer.quantize_model(quantize_beta)
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            _, predicted = torch.max(outputs.logits, 1)
            

            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            validation_accuracy = (100 * correct / total)
            validation_loss = running_loss * config.batch_size / total

            # evaluate norm of the activations
            if config.store_attention_output_norm:
                for j in range(12):
                    norm = torch.norm(activations[f"model.attention_output{j}"], p=2, dim=-1)
                    val_attention_output_norm_mean[j] = (val_attention_output_norm_mean[j]*i+torch.mean(norm).detach().cpu().numpy())/(i+1)
                    val_attention_output_norm_var[j] = (val_attention_output_norm_var[j]*i+torch.mean(torch.square(norm-val_attention_output_norm_mean[j])).detach().cpu().numpy())/(i+1)
                    del norm
            if config.store_output_norm:
                for j in range(12):
                    norm = torch.norm(activations[f"model.output{j}"], p=2, dim=-1)
                    val_output_norm_mean[j] = (val_output_norm_mean[j]*i+torch.mean(norm).detach().cpu().numpy())/(i+1)
                    val_output_norm_var[j] = (val_output_norm_var[j]*i+torch.mean(torch.square(norm-val_output_norm_mean[j])).detach().cpu().numpy())/(i+1)
                    del norm

            print(f'[{epoch+1}, {i+1:4d}] loss: {validation_loss:1.3e} acc: {validation_accuracy:3.2f}%', end="\r")
        print(f'[{epoch+1}, {i+1:4d}] loss: {validation_loss:1.3e} acc: {validation_accuracy:3.2f}%')
    quantizer.restore_model()
    pruner.restore_model()
    
    # store info on tensorboard
    writer.add_scalar('Loss/train', train_loss, config.starting_epoch+epoch+1)
    writer.add_scalar('Accuracy/train', train_accuracy, config.starting_epoch+epoch+1)
    writer.add_scalar('Loss/validation', validation_loss, config.starting_epoch+epoch+1)
    writer.add_scalar('Accuracy/validation', validation_accuracy, config.starting_epoch+epoch+1)
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], config.starting_epoch+epoch+1)
    if config.store_attention_output_norm:
        for j in range(12):
            writer.add_scalar(f'Attention output act norm mean (train)/layer{j}', train_attention_output_norm_mean[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Attention output act norm var (train)/layer{j}', train_attention_output_norm_var[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Attention output act norm mean (val)/layer{j}', val_attention_output_norm_mean[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Attention output act norm var (val)/layer{j}', val_attention_output_norm_var[j], config.starting_epoch+epoch+1)
    if config.store_output_norm:
        for j in range(12):
            writer.add_scalar(f'Output act norm mean (train)/layer{j}', train_output_norm_mean[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Output act norm var (train)/layer{j}', train_output_norm_var[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Output act norm mean (val)/layer{j}', val_output_norm_mean[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Output act norm var (val)/layer{j}', val_output_norm_var[j], config.starting_epoch+epoch+1)
    if config.store_attention_input_gradient_norm:
        for j in range(12):
            writer.add_scalar(f'Attention in grad norm mean (train)/layer{j}', train_attention_input_gradient_norm_mean[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Attention in grad norm var (train)/layer{j}', train_attention_input_gradient_norm_var[j], config.starting_epoch+epoch+1)
    if config.store_intermediate_input_gradient_norm:
        for j in range(12):
            writer.add_scalar(f'Intermediate in grad norm mean (train)/layer{j}', train_intermediate_input_gradient_norm_mean[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Intermediate in grad norm var (train)/layer{j}', train_intermediate_input_gradient_norm_var[j], config.starting_epoch+epoch+1)

    if prune_layers != []:
        writer.add_scalar('SAD', pruner.get_SAD(), epoch+1)
    
    # update learning rate on plateau
    if epoch > config.lr_warmup_epochs:
        if validation_loss > best_val_loss:
            progressless_epochs += 1
        else:
            best_val_loss = validation_loss
        if progressless_epochs > config.lr_patience:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*config.lr_decay_factor
            if optimizer.param_groups[0]['lr'] < config.lr_minimum:
                optimizer.param_groups[0]['lr'] = config.lr_minimum
            progressless_epochs = 0
    
    print(f'Epoch time: {datetime.now()-last_time}', end=" ")
    print(f'Running time: {datetime.now()-start_time}')
    last_time = datetime.now()

    # backup model/store final model
    torch.save(model.state_dict(), model_path)
    
stop_time = datetime.now()
    
print('Training Complete')
print(f'Total time: {str(stop_time-start_time)}')

test() # Test the model after training