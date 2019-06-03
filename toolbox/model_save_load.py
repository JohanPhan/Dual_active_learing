import cloudpickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
def load_network(Model, Model_state_dict):
    Model.load_state_dict(Model_state_dict)
    Model.eval()
    return Model
def load_optimizer(Optimizer, Optimizer_state_dict):
    Optimizer.load_state_dict(Optimizer_state_dict)
    return Optimizer
def load_selected(name):
    f = open(name + '.cloudpickle', 'rb')
    f.close
    f = open(name + '.cloudpickle', 'rb')
    file = cloudpickle.load(f)
    f.close
    return file
def save_selected(name, file):
    f = open(name + '.cloudpickle', 'wb')
    cloudpickle.dump(file, f)
    f.close
def load_checkpoint(net, FC, optimizer, Path):
    checkpoint = torch.load(Path)
    net.load_state_dict(checkpoint['net_state_dict'])
    FC.load_state_dict(checkpoint['FC_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    net.eval()
    FC.eval()
    return net, FC, optimizer
    
def remove_selected_item(data_set, selected_item):
    temp_set = list(filter(lambda x: x not in selected_item, data_set))
    return temp_set