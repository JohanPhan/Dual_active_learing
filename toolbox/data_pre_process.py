import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.utils.model_zoo as model_zoo
from model/vgg_pretrained import vgg16_pretrain 
def get_representation_matrices(dataset, device):
    net = vgg16_pretrain().to(device)
    data_set_loader = torch.utils.data.DataLoader(dataset, batch_size=100, 
                                          shuffle=False, num_workers=3)
    output_tensor = None
    if_empty = True
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_set_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs) 
            outputs = outputs.cpu()
            if if_empty == False:
                output_tensor = torch.cat((output_tensor, outputs), 0)
            else:
                output_tensor = outputs
                if_empty = False
    return(output_tensor)




