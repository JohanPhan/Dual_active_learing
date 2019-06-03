import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split




def load_trainset(**kwargs):
    default_transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size = 28, scale = (0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = kwargs.get('dataset',datasets.MNIST)
    root_arg = kwargs.get('root_dir','./data')
    transform_arg = kwargs.get('transform_train', default_transform_train)
    shuffle_arg = kwargs.get('shuffle',True)
    batch_size_arg = kwargs.get('batch_size',100)
    num_workers_arg = kwargs.get('num_workers', 1)
    trainset = dataset(root= root_arg, train= True,
                                            download=True, 
                                            transform= transform_arg)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_arg,
                                              shuffle= shuffle_arg, 
                                              num_workers=num_workers_arg)
    return trainset, trainloader
def load_testset(**kwargs):
    
    default_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset = kwargs.get('dataset',datasets.MNIST)
    root_arg = kwargs.get('root_dir','./data')
    transform_arg = kwargs.get('transform_test', default_transform_test)
    batch_size_arg = kwargs.get('batch_size', 100)
    shuffle_arg = kwargs.get('shuffle',False)
    num_workers_arg = kwargs.get('num_workers', 1)
    
   
    testset = dataset(root= root_arg, train=False,
                                           download=True, 
                                            transform=transform_arg)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=batch_size_arg,
                                             shuffle= shuffle_arg, 
                                             num_workers=num_workers_arg)
    return testset, testloader