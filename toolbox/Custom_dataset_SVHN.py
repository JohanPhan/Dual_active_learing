from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size = 32, scale = (0.5, 1.0)),
    transforms.ColorJitter(hue=.05, saturation=.1, brightness = 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class L_set_SV(Dataset):


    def __init__(self, dataset, labelset, transform=transform_train):
        
        self.dataset = dataset
        self.labelset = labelset
        self.transform = transform
        self.count = 0
        self.query_set = []
        self.query_label = []        
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
#         img = self.dataset[index]
#         img = Image.fromarray(img)
#         target = int(self.labelset[index])

        
#         if self.transform is not None:
#             img = self.transform(img)   

#         sample = (img, target)
#         return sample        
        img, target = self.dataset[index], self.labelset[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
        return img, target
    def update(self, in_data, in_label):
        
        self.dataset.append(in_data)
        self.labelset.append(in_label)
    def clean(self, length):
        for i in range(length):
            self.dataset.pop()
            self.labelset.pop() 
class LSTM_set(Dataset):
    def __init__(self, dataset, labelset, transform=transform_train):
        
        self.dataset = dataset
        self.targetset = labelset
        self.transform = transform
        self.count = 0
        self.query_set = []
        self.query_label = []        
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
   
        img_set, target = self.dataset[index], self.targetset[index]
        

        for i in range(len(img_set)):
            temp_img = Image.fromarray(img_set[i])
            if self.transform is not None:
                temp_img = self.transform(temp_img)
            print(temp_img.shape)   
            img_set[i] = temp_img
        
        return return_set, target
    def update_data(self, in_data):
        
        self.dataset.append(in_data)
    def update_target(self, target):
        self.targetset.append(target)
    def clean(self, length):
        for i in range(length):
            self.dataset.pop()
            self.targetset.pop()    
        